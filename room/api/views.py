from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from django.core.cache import cache
import uuid

# Shared with the websocket consumers so REST and sockets use the SAME cache key
# (session:<id>:messages) and de-duplicate by client_id.
from room.consumers import _cache_get_messages, _cache_append_message, session_messages_key
from room.models import Message, Room
from .serializers import MessageSerializer
from django.utils import timezone
from django.http import HttpResponse
import os
import google.generativeai as genai
import json

# SECURITY: no hardcoded fallback key (the old one was public on GitHub — revoke it).
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

from room.access import get_accessible_room, user_can_access_room


class RoomDetailView(APIView):
    """
    Returns the list of usernames in a room — only to members of that room.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, room_name):
        allowed, room = get_accessible_room(request.user, room_name, create_direct=True)
        if not allowed:
            return Response({"error": "Room not found"}, status=status.HTTP_404_NOT_FOUND)
        if room is None:
            return Response({"room": room_name, "users": [], "is_active": True, "match_reason": None})

        usernames = list(room.users.values_list('username', flat=True))
        return Response({
            "room": room_name, 
            "users": usernames,
            "is_active": room.is_active,
            "match_reason": room.match_reason
        })

class RoomStatusView(APIView):
    """
    Lightweight check to see if a room is still active.
    Used by frontend polling fallback for synchronized exit.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, room_name):
        allowed, room = get_accessible_room(request.user, room_name)
        if not allowed:
            return Response({"is_active": False})
        if room is None:
            # direct room not created yet / ephemeral session with no DB row
            return Response({"is_active": True})
        return Response({"is_active": room.is_active})

class RoomCloseView(APIView):
    """
    Sets a room's is_active status to False.
    Called when a user confirms severance of the connection.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, room_name):
        allowed, room = get_accessible_room(request.user, room_name)
        if not allowed or room is None:
            return Response({"error": "Room not found"}, status=status.HTTP_404_NOT_FOUND)
        room.is_active = False
        room.save(update_fields=['is_active'])
        return Response({"status": "closed"})

class ConversationListView(APIView):
    """
    Returns all rooms the user has participated in, 
    formatted as an Inbox/Conversation list.
    SECURITY: always the logged-in user's own inbox (the `username` query param is ignored).
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user

        rooms = user.rooms.all().order_by('-created_at')
        data = []
        for room in rooms:
            partner = room.users.exclude(id=user.id).first()
            last_message = Message.objects.filter(room=room).order_by('-date').first()
            
            data.append({
                "room_name": room.name,
                "partner_username": partner.username if partner else "Disconnected Node",
                "partner_image": partner.profile.persona_image_url if partner and hasattr(partner, 'profile') else None,
                "last_message": last_message.value[:50] if last_message else "No messages yet",
                "last_message_time": last_message.date.strftime("%Y-%m-%d %H:%M") if last_message else None,
                "is_active": room.is_active
            })
        return Response(data)


class MessageActionView(APIView):
    """
    Handle individual message actions: mark read, delete for me, delete for everyone.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, message_id):
        try:
            message = Message.objects.get(id=message_id)
        except Message.DoesNotExist:
            return Response({"error": "Message not found"}, status=status.HTTP_404_NOT_FOUND)

        action = request.data.get('action')
        
        # Security: User must be part of the room
        if request.user not in message.room.users.all():
            return Response({"error": "Unauthorized"}, status=status.HTTP_403_FORBIDDEN)

        import re
        safe_room_name = re.sub(r'[^a-zA-Z0-9_\-]', '', message.room.name)
        channel_layer = get_channel_layer()

        if action == 'mark_read':
            if message.user != request.user and not message.is_read:
                message.is_read = True
                message.read_timestamp = timezone.now()
                message.save()
                
                # Broadcast read receipt
                async_to_sync(channel_layer.group_send)(
                    f'chat_{safe_room_name}',
                    {
                        'type': 'message_read',
                        'id': message.id
                    }
                )

            return Response({"status": "read"})

        elif action == 'delete_for_me':
            if message.user == request.user:
                message.deleted_for_sender = True
                message.save()
            return Response({"status": "deleted_for_me"})

        elif action == 'delete_for_everyone':
            if message.user == request.user:
                message.deleted_for_everyone = True
                message.deleted_timestamp = timezone.now()
                message.save()
                
                # Broadcast deletion
                async_to_sync(channel_layer.group_send)(
                    f'chat_{safe_room_name}',
                    {
                        'type': 'message_deleted',
                        'id': message.id
                    }
                )

            return Response({"status": "deleted_for_everyone"})

        return Response({"error": "Invalid action"}, status=status.HTTP_400_BAD_REQUEST)

class TranscriptDownloadView(APIView):
    """
    Generates a downloadable text transcript of a chat room.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, room_name):
        try:
            room = Room.objects.get(name=room_name)
        except Room.DoesNotExist:
            return Response({"error": "Room not found"}, status=status.HTTP_404_NOT_FOUND)

        # Check if user is in room
        if request.user not in room.users.all():
            return Response({"error": "Unauthorized to access this transcript"}, status=status.HTTP_403_FORBIDDEN)

        messages = Message.objects.filter(room=room).order_by('date')
        
        transcript_lines = [f"Transcript for Chat: {room_name}", "="*40, ""]
        
        for msg in messages:
            timestamp = msg.date.strftime("%Y-%m-%d %H:%M:%S")
            sender = msg.user.username
            content = msg.value
            
            # Indicate deletion in transcript to the user requesting it
            if msg.deleted_for_everyone:
                content = "[Message deleted]"
            elif msg.deleted_for_sender and request.user == msg.user:
                continue 
                
            transcript_lines.append(f"[{timestamp}] {sender}: {content}")

        text_content = "\n".join(transcript_lines)
        
        response = HttpResponse(text_content, content_type='text/plain')
        response['Content-Disposition'] = f'attachment; filename="transcript_{room_name}.txt"'
        return response


class ChatAnalysisView(APIView):
    """
    Passive Chat Analysis Engine.
    Triggers the background Celery task to analyze recent messages in a room 
    and update both participants' profiles.
    Returns immediately.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, room_name):
        try:
            room = Room.objects.get(name=room_name)
        except Room.DoesNotExist:
            return Response({"error": "Room not found"}, status=status.HTTP_404_NOT_FOUND)

        if request.user not in room.users.all():
            return Response({"error": "Unauthorized"}, status=status.HTTP_403_FORBIDDEN)

        # Trigger Celery Task asynchronously
        from room.tasks import generate_chat_analysis_task
        generate_chat_analysis_task.delay(room_name)

        return Response({
            "message": "Chat analysis started in the background."
        }, status=status.HTTP_202_ACCEPTED)


class MessageListView(APIView):
    """
    GET: Retrieve recent messages for a room.
    POST: Send a new message to the room.
    SECURITY: requires login; only members of the room can read or post, and the sender
    is always the logged-in user (any `username` sent by the client is ignored).
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, room_name):
        allowed, room = get_accessible_room(request.user, room_name)
        if not allowed:
            return Response({"error": "Room not found"}, status=status.HTTP_404_NOT_FOUND)

        requesting_user = request.user
        
        # Pagination params
        cursor = request.query_params.get('cursor') # timestamp or ID
        limit = 50

        # Case 1: Session Room (Ephemeral messages in Redis)
        if room_name.startswith('session_'):
            session_id = room_name.replace('session_', '')
            from django.core.cache import cache
            
            raw_msgs = cache.get(session_messages_key(session_id)) or '[]'
            try:
                msg_list = json.loads(raw_msgs)
            except:
                msg_list = []
                
            raw_calls = cache.get(f'session:{session_id}:calls') or '[]'
            try:
                call_list = json.loads(raw_calls)
            except:
                call_list = []

            # We don't need to manually merge call_list visually because we already inject visual call placeholders into messages list!
            # So the visual messages list is enough for chat UI.
            
            all_msgs = msg_list
            
            # Sort by timestamp (ISO strings sort correctly)
            all_msgs.sort(key=lambda x: x.get('timestamp', x.get('ts', '')))
            
            # Simple offset-based pagination for Redis list for now, 
            # as it's typically short-lived and doesn't have DB IDs
            # If cursor is provided, it's an ISO timestamp
            if cursor:
                all_msgs = [m for m in all_msgs if m.get('timestamp', m.get('ts', '')) < cursor]
            
            # Take last {limit}
            page_msgs = all_msgs[-limit:]
            has_more = len(all_msgs) > limit
            
            data = []
            for msg in page_msgs:
                data.append({
                    "id": msg.get('id', msg.get('ts')), # Use UUID or timestamp
                    "sender": msg.get('sender', msg.get('sender_id')),
                    "text": msg.get('text'),
                    "status": msg.get('status', 'read'), 
                    "timestamp": msg.get('timestamp', msg.get('ts')),
                    "is_ephemeral": True,
                    "is_call_log": msg.get('is_call_log', False),
                    "call_mode": msg.get('call_mode'),
                    "call_status": msg.get('call_status'),
                    "call_duration": msg.get('call_duration'),
                })
            
            next_cursor = page_msgs[0].get('ts') if page_msgs else None
            return Response({
                "results": data,
                "has_more": has_more,
                "next_cursor": next_cursor
            })

        # Case 2: Friend/Room Chat (PostgreSQL)
        if not room:
            return Response({"results": [], "has_more": False})

        messages = Message.objects.filter(room=room).order_by('-date')
        
        if cursor:
            messages = messages.filter(id__lt=cursor)
        
        # Apply limit + 1 to check for has_more
        page_msgs = list(messages[:limit + 1])
        has_more = len(page_msgs) > limit
        if has_more:
            page_msgs = page_msgs[:limit]
        
        # Reverse to get chronological order for the client
        page_msgs.reverse()
        
        # Robust CallLog fetching for two-party rooms
        calls = []
        from calls.models import CallLog
        from django.db import models
        
        # Two-person rooms (direct or session): use the room's real members instead of
        # splitting the name on "_" (which broke for usernames containing "_").
        if room and room.users.count() == 2:
            room_users = list(room.users.all())
            user1, user2 = room_users[0], room_users[1]
            calls_qs = CallLog.objects.filter(
                models.Q(caller=user1, receiver=user2) | 
                models.Q(caller=user2, receiver=user1)
            ).order_by('-created_at')[:limit]
            calls = list(calls_qs)
            calls.reverse()

        data = []
        for msg in page_msgs:
            if requesting_user and msg.deleted_for_sender and msg.user == requesting_user:
                continue
            
            text = msg.value
            if msg.deleted_for_everyone:
                text = "This message was deleted."
                
            data.append({
                "id": msg.id,
                "sender": msg.user.username,
                "text": text,
                "status": msg.status,
                "isRead": msg.status == 'read',
                "deletedForEveryone": msg.deleted_for_everyone,
                "timestamp": msg.date.strftime("%I:%M %p"),
                "date_iso": msg.date.isoformat(),
                "is_ephemeral": False,
                "is_call_log": False
            })
        
        if data:
            s_msg = data[0]
            print(f"DEBUG: [MessageListView] Room: {room_name}, Returning {len(data)} msgs. Latest msg from {s_msg['sender']} has status: {s_msg['status']}")

        for call in calls:
            data.append({
                "id": str(call.call_id),
                "sender": call.caller.username if call.caller else 'Unknown',
                "text": "",
                "status": "read", # Force read for call logs visually
                "isRead": True,
                "deletedForEveryone": False,
                "timestamp": call.created_at.strftime("%I:%M %p"),
                "date_iso": call.created_at.isoformat(),
                "is_ephemeral": False,
                "is_call_log": True,
                "call_mode": call.mode,
                "call_status": call.status,
                "call_duration": call.duration_seconds
            })
            
        # Re-sort combined data by ISO timestamp
        data.sort(key=lambda x: x.get('date_iso', ''))

        next_cursor = page_msgs[0].id if page_msgs else None

        return Response({
            "results": data,
            "has_more": has_more,
            "next_cursor": next_cursor
        }, status=status.HTTP_200_OK)

    def post(self, request, room_name):
        allowed, room = get_accessible_room(request.user, room_name, create_direct=True)
        if not allowed:
            return Response({"error": "Room not found"}, status=status.HTTP_404_NOT_FOUND)
        requesting_user = request.user

        text = request.data.get('text')
        if not text:
            return Response({"error": "Message text is required"}, status=status.HTTP_400_BAD_REQUEST)

        import datetime
        timestamp = datetime.datetime.now().strftime("%I:%M %p")

        if room_name.startswith('session_'):
            # EPHEMERAL: Save to Redis
            msg_id = f"eph_{uuid.uuid4().hex[:8]}"
            msg_data = {
                "id": msg_id,
                "sender": requesting_user.username,
                "text": text,
                "status": "read",
                "isRead": True,
                "deletedForEveryone": False,
                "timestamp": timestamp,
                "is_ephemeral": True,
                "client_id": request.data.get('client_id')
            }
            _cache_append_message(room_name, msg_data)

            # Broadcast
            try:
                import re
                safe_room_name = re.sub(r'[^a-zA-Z0-9_\-]', '', room_name)
                channel_layer = get_channel_layer()
                if channel_layer:
                    async_to_sync(channel_layer.group_send)(
                        f'chat_{safe_room_name}',
                        {
                            'type': 'chat_message',
                            'id': msg_id,
                            'message': text,
                            'username': requesting_user.username,
                            'timestamp': timestamp,
                            'isRead': True,
                            'deletedForEveryone': False,
                            'client_id': request.data.get('client_id')
                        }
                    )
            except: pass

            return Response(msg_data, status=status.HTTP_201_CREATED)

        if room is None:
            return Response({"error": "Room not found"}, status=status.HTTP_404_NOT_FOUND)

        msg = Message.objects.create(
            user=requesting_user,
            room=room,
            value=text
        )

        # Broadcast via Channels (best effort — won't fail the request if Redis is down)
        try:
            import re
            safe_room_name = re.sub(r'[^a-zA-Z0-9_\-]', '', room_name)
            channel_layer = get_channel_layer()
            if channel_layer:
                async_to_sync(channel_layer.group_send)(
                    f'chat_{safe_room_name}',
                    {
                        'type': 'chat_message',
                        'id': msg.id,
                        'message': msg.value,
                        'username': msg.user.username,
                        'timestamp': msg.date.strftime("%I:%M %p"),
                        'isRead': msg.is_read,
                        'deletedForEveryone': msg.deleted_for_everyone,
                        'client_id': request.data.get('client_id')
                    }
                )
        except Exception as e:
            pass  # Real-time push failed but message is saved — frontend will poll

        # Let the other person know even if they don't have this chat open
        # (toast + unread notification). Previously they got nothing.
        if room_name.startswith('direct_'):
            try:
                from room.notify import notify_direct_message
                notify_direct_message(requesting_user, room, msg, request.data.get('client_id'))
            except Exception as e:
                print(f"MessageListView: notify failed: {e}")

        return Response({
            "id": msg.id,
            "client_id": request.data.get('client_id'),
            "status": msg.status,
            "sender": msg.user.username,
            "text": msg.value,
            "isRead": msg.is_read,
            "deletedForEveryone": msg.deleted_for_everyone,
            "timestamp": msg.date.strftime("%I:%M %p")
        }, status=status.HTTP_201_CREATED)


