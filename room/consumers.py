"""
room/consumers.py
-----------------
Two WebSocket consumers:

1. NotificationConsumer  ws/notifications/
   - One persistent connection per user
   - Presence, chat-request flow, session messaging, friend messaging,
     typing indicators, and full call signalling (offer → answer → ICE → end)

2. ChatConsumer  ws/chat/<room_name>/
   - Room-scoped connection for an active chat session
   - Room messages, WebRTC signalling forwarding, force-exit, AI fraud monitoring
"""

import json
import asyncio
import uuid
import os
from datetime import datetime, timezone as dt_tz

from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async, async_to_sync


# ─────────────────────────── helpers ────────────────────────────────────────

def _cache():
    from django.core.cache import cache
    return cache

def session_messages_key(room_or_session_id):
    """Single source of truth for the ephemeral session message cache key.
    Accepts either 'session_<id>' (room name) or the bare '<id>'.
    (Before: some writers used session:session_<id>:messages and readers used
    session:<id>:messages, so session history vanished on reload.)"""
    sid = str(room_or_session_id)
    if sid.startswith('session_'):
        sid = sid[len('session_'):]
    return f'session:{sid}:messages'


def _cache_get_messages(room_name):
    import json
    raw = _cache().get(session_messages_key(room_name)) or '[]'
    try:
        return json.loads(raw)
    except Exception:
        return []

def _cache_append_message(room_name, msg_data):
    import json
    msgs = _cache_get_messages(room_name)
    # The frontend sends each message over the socket AND via REST — store it once.
    cid = msg_data.get('client_id')
    if cid and any(m.get('client_id') == cid for m in msgs):
        return
    msgs.append(msg_data)
    _cache().set(session_messages_key(room_name), json.dumps(msgs), timeout=86400)


@sync_to_async
def _cache_get(key):
    return _cache().get(key)


@sync_to_async
def _cache_set(key, val, timeout=None):
    _cache().set(key, val, timeout=timeout)


@sync_to_async
def _cache_delete(key):
    # Returns True only for the caller that actually removed the key (Django >= 3.1),
    # which lets concurrent handlers "claim" a call exactly once.
    return bool(_cache().delete(key))


@sync_to_async
def _cache_get_or_set_list(key, item, timeout=120):
    """Append item to a JSON list stored in cache."""
    raw = _cache().get(key) or '[]'
    lst = json.loads(raw)
    lst.append(item)
    _cache().set(key, json.dumps(lst), timeout=timeout)


@sync_to_async
def _cache_get_list(key):
    raw = _cache().get(key) or '[]'
    return json.loads(raw)


@sync_to_async
def _update_conn_count(key, delta):
    cache = _cache()
    # Atomic-ish update
    val = cache.get(key, 0)
    new_val = max(0, val + delta)
    cache.set(key, new_val, timeout=3600)
    return new_val


@sync_to_async
def _get_user_by_id(user_id):
    from django.contrib.auth.models import User
    try:
        return User.objects.get(pk=user_id)
    except User.DoesNotExist:
        return None


@sync_to_async
def _get_user_by_username(username):
    from django.contrib.auth.models import User
    try:
        return User.objects.get(username=username)
    except User.DoesNotExist:
        return None

@sync_to_async
def _is_user_premium(username):
    from django.contrib.auth.models import User
    try:
        user = User.objects.get(username=username)
        # Tiers are free/weekly/monthly (not plus/premium/elite) — use the model's own check.
        return user.profile.is_premium
    except Exception:
        return False

@sync_to_async
def _check_and_deduct_ai_quota(username):
    """
    Check if the user has remaining AI LLM search quota.
    If so, deduct 1 and return True. Otherwise return False.
    """
    from django.contrib.auth.models import User
    try:
        user = User.objects.get(username=username)
        # Same limits, daily reset and atomic increment as Smart Search (users/quota.py)
        from users import quota
        return quota.consume(user.profile, 'ai')
    except Exception:
        return False


@sync_to_async
def _save_friend_message(sender, room_name, text):
    """Persist a friend (direct) message to PostgreSQL.
    SECURITY: returns None (and saves nothing) unless `sender` is a member of the room."""
    from room.models import Message
    from room.access import get_accessible_room

    if not room_name.startswith('direct_'):
        return None
    allowed, room = get_accessible_room(sender, room_name, create_direct=True)
    if not allowed or room is None:
        return None

    msg = Message.objects.create(user=sender, room=room, value=text)

    # 3. Notify the recipient (same helper the REST send path uses)
    from room.notify import notify_direct_message
    notify_direct_message(sender, room, msg)

    return {
        'id': msg.id,
        'sender': sender.username,
        'text': text,
        'status': msg.status,
        'timestamp': msg.date.strftime('%I:%M %p'),
    }


@sync_to_async
def _set_messages_read(room_name, user):
    """Mark all unread messages in a room as read for the given user (the recipient)."""
    if not user or not user.is_authenticated:
        return False
    from room.models import Message, MSG_READ
    from django.utils import timezone
    # All messages NOT sent by this user in this room are now read
    updated_count = Message.objects.filter(
        room__name=room_name,
        status__in=['sent', 'delivered']
    ).exclude(user_id=user.id).update(
        status=MSG_READ,
        read_timestamp=timezone.now(),
        is_read=True
    )
    
    # Also clear any ephemeral ChatNotifications for this room/recipient
    from matchmaking.models import ChatNotification
    qs = ChatNotification.objects.filter(
        receiver=user,
        room_name=room_name,
        is_read=False
    )
    if qs.exists():
        qs.update(is_read=True)
        # Broadcast a signal to refresh the header count/inbox
        from channels.layers import get_channel_layer
        from asgiref.sync import async_to_sync
        layer = get_channel_layer()
        if layer:
            async_to_sync(layer.group_send)(
                f'user_{user.id}',
                {
                    'type': 'friend_message_recv', # Reuse this event to trigger refresh
                    'conversation_id': room_name,
                    'text': 'clear_notification',
                    'sender': 'system'
                }
            )

    print(f"DEBUG: [_set_messages_read] Room '{room_name}' - User '{user.username}' (ID: {user.id}) marked {updated_count} peer messages as READ and cleared notifs")
    return updated_count > 0


@sync_to_async
def _save_call_log(call_id, caller_username, receiver_username, mode, status, started_at, ended_at, duration_seconds, context_id=None):
    """Persist a call record and inject a log message into the chat."""
    
    # 1. Format the visual message text for the chat UI
    # e.g. "Audio call · 4m 12s" or "Missed call"
    msg_type_str = "Video" if mode == 'video' else "Audio"
    if status == 'ended':
        m = duration_seconds // 60
        s = duration_seconds % 60
        visual_text = f"📞 {msg_type_str} call · {m}m {s}s"
    elif status == 'declined':
        visual_text = f"☎️ Missed {msg_type_str.lower()} call (Declined)"
    elif status in ['no_answer', 'cancelled', 'unavailable']:
        visual_text = f"☎️ Missed {msg_type_str.lower()} call"
    else:
        visual_text = f"☎️ Call {status}"

    from django.core.cache import cache
    import json
    
    # 2. Save log into Chat History
    if context_id and context_id.startswith('session_'):
        # Save to Redis for discovery session
        session_id = context_id.replace('session_', '') # Ensure clean session_id
        
        # A) Visual text for UI
        msg_obj = {
            'id': str(uuid.uuid4()),
            'sender_id': caller_username,  # Frontend expects username in 'sender' usually, but 'sender_id' requested
            'sender': caller_username,     # keeping 'sender' for frontend backward compat
            'text': visual_text,
            'timestamp': ended_at.isoformat() if ended_at else datetime.now().isoformat(),
            'status': 'read',
            'is_call_log': True,
            'call_mode': mode,
            'call_status': status,
            'call_duration': duration_seconds
        }
        raw = cache.get(f'session:{session_id}:messages') or '[]'
        lst = json.loads(raw)
        lst.append(msg_obj)
        cache.set(f'session:{session_id}:messages', json.dumps(lst), timeout=86400)

        # B) Structured Call Log
        call_obj = {
            'call_id': str(call_id),
            'caller_id': caller_username,
            'duration': duration_seconds,
            'status': status,
            'timestamp': ended_at.isoformat() if ended_at else datetime.now().isoformat(),
        }
        raw_calls = cache.get(f'session:{session_id}:calls') or '[]'
        call_lst = json.loads(raw_calls)
        call_lst.append(call_obj)
        cache.set(f'session:{session_id}:calls', json.dumps(call_lst), timeout=86400)
        
    from channels.layers import get_channel_layer
    channel_layer = get_channel_layer()
    
    # Send real-time UI notification via WebSocket group
    if context_id:
        room_group_name = f'chat_{context_id}'
        
        # Determine actual ID for the real-time event
        # If it's a UUID, convert to string
        push_id = str(call_id) if call_id else str(uuid.uuid4())
        
        msg_payload = {
            'id': push_id,
            'sender': caller_username,
            'text': "", # Rendered visually via is_call_log anyway
            'timestamp': ended_at.isoformat() if ended_at else datetime.now().isoformat(),
            'status': 'read',
            'is_call_log': True,
            'call_mode': mode,
            'call_status': status,
            'call_duration': duration_seconds,
            'is_ephemeral': context_id.startswith('session_')
        }
        
        # Send real-time UI notification via WebSocket group
        if 'room_group_name' in locals():
            async_to_sync(channel_layer.group_send)(room_group_name, {
                'type': 'chat_message',
                'id': msg_payload['id'],
                'message': msg_payload['text'],
                'username': msg_payload['sender'],
                'is_call_log': msg_payload['is_call_log'],
                'call_mode': mode,
                'call_status': status,
                'call_duration': duration_seconds,
                'is_ephemeral': msg_payload.get('is_ephemeral', False),
            })

    # 3. Save to PostgreSQL CallLog explicitly
    from calls.models import CallLog
    from django.contrib.auth.models import User
    try:
        caller = User.objects.get(username=caller_username)
    except User.DoesNotExist:
        caller = None
    try:
        receiver = User.objects.get(username=receiver_username)
    except User.DoesNotExist:
        receiver = None

    if receiver and hasattr(receiver, 'profile'):
        # Trust Score Tracking for Receiver
        if status == 'ended':
            receiver.profile.calls_received += 1
            receiver.profile.calls_answered += 1
            receiver.profile.save(update_fields=['calls_received', 'calls_answered'])
            from users.trust import apply_trust_event
            apply_trust_event(receiver.id, 'call_answered', 0, "Answered a call")
        elif status in ['declined', 'no_answer', 'cancelled', 'unavailable']:
            receiver.profile.calls_received += 1
            receiver.profile.save(update_fields=['calls_received'])
            from users.trust import apply_trust_event
            apply_trust_event(receiver.id, 'call_missed', 0, "Missed or declined a call")

    CallLog.objects.create(
        call_id=call_id,
        caller=caller,
        receiver=receiver,
        mode=mode,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=duration_seconds,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  NotificationConsumer
# ═══════════════════════════════════════════════════════════════════════════════

class NotificationConsumer(AsyncWebsocketConsumer):

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def connect(self):
        self.user = self.scope['user']
        if not self.user.is_authenticated:
            await self.close()
            return

        self.group_name = f'user_{self.user.id}'

        await self.channel_layer.group_add(self.group_name, self.channel_name)

        # Mark user online — 60 s heartbeat TTL
        await _cache_set(f'user_online_{self.user.id}', True, timeout=60)
        # Store username → id mapping (useful for call routing)
        await _cache_set(f'uid_{self.user.username}', self.user.id, timeout=86400)

        await self.accept()

    async def disconnect(self, close_code):
        if not self.user.is_authenticated:
            return

        # Mark offline immediately
        await _cache_delete(f'user_online_{self.user.id}')

        # If they were in a call, end it unless they reconnect within the grace period
        active_call = await _cache_get(f'in_call_{self.user.id}')
        if active_call:
            asyncio.create_task(self._end_call_if_still_gone(active_call))

        # Update last_seen in DB
        @sync_to_async
        def _update_last_seen():
            from django.utils import timezone
            if hasattr(self.user, 'profile'):
                self.user.profile.last_seen = timezone.now()
                self.user.profile.save(update_fields=['last_seen'])

        await _update_last_seen()

        # Notify anyone who has a pending chat request FROM this user
        pending_key = f'req_sent_by_{self.user.id}'
        pending_recipients = await _cache_get_list(pending_key)
        for recipient_id in pending_recipients:
            await self.channel_layer.group_send(
                f'user_{recipient_id}',
                {
                    'type': 'user_went_offline',
                    'username': self.user.username,
                }
            )
        await _cache_delete(pending_key)

        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    # ── incoming messages ──────────────────────────────────────────────────

    async def receive(self, text_data):
        print("WEBSOCKET MESSAGE RECEIVED:", text_data)
        try:
            data = json.loads(text_data)
        except json.JSONDecodeError:
            return

        msg_type = data.get('type')

        # ── presence heartbeat ────────────────────────────────────────────
        if msg_type == 'ping':
            await _cache_set(f'user_online_{self.user.id}', True, timeout=60)

            @sync_to_async
            def _touch_last_seen():
                from django.utils import timezone
                if hasattr(self.user, 'profile'):
                    self.user.profile.last_seen = timezone.now()
                    self.user.profile.save(update_fields=['last_seen'])

            await _touch_last_seen()

        # ── chat request flow ─────────────────────────────────────────────

        elif msg_type == 'send_chat_request':
            to_user_id = data.get('to_user_id')
            if not to_user_id:
                return

            receiver_online = await _cache_get(f'user_online_{to_user_id}')
            if not receiver_online:
                await self.send(text_data=json.dumps({
                    'type': 'chat_request_failed',
                    'reason': 'User is offline',
                }))
                return

            session_id = str(uuid.uuid4())
            session_data = {
                'from_user_id': self.user.id,
                'from_username': self.user.username,
                'to_user_id': to_user_id,
                'status': 'pending',
            }
            await _cache_set(f'session:{session_id}', json.dumps(session_data), timeout=120)

            # Track pending requests sent by this user (for offline notification)
            await _cache_get_or_set_list(
                f'req_sent_by_{self.user.id}', to_user_id, timeout=120
            )

            await self.channel_layer.group_send(
                f'user_{to_user_id}',
                {
                    'type': 'chat_request_received',
                    'session_id': session_id,
                    'from_username': self.user.username,
                    'from_user_id': self.user.id,
                }
            )

        elif msg_type == 'accept_chat_request':
            session_id = data.get('session_id')
            raw = await _cache_get(f'session:{session_id}')
            if not raw:
                return
            session = json.loads(raw)
            # Only the person the request was sent to can accept it
            if str(session.get('to_user_id')) != str(self.user.id):
                return
            session['status'] = 'active'
            await _cache_set(f'session:{session_id}', json.dumps(session), timeout=3600)

            for uid in [session['from_user_id'], self.user.id]:
                await self.channel_layer.group_send(
                    f'user_{uid}',
                    {
                        'type': 'session_started',
                        'session_id': session_id,
                        'peer_username': (
                            session['from_username']
                            if uid != session['from_user_id']
                            else self.user.username
                        ),
                    }
                )

        elif msg_type == 'decline_chat_request':
            session_id = data.get('session_id')
            raw = await _cache_get(f'session:{session_id}')
            if not raw:
                return
            session = json.loads(raw)
            if str(session.get('to_user_id')) != str(self.user.id):
                return
            await _cache_delete(f'session:{session_id}')

            await self.channel_layer.group_send(
                f'user_{session["from_user_id"]}',
                {
                    'type': 'chat_request_declined',
                    'session_id': session_id,
                    'declined_by': self.user.username,
                }
            )

        # ── session messaging (ephemeral, stored in Redis) ─────────────────

        elif msg_type == 'session_message':
            session_id = data.get('session_id')
            text = data.get('text', '').strip()
            if not session_id or not text:
                return

            raw = await _cache_get(f'session:{session_id}:meta')
            if not raw:
                # Support old format just in case
                raw = await _cache_get(f'session:{session_id}')
                if not raw:
                    return
            session = json.loads(raw)

            # SECURITY: only the two participants of this session may post into it.
            if str(self.user.id) not in (str(session.get('from_user_id')), str(session.get('to_user_id'))):
                return

            msg_obj = {
                'id': str(uuid.uuid4()),
                'client_id': data.get('client_id'),
                'sender_id': self.user.username,
                'sender': self.user.username,
                'text': text,
                'timestamp': datetime.now(dt_tz.utc).isoformat(),
                'status': 'sent'
            }
            await _cache_get_or_set_list(
                f'session:{session_id}:messages', msg_obj, timeout=86400
            )

            # Forward to the other participant
            peer_id = (
                session['to_user_id']
                if str(self.user.id) == str(session['from_user_id'])
                else session['from_user_id']
            )
            await self.channel_layer.group_send(
                f'user_{peer_id}',
                {
                    'type': 'session_message_recv',
                    'session_id': session_id,
                    'sender': self.user.username,
                    'text': text,
                    'client_id': data.get('client_id'),
                }
            )

        # ── friend messaging (permanent, stored in PostgreSQL) ─────────────

        elif msg_type == 'friend_message':
            conversation_id = data.get('conversation_id')  # e.g. "direct_alice_bob"
            text = (data.get('text') or '').strip()
            if not isinstance(conversation_id, str) or not conversation_id or not text:
                return

            msg_data = await _save_friend_message(self.user, conversation_id, text)
            if msg_data is None:
                # SECURITY: not a member of this conversation — drop silently.
                return

            # Determine partner (robust to usernames containing "_")
            from room.access import direct_room_peer
            peer = await sync_to_async(direct_room_peer)(conversation_id, self.user)
            peer_id = peer.id if peer else None

            if peer_id:
                peer_online = await _cache_get(f'user_online_{peer_id}')
                if not peer_online:
                    # Queue notification for offline user
                    await _cache_get_or_set_list(
                        f'offline_msgs:{peer_id}',
                        {'from': self.user.username, 'preview': text[:80]},
                        timeout=86400
                    )
                    
                    # FCM Push for Offline Friend
                    
                    from utils.fcm import send_fcm_push
                    
                    body_preview = text[:100] + ('...' if len(text) > 100 else '')
                    await sync_to_async(send_fcm_push)(
                        user_id=peer_id,
                        title=self.user.username,
                        body=body_preview,
                        data_payload={
                            'type': 'friend_message',
                            'conversation_id': conversation_id
                        }
                    )

        # ── typing indicators (forward only, never stored) ─────────────────

        elif msg_type == 'read_receipt':
            context_id = data.get('session_id') or data.get('conversation_id')
            if not isinstance(context_id, str) or not context_id:
                return

            if context_id.startswith('direct_'):
                # Only a member of the conversation can mark it read (works for usernames with "_")
                from room.access import direct_room_peer
                peer = await sync_to_async(direct_room_peer)(context_id, self.user)
                if peer is None:
                    return
                has_changes = await _set_messages_read(context_id, self.user)
                peer_id = peer.id
                if has_changes:
                    await self.channel_layer.group_send(
                        f'user_{peer_id}',
                        {
                            'type': 'read_receipt_recv',
                            'context_id': context_id,
                            'read_by': self.user.username,
                        }
                    )
            else:
                # For sessions, just forward it (ephemeral). Accept 'session_<id>' or bare '<id>'.
                sid = context_id[len('session_'):] if context_id.startswith('session_') else context_id
                raw = await _cache_get(f'session:{sid}:meta') or await _cache_get(f'session:{sid}')
                if raw:
                    session = json.loads(raw)
                    ids = (str(session.get('from_user_id')), str(session.get('to_user_id')))
                    if str(self.user.id) not in ids:
                        return
                    peer_id = (
                        session['to_user_id']
                        if str(self.user.id) == str(session['from_user_id'])
                        else session['from_user_id']
                    )
                    await self.channel_layer.group_send(
                        f'user_{peer_id}',
                        {
                            'type': 'read_receipt_recv',
                            'context_id': context_id,
                            'read_by': self.user.username,
                        }
                    )

        # ── call signalling ────────────────────────────────────────────────

        elif msg_type == 'call_offer':
            print(f"--- WEBRTC [call_offer] RECEIVED from {self.user.username}: {data}")
            to_user_id = data.get('to_user_id')
            to_username = data.get('to_username')
            call_id = data.get('call_id', str(uuid.uuid4()))
            sdp = data.get('sdp')
            mode = data.get('mode', 'video')
            context_id = data.get('context_id')

            if to_username and not to_user_id:
                to_user_id = await _cache_get(f'uid_{to_username}')

            if not to_user_id:
                print(f"--- WEBRTC [call_offer] FAILED: Could not resolve to_user_id for username {to_username}")
                return
            try:
                to_user_id = int(to_user_id)
            except (TypeError, ValueError):
                return
            if to_user_id == self.user.id:
                return
            callee = await _get_user_by_id(to_user_id)
            if callee is None:
                return
            # Call ids must be UUIDs (CallLog.call_id is a UUIDField), and a client may
            # never reuse/hijack an existing call id.
            try:
                call_id = str(uuid.UUID(str(call_id)))
            except (ValueError, TypeError):
                call_id = str(uuid.uuid4())
            if await _cache_get(f'call:{call_id}'):
                call_id = str(uuid.uuid4())

            receiver_online = await _cache_get(f'user_online_{to_user_id}')
            if not receiver_online:
                print(f"--- WEBRTC [call_offer] FAILED: User {to_user_id} is offline")
                await self.send(text_data=json.dumps({
                    'type': 'call_failed',
                    'reason': 'User is offline',
                    'call_id': call_id,
                }))
                return

            in_call = await _cache_get(f'in_call_{to_user_id}')
            if in_call:
                print(f"--- WEBRTC [call_offer] FAILED: User {to_user_id} is busy")
                await self.send(text_data=json.dumps({
                    'type': 'call_failed',
                    'reason': 'User is busy',
                    'call_id': call_id,
                }))
                return

            # Store call metadata
            call_meta = {
                'call_id': call_id,
                'caller_id': self.user.id,
                'caller_username': self.user.username,
                'callee_id': to_user_id,
                'callee_username': callee.username,
                'mode': mode,
                'offered_at': datetime.now(dt_tz.utc).isoformat(),
                'connected_at': None,   # set when the callee answers
                'context_id': context_id,
            }
            # Generous TTL: the 30s no-answer rule is enforced by _call_timeout, not by expiry
            await _cache_set(f'call:{call_id}', json.dumps(call_meta), timeout=120)
            
            # Fetch caller profile data
            @sync_to_async
            def get_profile_data(user):
                try:
                    p = user.profile
                    return {
                        'image_url': p.image.url if p.image else '',
                        'interests': p.interests
                    }
                except:
                    return {'image_url': '', 'interests': []}
            
            profile_data = await get_profile_data(self.user)

            print(f"--- WEBRTC [call_offer] Routing incoming_call to user_{to_user_id}")
            await self.channel_layer.group_send(
                f'user_{to_user_id}',
                {
                    'type': 'incoming_call',
                    'call_id': call_id,
                    'caller_username': self.user.username,
                    'caller_id': self.user.id,
                    'sdp': sdp,
                    'mode': mode,
                    'context_id': context_id,
                    'caller_image_url': profile_data.get('image_url', ''),
                    'caller_interests': profile_data.get('interests', []),
                }
            )

            # 30 second auto-cancel if not answered
            asyncio.create_task(self._call_timeout(call_id, to_user_id, 30))

        elif msg_type == 'call_answer':
            print(f"--- WEBRTC [call_answer] RECEIVED from {self.user.username}: {data}")
            call_id = data.get('call_id')
            sdp_answer = data.get('sdp_answer')

            raw = await _cache_get(f'call:{call_id}')
            if not raw:
                print(f"--- WEBRTC [call_answer] ERROR: Call {call_id} not found in Redis!")
                return
            call_meta = json.loads(raw)
            # Only the person being called can answer, and only once
            if call_meta.get('callee_id') != self.user.id or call_meta.get('connected_at'):
                return
            call_meta['connected_at'] = datetime.now(dt_tz.utc).isoformat()
            await _cache_set(f'call:{call_id}', json.dumps(call_meta), timeout=7200)

            # Mark both as in_call
            await _cache_set(f'in_call_{call_meta["caller_id"]}', call_id, timeout=7200)
            await _cache_set(f'in_call_{call_meta["callee_id"]}', call_id, timeout=7200)

            print(f"--- WEBRTC [call_answer] Routing call_answered back to caller_{call_meta['caller_id']}")
            await self.channel_layer.group_send(
                f'user_{call_meta["caller_id"]}',
                {
                    'type': 'call_answered',
                    'call_id': call_id,
                    'sdp_answer': sdp_answer,
                    'connected_at': call_meta['connected_at']
                }
            )

        elif msg_type == 'call_decline':
            call_id = data.get('call_id')
            raw = await _cache_get(f'call:{call_id}')
            if not raw:
                return
            call_meta = json.loads(raw)
            if self.user.id not in (call_meta.get('caller_id'), call_meta.get('callee_id')):
                return
            if call_meta.get('connected_at'):
                # Already answered: a "decline" now means hang up.
                await self._finish_call(call_id, call_meta, reason='hangup')
                return
            await self._finish_call(call_id, call_meta, reason='declined')

        elif msg_type == 'ice_candidate':
            call_id = data.get('call_id')
            candidate = data.get('candidate')

            raw = await _cache_get(f'call:{call_id}')
            if not raw:
                # Call might have expired or not exist yet if ICE generated too fast
                # We can cache it temporarily or just drop it. For now drop or log.
                print(f"--- WEBRTC [ice_candidate] ERROR: Call {call_id} not found locally for routing!")
                return
            
            call_meta = json.loads(raw)
            if self.user.id not in (call_meta.get('caller_id'), call_meta.get('callee_id')):
                return

            # Identify who gets the ICE candidate
            target_peer_id = (
                call_meta['callee_id'] 
                if self.user.id == call_meta['caller_id'] 
                else call_meta['caller_id']
            )

            print(f"--- WEBRTC [ice_candidate] Routing from {self.user.username} to peer_{target_peer_id}")
            await self.channel_layer.group_send(
                f'user_{target_peer_id}',
                {
                    'type': 'ice_candidate_forward',
                    'call_id': call_id,
                    'candidate': candidate,
                }
            )

        elif msg_type == 'call_end':
            call_id = data.get('call_id')
            raw = await _cache_get(f'call:{call_id}')
            if not raw:
                return
            call_meta = json.loads(raw)
            if self.user.id not in (call_meta.get('caller_id'), call_meta.get('callee_id')):
                return
            # Hanging up before the other side answered = cancelled; afterwards = ended.
            await self._finish_call(
                call_id, call_meta,
                reason='hangup' if call_meta.get('connected_at') else 'cancelled',
            )

        # ── legacy call signals (kept for backward compatibility) ──────────

        elif msg_type == 'initiate_call':
            target_user_id = data.get('target_user_id')
            room_id = data.get('room_id')
            mode = data.get('mode', 'video')
            await self.channel_layer.group_send(
                f'user_{target_user_id}',
                {
                    'type': 'incoming_call',
                    'caller_username': self.user.username,
                    'caller_id': self.user.id,
                    'room_id': room_id,
                    'mode': mode,
                }
            )

        elif msg_type == 'accept_call':
            caller_id = data.get('caller_id')
            room_id = data.get('room_id')
            if caller_id:
                await self.channel_layer.group_send(
                    f'user_{caller_id}',
                    {'type': 'call_accepted', 'room_id': room_id}
                )

        elif msg_type == 'decline_call':
            caller_id = data.get('caller_id')
            if caller_id:
                await self.channel_layer.group_send(
                    f'user_{caller_id}',
                    {'type': 'call_declined_signal'}
                )

    # ── call timeout task ──────────────────────────────────────────────────

    async def _call_timeout(self, call_id, callee_id, timeout_secs):
        await asyncio.sleep(timeout_secs)
        raw = await _cache_get(f'call:{call_id}')
        if not raw:
            return  # Already ended / declined

        call_meta = json.loads(raw)
        if call_meta.get('connected_at'):
            # Call was answered — leave it alone.
            # (Bug fixed: this used to check 'started_at', which answer never set,
            #  so every answered call was killed after 30 seconds.)
            return

        await self._finish_call(call_id, call_meta, reason='no_answer')

    async def _finish_call(self, call_id, call_meta, reason):
        """
        End a call exactly once: clear state, write one CallLog, notify BOTH sides.
        reason: 'hangup' (after answer) | 'cancelled' (caller gave up) |
                'declined' | 'no_answer' | 'disconnected'
        """
        # Claim the call atomically — only the handler that deletes it logs it
        if not await _cache_delete(f'call:{call_id}'):
            return
        for uid in (call_meta.get('caller_id'), call_meta.get('callee_id')):
            if uid is None:
                continue
            current = await _cache_get(f'in_call_{uid}')
            if current in (None, call_id):
                await _cache_delete(f'in_call_{uid}')

        now = datetime.now(dt_tz.utc)
        connected_at = call_meta.get('connected_at')
        if connected_at:
            started_at = datetime.fromisoformat(connected_at)
            duration_seconds = max(0, int((now - started_at).total_seconds()))
            status = 'ended'
        else:
            started_at = None
            duration_seconds = 0
            status = {'declined': 'declined', 'no_answer': 'no_answer'}.get(reason, 'cancelled')

        try:
            await _save_call_log(
                call_id=call_id,
                caller_username=call_meta.get('caller_username', ''),
                receiver_username=call_meta.get('callee_username', ''),
                mode=call_meta.get('mode', 'video'),
                status=status,
                started_at=started_at,
                ended_at=now,
                duration_seconds=duration_seconds,
                context_id=call_meta.get('context_id'),
            )
        except Exception as e:
            print(f"--- WEBRTC [finish_call] could not save call log: {e}")

        for uid in (call_meta.get('caller_id'), call_meta.get('callee_id')):
            if uid is None:
                continue
            await self.channel_layer.group_send(
                f'user_{uid}',
                {
                    'type': 'call_declined_signal' if status == 'declined' else 'call_ended_signal',
                    'call_id': call_id,
                    'reason': reason,
                    'duration_seconds': duration_seconds,
                }
            )

    async def _end_call_if_still_gone(self, call_id, grace_secs=15):
        """If a participant's socket drops mid-call and they don't come back
        within the grace period, end the call (otherwise both users stayed
        'busy' for 2 hours)."""
        await asyncio.sleep(grace_secs)
        if await _cache_get(f'user_online_{self.user.id}'):
            return  # reconnected
        raw = await _cache_get(f'call:{call_id}')
        if raw:
            await self._finish_call(call_id, json.loads(raw), reason='disconnected')
        else:
            await _cache_delete(f'in_call_{self.user.id}')

    # ── group-send handler methods ─────────────────────────────────────────

    async def notification_message(self, event):
        await self.send(text_data=json.dumps({
            'type': 'notification',
            'message': event['message'],
        }))

    async def chat_request_received(self, event):
        await self.send(text_data=json.dumps({
            'type': 'chat_request_received',
            'session_id': event['session_id'],
            'from_username': event['from_username'],
            'from_user_id': event['from_user_id'],
        }))

    async def chat_request_declined(self, event):
        await self.send(text_data=json.dumps({
            'type': 'chat_request_declined',
            'session_id': event['session_id'],
            'declined_by': event['declined_by'],
        }))

    async def session_started(self, event):
        await self.send(text_data=json.dumps({
            'type': 'session_started',
            'session_id': event['session_id'],
            'peer_username': event['peer_username'],
        }))

    async def session_message_recv(self, event):
        await self.send(text_data=json.dumps({
            'type': 'session_message',
            'session_id': event['session_id'],
            'sender': event['sender'],
            'text': event['text'],
            'client_id': event.get('client_id'),
        }))

    async def friend_message_recv(self, event):
        print(f"DEBUG: [NotificationConsumer] Signal received for user {self.user.username} (ID: {self.user.id})")
        await self.send(text_data=json.dumps({
            'type': 'friend_message',
            'conversation_id': event['conversation_id'],
            'id': event.get('id'),
            'client_id': event.get('client_id'),
            'sender': event['sender'],
            'text': event['text'],
            'timestamp': event.get('timestamp'),
            'notif_key': event.get('notif_key'),
        }))

    async def typing_indicator(self, event):
        await self.send(text_data=json.dumps({
            'type': event['event'],
            'username': event['username'],
            'context_id': event.get('context_id'),
        }))

    async def incoming_call(self, event):
        await self.send(text_data=json.dumps({
            'type': 'incoming_call',
            'call_id': event.get('call_id'),
            'caller_username': event['caller_username'],
            'caller_id': event.get('caller_id'),
            'room_id': event.get('room_id'),
            'sdp': event.get('sdp'),
            'mode': event.get('mode', 'video'),
            'context_id': event.get('context_id'),
            'caller_image_url': event.get('caller_image_url', ''),
            'caller_interests': event.get('caller_interests', []),
        }))

    async def call_answered(self, event):
        await self.send(text_data=json.dumps({
            'type': 'call_answered',
            'call_id': event['call_id'],
            'sdp_answer': event.get('sdp_answer'),
        }))

    async def call_accepted(self, event):
        await self.send(text_data=json.dumps({
            'type': 'call_accepted',
            'room_id': event['room_id'],
        }))

    async def call_declined_signal(self, event):
        await self.send(text_data=json.dumps({
            'type': 'call_declined',
            'call_id': event.get('call_id'),
        }))

    async def ice_candidate_forward(self, event):
        await self.send(text_data=json.dumps({
            'type': 'ice_candidate',
            'call_id': event['call_id'],
            'candidate': event['candidate'],
        }))

    async def call_ended_signal(self, event):
        await self.send(text_data=json.dumps({
            'type': 'call_ended',
            'call_id': event.get('call_id'),
            'duration_seconds': event.get('duration_seconds', 0),
            'reason': event.get('reason'),
        }))

    async def read_receipt_recv(self, event):
        await self.send(text_data=json.dumps({
            'type': 'read_receipt',
            'context_id': event['context_id'],
            'read_by': event['read_by'],
        }))


# ═══════════════════════════════════════════════════════════════════════════════
#  ChatConsumer  (room-scoped)
# ═══════════════════════════════════════════════════════════════════════════════

class ChatConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_history = []
        self.user = None

    async def connect(self):
        self.room_group_name = None
        self.user = self.scope.get('user')
        self.room_name = self.scope['url_route']['kwargs']['room_name']

        # SECURITY: no anonymous sockets, and only members of the room may join its group.
        # Previously anyone could connect to ws/chat/<any room>/ and read every live
        # message and WebRTC signal, or inject messages as anyone.
        if not self.user or not self.user.is_authenticated:
            await self.close(code=4401)
            return
        from channels.db import database_sync_to_async
        from room.access import get_accessible_room

        allowed, _ = await database_sync_to_async(get_accessible_room)(
            self.user, self.room_name, True
        )
        if not allowed:
            await self.close(code=4403)
            return

        import re
        safe_room_name = re.sub(r'[^a-zA-Z0-9_\-]', '', self.room_name)
        self.room_group_name = f'chat_{safe_room_name}'

        await self.channel_layer.group_add(self.room_group_name, self.channel_name)

        # TTL Logic for Session Reset
        if self.room_name.startswith('session_') and self.user and self.user.is_authenticated:
            session_id = self.room_name.replace('session_', '')
            users = await _cache_get_list(f'session:{session_id}:active')
            if self.user.id not in users:
                users.append(self.user.id)
                await _cache_set(f'session:{session_id}:active', json.dumps(users), timeout=86400)
            
            # Reset TTLs to 24h because someone joined
            meta = await _cache_get(f'session:{session_id}:meta')
            msgs = await _cache_get(f'session:{session_id}:messages')
            calls = await _cache_get(f'session:{session_id}:calls')
            if meta: await _cache_set(f'session:{session_id}:meta', meta, timeout=86400)
            if msgs: await _cache_set(f'session:{session_id}:messages', msgs, timeout=86400)
            if calls: await _cache_set(f'session:{session_id}:calls', calls, timeout=86400)

        # Connection accounting to handle refreshes gracefully
        if self.user and self.user.is_authenticated:
            conn_key = f"conn_count:{self.room_name}:{self.user.id}"
            count = await _update_conn_count(conn_key, 1)
            if count == 1:
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        'type': 'user_joined',
                        'sender': self.channel_name,
                        'username': self.user.username,
                    }
                )
        else:
            # Anonymous or guest
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'user_joined',
                    'sender': self.channel_name,
                    'username': 'Anonymous',
                }
            )
        await self.accept()

    async def disconnect(self, close_code):
        if not self.room_group_name:
            return  # connection was rejected in connect()
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

        if self.user and self.user.is_authenticated:
            conn_key = f"conn_count:{self.room_name}:{self.user.id}"
            count = await _update_conn_count(conn_key, -1)
            if count == 0:
                # Start grace period background task
                asyncio.create_task(self._delayed_disconnect(conn_key))

    async def _delayed_disconnect(self, conn_key):
        """Wait for grace period, then notify if user hasn't reconnected."""
        await asyncio.sleep(5)
        count = await _cache_get(conn_key) or 0
        if count > 0:
            return # User reconnected during grace period!

        # Truly gone -> Notify group
        await self.channel_layer.group_send(
            self.room_group_name,
            {'type': 'user_left', 'sender': 'system'}
        )

        # Cleanup session cache if it's an ephemeral room
        if self.room_name.startswith('session_') and self.user and self.user.is_authenticated:
            session_id = self.room_name.replace('session_', '')
            users = await _cache_get_list(f'session:{session_id}:active')

            # --- TRUST SCORE: Session Quality Tracking ---
            try:
                from room.models import Room
                from django.utils import timezone
                room = await sync_to_async(Room.objects.get)(name=self.room_name)
                delta = (timezone.now() - (room.created_at or timezone.now())).total_seconds()
                
                msgs_raw = await _cache_get(f'session:{session_id}:messages')
                has_messages = False
                if msgs_raw:
                    try:
                        has_messages = len(json.loads(msgs_raw)) > 0
                    except: pass
                
                # Check if session lasted over 2 mins and had messages
                # Avoid double counting if we already tracked it for this user
                tracked_key = f'tracked_qual_session_{session_id}_{self.user.id}'
                already_tracked = await _cache_get(tracked_key)

                if delta > 120 and has_messages and not already_tracked:
                    await _cache_set(tracked_key, True, timeout=86400)
                    @sync_to_async
                    def _update_qual_session(u):
                        if hasattr(u, 'profile'):
                            u.profile.qualifying_sessions += 1
                            u.profile.save(update_fields=['qualifying_sessions'])
                            from users.trust import apply_trust_event
                            apply_trust_event(u.id, 'session_end_qualified', 0, "Completed a 2+ min session")
                    await _update_qual_session(self.user)
            except Exception as e:
                print(f"DEBUG: Session Qualify Tracking Error: {e}")
            # ---------------------------------------------

            if self.user.id in users:
                users.remove(self.user.id)
                await _cache_set(f'session:{session_id}:active', json.dumps(users), timeout=86400)
            
            if len(users) == 0:
                # Both users left -> delete session immediately
                await _cache_delete(f'session:{session_id}:meta')
                await _cache_delete(f'session:{session_id}:messages')
                await _cache_delete(f'session:{session_id}:calls')
                await _cache_delete(f'session:{session_id}:active')
            else:
                # One user offline -> 10m TTL
                meta = await _cache_get(f'session:{session_id}:meta')
                msgs = await _cache_get(f'session:{session_id}:messages')
                calls = await _cache_get(f'session:{session_id}:calls')
                if meta: await _cache_set(f'session:{session_id}:meta', meta, timeout=600)
                if msgs: await _cache_set(f'session:{session_id}:messages', msgs, timeout=600)
                if calls: await _cache_set(f'session:{session_id}:calls', calls, timeout=600)

    async def receive(self, text_data):
        if not self.room_group_name:
            return
        try:
            data = json.loads(text_data)
        except (TypeError, ValueError):
            return
        if not isinstance(data, dict):
            return
        message_type = data.get('type')
        # SECURITY: identity always comes from the authenticated socket, never from the client.
        my_username = self.user.username

        if message_type == 'chat_message':
            message = data.get('message')
            if not isinstance(message, str) or not message.strip():
                return
            username = my_username

            self.message_history.append(f'{username}: {message}')
            if len(self.message_history) > 10:
                self.message_history = self.message_history[-10:]

            # Real ISO-8601 UTC time; the browser shows it in the viewer's own timezone
            sent_at = datetime.now(dt_tz.utc).isoformat()
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'chat_message',
                    'message': message,
                    'username': username,
                    'client_id': data.get('client_id'),
                    'date_iso': sent_at,
                }
            )

            # Store in Redis if ephemeral session
            if self.room_name.startswith('session_'):
                timestamp = sent_at  # was "04:52 PM" text -> "Invalid Date" in the UI
                msg_data = {
                    "id": f"eph_{uuid.uuid4().hex[:8]}",
                    "sender": username,
                    "text": message,
                    "status": "read",
                    "isRead": True,
                    "deletedForEveryone": False,
                    "timestamp": timestamp,
                    "is_ephemeral": True,
                    "client_id": data.get('client_id')
                }
                # Since we are in AsyncWebsocketConsumer, we should use a sync_to_async or just call it if it's fast
                await sync_to_async(_cache_append_message)(self.room_name, msg_data)

            if len(self.message_history) % 5 == 0:
                if self.user and self.user.is_authenticated:
                    is_premium = await _is_user_premium(self.user.username)
                    if is_premium:
                        has_quota = await _check_and_deduct_ai_quota(self.user.username)
                        if has_quota:
                            asyncio.create_task(self.analyze_chat_history(username))

        elif message_type == 'manual_analyze_chat':
            if self.user and self.user.is_authenticated:
                has_quota = await _check_and_deduct_ai_quota(self.user.username)
                if has_quota:
                    # Analyze the most recent sender or the other person
                    recent_sender = "Peer"
                    if len(self.message_history) > 0:
                        recent_sender = self.message_history[-1].split(":")[0]
                    asyncio.create_task(self.analyze_chat_history(recent_sender))
                else:
                    await self.send(text_data=json.dumps({
                        'type': 'system_error',
                        'message': 'AI Analysis Quota Exceeded. Please upgrade your plan.'
                    }))

        elif message_type == 'video_signal':
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'video_signal',
                    'signal': data.get('signal'),
                    'sender': self.channel_name,
                    'username': my_username,
                    'mode': data.get('mode'),
                }
            )

        elif message_type == 'end_call':
            await self.channel_layer.group_send(
                self.room_group_name,
                {'type': 'end_call', 'sender': self.channel_name, 'username': my_username}
            )

        elif message_type == 'force_exit':
            await self.channel_layer.group_send(
                self.room_group_name,
                {'type': 'force_exit', 'sender': self.channel_name}
            )

        elif message_type in ('typing_start', 'typing_stop'):
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'typing_indicator',
                    'event': message_type,
                    'username': my_username,
                    'sender': self.channel_name,
                }
            )

        elif message_type == 'read_receipt':
            # 1. Update DB (if friend room)
            if self.room_name.startswith('direct_') and self.user.is_authenticated:
                await _set_messages_read(self.room_name, self.user)
            
            # 2. Broadcast to room group
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'message_read_receipt',
                    'read_by': self.user.username if self.user.is_authenticated else 'Guest',
                }
            )

    # ── group-send handlers ────────────────────────────────────────────────

    async def chat_message(self, event):
        await self.send(text_data=json.dumps({
            'type': 'chat_message',
            'message': event['message'],
            'username': event['username'],
            'client_id': event.get('client_id'),
            'id': event.get('id'),
            'date_iso': event.get('date_iso'),
            'timestamp': event.get('timestamp'),
            # call-log entries pushed into the room
            'is_call_log': event.get('is_call_log', False),
            'call_mode': event.get('call_mode'),
            'call_status': event.get('call_status'),
            'call_duration': event.get('call_duration'),
        }))

    async def video_signal(self, event):
        if self.channel_name != event['sender']:
            await self.send(text_data=json.dumps({
                'type': 'video_signal',
                'signal': event['signal'],
                'username': event.get('username'),
                'mode': event.get('mode'),
            }))

    async def end_call(self, event):
        if self.channel_name != event['sender']:
            await self.send(text_data=json.dumps({
                'type': 'end_call',
                'username': event.get('username'),
            }))

    async def force_exit(self, event):
        if self.channel_name != event.get('sender'):
            await self.send(text_data=json.dumps({'type': 'force_exit'}))

    async def user_left(self, event):
        await self.send(text_data=json.dumps({
            'type': 'user_left',
            'sender': event['sender'],
        }))

    async def user_joined(self, event):
        if self.channel_name != event['sender']:
            await self.send(text_data=json.dumps({
                'type': 'user_joined',
                'sender': event['sender'],
                'username': event.get('username', 'Unknown'),
            }))

    async def typing_indicator(self, event):
        if self.channel_name != event.get('sender'):
            await self.send(text_data=json.dumps({
                'type': event['event'],
                'username': event['username'],
            }))

    async def message_read(self, event):
        await self.send(text_data=json.dumps({
            'type': 'message_read',
            'id': event['id'],
        }))

    async def message_deleted(self, event):
        await self.send(text_data=json.dumps({
            'type': 'message_deleted',
            'id': event['id'],
        }))

    async def message_read_receipt(self, event):
        await self.send(text_data=json.dumps({
            'type': 'read_receipt',
            'read_by': event.get('read_by'),
        }))

    async def analysis_alert(self, event):
        await self.send(text_data=json.dumps({
            'type': 'analysis_alert',
            'username': event['username'],
            'reason': event['reason'],
            'image_url': event['image_url'],
        }))

    # ── AI fraud/intent monitoring ─────────────────────────────────────────

    async def analyze_chat_history(self, recent_sender_username):
        if not self.message_history:
            return

        chat_log = '\n'.join(self.message_history)

        @sync_to_async
        def call_groq():
            try:
                import json as _json
                from groq_client import groq_generate

                prompt = f"""
You are a ruthless AI moderator monitoring a live chat between two people.
Read this transcript of the last 10 messages:

{chat_log}

Analyze the intent of '{recent_sender_username}'. Are they being manipulative, overly creepy, asking for money, or acting like a fraudster/bot?

Respond ONLY in JSON:
{{
    "is_dangerous": false,
    "reason": "short explanation if dangerous, else null",
    "image_prompt": "If dangerous, a 10-word visual prompt describing them as a monster or fraud, else null"
}}
"""
                text = groq_generate(prompt)
                text = text.strip()
                if text.startswith('```json'):
                    text = text[7:-3]
                elif text.startswith('```'):
                    text = text[3:-3]
                return _json.loads(text)
            except Exception as e:
                return {'error': str(e)}

        analysis = await call_groq()

        if analysis and analysis.get('is_dangerous'):
            base_prompt = analysis.get('image_prompt', 'An abstract digital monster representing deceit')
            style = ', dark glitch art, red warning neon, ominous, 4k'
            import urllib.parse
            from django.utils import timezone
            encoded = urllib.parse.quote(base_prompt + style)
            img_url = f'https://pollinations.ai/p/{encoded}?width=400&height=400&nologo=true'

            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'analysis_alert',
                    'username': recent_sender_username,
                    'reason': analysis.get('reason'),
                    'image_url': img_url,
                }
            )