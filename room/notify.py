"""
Tell the other person in a direct conversation that a new message arrived.

Before this helper existed, messages sent from the DM page (REST + chat socket) were only
broadcast *inside* the chat room — so the recipient got nothing (no toast, no unread
badge) unless they already had that exact conversation open.
"""
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.utils import timezone


def notify_direct_message(sender, room, msg, client_id=None):
    """Create/refresh the unread ChatNotification and push a real-time signal to every
    other member of `room` (on their personal `user_<id>` notification socket)."""
    from matchmaking.models import ChatNotification
    from matchmaking.api.serializers import ChatNotificationSerializer

    layer = get_channel_layer()
    for partner in room.users.exclude(id=sender.id):
        # One unread notification per (sender, receiver) — refresh it instead of stacking
        notif, created = ChatNotification.objects.get_or_create(
            sender=sender, receiver=partner, is_read=False,
            defaults={'room_name': room.name, 'message': 'sent you a message!'},
        )
        if not created:
            notif.created_at = timezone.now()
            notif.room_name = room.name
            notif.message = 'sent you a message!'
            notif.save(update_fields=['created_at', 'room_name', 'message'])

        if layer is None:
            continue
        # Same key the frontend's notification poll uses, so the toast isn't shown twice
        created_at = ChatNotificationSerializer(notif).data.get('created_at')
        try:
            async_to_sync(layer.group_send)(
                f'user_{partner.id}',
                {
                    'type': 'friend_message_recv',
                    'conversation_id': room.name,
                    'id': msg.id,
                    'client_id': client_id,
                    'sender': sender.username,
                    'text': msg.value,
                    'timestamp': msg.date.strftime('%I:%M %p'),
                    'notif_key': f'{notif.id}_{created_at}',
                },
            )
        except Exception as e:  # real-time push is best effort; the message is saved
            print(f"notify_direct_message: push failed: {e}")
