"""
Room access control — the single place that decides whether a user may read/write a room.

SECURITY: before this module existed, the chat REST API and websockets trusted a
`username` sent by the client and never checked room membership, so anyone could
read or post into any conversation.

Rules:
  * DB room the user is a member of            -> allowed
  * `session_<id>` ephemeral room              -> allowed if the user is one of the two
                                                   participants stored in the session cache
  * `direct_<a>_<b>` friend/direct room        -> allowed only if the caller is <a> or <b>
                                                   (name must be the sorted pair, as the
                                                   frontend builds it). Missing room/members
                                                   are created/repaired for the two users.
Everything else is denied.
"""
import json

from django.contrib.auth.models import User
from django.core.cache import cache

from room.models import Room


def direct_room_peer(room_name, user):
    """
    For a `direct_<a>_<b>` room name, return the *other* User if `user` is one of the
    two people in it, else None. Works for usernames that contain underscores,
    because we anchor on the caller's own username instead of splitting on "_".
    """
    if not room_name or not room_name.startswith('direct_'):
        return None
    me = user.username
    rest = room_name[len('direct_'):]
    candidates = []
    if rest.startswith(me + '_'):
        candidates.append(rest[len(me) + 1:])
    if rest.endswith('_' + me):
        candidates.append(rest[:-(len(me) + 1)])
    for peer_name in candidates:
        if not peer_name or peer_name == me:
            continue
        a, b = sorted([me, peer_name])
        if room_name != f'direct_{a}_{b}':
            continue
        peer = User.objects.filter(username=peer_name).first()
        if peer:
            return peer
    return None


def _session_participant_ids(session_id):
    for key in (f'session:{session_id}:meta', f'session:{session_id}'):
        raw = cache.get(key)
        if not raw:
            continue
        try:
            meta = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        except (TypeError, ValueError):
            continue
        ids = {str(meta.get('from_user_id')), str(meta.get('to_user_id'))}
        ids.discard('None')
        return ids
    return set()


def get_accessible_room(user, room_name, create_direct=False):
    """
    Returns (allowed: bool, room: Room | None).
    `room` may be None for an allowed ephemeral session room that has no DB row,
    or for a direct room that doesn't exist yet when create_direct=False.
    """
    if not user or not user.is_authenticated or not room_name:
        return False, None

    room = Room.objects.filter(name=room_name).order_by('id').first()

    if room is not None and room.users.filter(id=user.id).exists():
        return True, room

    if room_name.startswith('session_'):
        session_id = room_name[len('session_'):]
        if str(user.id) in _session_participant_ids(session_id):
            return True, room
        return False, None

    if room_name.startswith('direct_'):
        peer = direct_room_peer(room_name, user)
        if peer is None:
            return False, None
        if room is None and create_direct:
            room = Room.objects.create(name=room_name)
            try:
                from matchmaking.api.views import get_commonality_reason
                room.match_reason = get_commonality_reason(user.profile, peer.profile)
                room.save(update_fields=['match_reason'])
            except Exception:
                pass
        if room is not None:
            # Repair rooms created before this fix with no/partial members.
            room.users.add(user, peer)
        return True, room

    return False, None


def user_can_access_room(user, room_name):
    allowed, _ = get_accessible_room(user, room_name, create_direct=False)
    return allowed
