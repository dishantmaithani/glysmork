"""
Process saved offline searches ("Search Offline").

Run it from cron, e.g. every 30 minutes:
    */30 * * * * cd /home/ubuntu/glysmork && /home/ubuntu/venv/bin/python manage.py process_offline_matches

What changed (it was never scheduled, so 0 offline matches were ever made, and it also):
  * now only considers people who opted in with "Available for Offline Search", are public
    and not banned (before: anyone, including banned users);
  * uses the same search engine as live search (calibrated scores, NOT/hate, whole words)
    instead of one LLM call per random candidate (slow, costly, and the model was retired);
  * Persona-Match offline searches are matched on profile similarity;
  * creates a normal direct conversation WITH both people as members (before: an
    "offline_*" room with no members that nobody could open) and notifies both, live
    if they're online.
"""
from django.core.management.base import BaseCommand
from django.db.models import Q
from django.utils import timezone

from matchmaking.models import OfflineSearch, MatchHistory, Friendship
from users.models import Profile

MAX_MATCHES_PER_SEARCH = 4
CANDIDATE_POOL = 40


class Command(BaseCommand):
    help = 'Processes saved offline matchmaking requests'

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true',
                            help='Show what would match without creating rooms or notifications')

    def handle(self, *args, **options):
        from matchmaking.engine import run_hybrid_discovery, rank_by_profile_similarity, has_profile_data
        dry = options.get('dry_run')
        self.stdout.write(self.style.SUCCESS(f'Offline matchmaking started{" (dry run)" if dry else ""}'))

        # 1. Expire searches not refreshed in 24 hours
        stale = timezone.now() - timezone.timedelta(hours=24)
        expired = OfflineSearch.objects.filter(is_active=True, daily_refresh_timestamp__lt=stale)
        if not dry:
            n = expired.update(is_active=False)
            if n:
                self.stdout.write(f'Expired {n} searches older than 24h')

        # (also excludes expired ones explicitly, so --dry-run shows exactly what a real run would)
        searches = OfflineSearch.objects.filter(is_active=True, matches_found__lt=MAX_MATCHES_PER_SEARCH,
                                                daily_refresh_timestamp__gte=stale) \
                                        .select_related('user', 'user__profile')
        total_new = 0
        for search in searches:
            user = search.user
            me = getattr(user, 'profile', None)
            if me is None or me.is_banned:
                continue

            # Never re-match people already matched, friends, or yourself
            excluded = {user.id}
            for u1, u2 in MatchHistory.objects.filter(Q(user1=user) | Q(user2=user)).values_list('user1_id', 'user2_id'):
                excluded.update((u1, u2))
            for a, b in Friendship.objects.filter(Q(from_user=user) | Q(to_user=user)).values_list('from_user_id', 'to_user_id'):
                excluded.update((a, b))

            qs = (Profile.objects.select_related('user')
                  .filter(is_profile_public=True, is_banned=False, available_for_offline_search=True)
                  .exclude(user_id__in=excluded))
            if search.gender_filter in ('M', 'F', 'O'):
                qs = qs.filter(gender=search.gender_filter)
            if search.country_filter:
                qs = qs.filter(country__in=search.country_filter)
            if search.language_filter:
                # in Python: SQLite (production) can't do JSON `contains` lookups
                from matchmaking.engine import speaks_any
                candidates = [p for p in qs.order_by('-last_seen')[:500]
                              if speaks_any(p, search.language_filter)][:CANDIDATE_POOL]
            else:
                candidates = list(qs.order_by('-last_seen')[:CANDIDATE_POOL])
            if not candidates:
                continue

            try:
                if (search.intent or '').strip().lower() == 'persona match':
                    if not has_profile_data(me):
                        continue
                    matches = rank_by_profile_similarity(me, candidates)
                else:
                    matches, _ = run_hybrid_discovery(search.intent, candidates, searcher_profile=me)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'  {user.username}: engine error: {e}'))
                continue

            room_left = MAX_MATCHES_PER_SEARCH - search.matches_found
            for m in matches[:room_left]:
                cand = m['profile'].user
                pct = round(m['vector_score'] * 100)
                terms = ', '.join((m.get('matched_terms') or [])[:3])
                self.stdout.write(f'  {user.username} -> {cand.username} ({pct}%{": " + terms if terms else ""})')
                if dry:
                    continue
                self._connect(user, cand, terms)
                search.matches_found += 1
                total_new += 1

            if not dry:
                if search.matches_found >= MAX_MATCHES_PER_SEARCH:
                    search.is_active = False
                search.save(update_fields=['matches_found', 'is_active'])

        self.stdout.write(self.style.SUCCESS(f'Offline matchmaking done: {total_new} new matches'))

    def _connect(self, user, cand, terms):
        """Create the conversation for both people, log history, notify both."""
        from room.models import Room
        from matchmaking.models import ChatNotification

        a, b = sorted([user.username, cand.username])
        room, _ = Room.objects.get_or_create(name=f'direct_{a}_{b}')
        room.users.add(user, cand)
        room.is_active = True
        room.match_reason = f'Offline match{": " + terms if terms else ""}'
        room.save(update_fields=['is_active', 'match_reason'])

        MatchHistory.objects.get_or_create(
            user1=user if user.id < cand.id else cand,
            user2=cand if user.id < cand.id else user,
        )
        for sender, receiver, text in (
            (cand, user, f'Offline match found: {cand.username} matches your search!'),
            (user, cand, f'{user.username} was matched with you by offline search.'),
        ):
            ChatNotification.objects.create(sender=sender, receiver=receiver, room_name=room.name, message=text[:500])
            self._push(receiver, sender, room.name, text)

    def _push(self, receiver, sender, room_name, text):
        try:
            from asgiref.sync import async_to_sync
            from channels.layers import get_channel_layer
            layer = get_channel_layer()
            if layer:
                async_to_sync(layer.group_send)(f'user_{receiver.id}', {
                    'type': 'friend_message_recv', 'conversation_id': room_name,
                    'sender': sender.username, 'text': text,
                })
        except Exception:
            pass  # the notification is stored; the poll will show it
