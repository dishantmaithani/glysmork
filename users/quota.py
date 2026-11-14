"""
Daily search quotas — the single source of truth for limits, resets and the numbers
the dashboard shows.

Before this module the limits were hardcoded separately in the matchmaking view and in
the frontend, the daily reset only happened when the user *searched* (so the dashboard
showed yesterday's usage as today's), and "Persona Match" searches were counted in a
counter the dashboard never displayed.
"""
from django.utils import timezone

# None = unlimited
LIMITS = {
    'free':    {'ai': 4,  'standard': 4,   'roulette': 20},
    'premium': {'ai': 40, 'standard': 100, 'roulette': None},
}

FIELDS = {
    'ai': 'daily_ai_llm_searches',          # Smart Search (uses your onboarding data + AI)
    'standard': 'daily_standard_searches',  # Persona / profile-based match
    'roulette': 'daily_roulette_searches',  # Random match
}


def today():
    return timezone.localtime(timezone.now()).date()


def limits_for(profile):
    return LIMITS['premium' if profile.is_premium else 'free']


def reset_if_new_day(profile, save=True):
    """Zero the daily counters if they were last reset on an earlier day. Returns True if reset."""
    t = today()
    last = profile.last_quota_reset_date
    # The model default is timezone.now (a datetime) — on a freshly created profile the
    # in-memory value is a datetime, not a date. Normalise before comparing.
    if hasattr(last, 'date') and callable(getattr(last, 'date')):
        last = timezone.localtime(last).date() if timezone.is_aware(last) else last.date()
    if last and last >= t:
        return False
    for f in FIELDS.values():
        setattr(profile, f, 0)
    profile.last_quota_reset_date = t
    if save:
        profile.save(update_fields=list(FIELDS.values()) + ['last_quota_reset_date'])
    return True


def has_remaining(profile, kind):
    limit = limits_for(profile)[kind]
    return limit is None or getattr(profile, FIELDS[kind]) < limit


def consume(profile, kind):
    """Atomically add one use of `kind` if under the limit. Returns True if allowed."""
    from django.db.models import F
    from users.models import Profile

    reset_if_new_day(profile)
    field = FIELDS[kind]
    limit = limits_for(profile)[kind]
    qs = Profile.objects.filter(pk=profile.pk)
    if limit is not None:
        qs = qs.filter(**{f'{field}__lt': limit})
    updated = qs.update(**{field: F(field) + 1})
    # Always re-read the counters so callers (and quota.summary) see the real values,
    # even if this in-memory profile object was stale.
    profile.refresh_from_db(fields=list(FIELDS.values()) + ['last_quota_reset_date'])
    return bool(updated)


def summary(profile):
    """What the dashboard shows: used / limit / remaining for each kind, plus reset time."""
    reset_if_new_day(profile)
    lim = limits_for(profile)
    out = {'is_premium': profile.is_premium}
    for kind, field in FIELDS.items():
        used = getattr(profile, field) or 0
        limit = lim[kind]
        out[kind] = {
            'used': used,
            'limit': limit,  # null = unlimited
            'remaining': None if limit is None else max(0, limit - used),
        }
    # counters reset at midnight in the server's TIME_ZONE (UTC)
    nxt = timezone.localtime(timezone.now()).replace(hour=0, minute=0, second=0, microsecond=0) + timezone.timedelta(days=1)
    out['resets_at'] = nxt.isoformat()
    return out
