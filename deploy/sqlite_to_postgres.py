"""One-off: copy every row of the old SQLite database into a migrated Postgres database.

Usage (on the server, site stopped, Postgres schema created with `manage.py migrate`):

    DATABASE_URL=postgres://... python deploy/sqlite_to_postgres.py /path/to/db.sqlite3

* Goes through the Django models (so booleans, datetimes and JSON are converted
  properly), keeps every primary key, and bypasses signals (bulk_create) so no
  duplicate profiles get created. auto_now/auto_now_add are switched off while
  copying so original timestamps are kept.
* Empties the target tables first (including the content types / permissions that
  `migrate` created, so their ids match the copied rows), runs in ONE transaction
  (Postgres FK checks are deferred to commit), then resets the id sequences.
* Afterwards compares every row of every table between the two databases and
  rolls everything back if anything differs.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chat.settings')

if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
    raise SystemExit('usage: sqlite_to_postgres.py /path/to/db.sqlite3')
SRC = os.path.abspath(sys.argv[1])

from django.conf import settings  # noqa: E402

settings.DATABASES['old'] = {'ENGINE': 'django.db.backends.sqlite3', 'NAME': SRC}

import django  # noqa: E402

django.setup()

from django.apps import apps  # noqa: E402
from django.core.management.color import no_style  # noqa: E402
from django.db import connections, models, transaction  # noqa: E402

src, dst = connections['old'], connections['default']
if dst.vendor != 'postgresql':
    raise SystemExit(f'target database is {dst.vendor}, expected postgresql (set DATABASE_URL)')

# Both databases must be at exactly the same migrations, or columns would differ.
q = 'SELECT app, name FROM django_migrations'
with src.cursor() as c:
    m_src = set(c.execute(q).fetchall())
with dst.cursor() as c:
    c.execute(q)
    m_dst = set(c.fetchall())
if m_src != m_dst:
    raise SystemExit(f'migration mismatch: only in sqlite {sorted(m_src - m_dst)}, '
                     f'only in postgres {sorted(m_dst - m_src)}')

MODELS = [m for m in apps.get_models(include_auto_created=True)
          if m._meta.managed and not m._meta.proxy]
src_tables = set(src.introspection.table_names())
model_tables = {m._meta.db_table for m in MODELS}
extra = sorted(src_tables - model_tables - {'django_migrations', 'sqlite_sequence'})
if extra:
    raise SystemExit(f'sqlite has tables with no model (would be lost): {extra}')

# keep original timestamps
for m in MODELS:
    for f in m._meta.concrete_fields:
        if isinstance(f, (models.DateField, models.TimeField)):
            f.auto_now = f.auto_now_add = False


def rows(model, alias):
    """All rows as comparable JSON, ordered by pk."""
    fields = [f.attname for f in model._meta.concrete_fields]
    out = []
    for r in model._base_manager.using(alias).order_by('pk').values_list(*fields):
        out.append(json.dumps(r, default=str, sort_keys=True))
    return out


class Mismatch(Exception):
    pass


try:
    with transaction.atomic(using='default'):
        with dst.cursor() as c:
            c.execute('TRUNCATE ' + ', '.join(dst.ops.quote_name(t) for t in sorted(model_tables))
                      + ' RESTART IDENTITY CASCADE')
        for m in MODELS:
            t = m._meta.db_table
            if t not in src_tables:
                print(f'  {t:<40} not in sqlite (new table, left empty)')
                continue
            objs = list(m._base_manager.using('old').order_by('pk'))
            for i in range(0, len(objs), 500):
                m._base_manager.using('default').bulk_create(objs[i:i + 500])
            print(f'  {t:<40} {len(objs):>6} rows')
        with dst.cursor() as c:
            for sql in dst.ops.sequence_reset_sql(no_style(), MODELS):
                c.execute(sql)
        bad = []
        for m in MODELS:
            if m._meta.db_table in src_tables and rows(m, 'old') != rows(m, 'default'):
                bad.append(m._meta.db_table)
        if bad:
            raise Mismatch(bad)
except Mismatch as e:
    raise SystemExit(f'ROLLED BACK - rows differ after copy in: {e.args[0]}')

print(f'OK: {len(MODELS)} tables copied and verified row-by-row; sequences reset.')
