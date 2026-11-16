#!/usr/bin/env bash
# Nightly backup of the production SQLite database.
#
# Uses SQLite's online backup API, so it is safe while the site is running (a plain
# `cp` of a live SQLite/WAL database can produce a corrupt copy). Keeps the newest
# $KEEP compressed copies in $DEST.
#
# Cron (installed on the server):
#   30 3 * * * /home/ubuntu/glysmork/deploy/backup_sqlite.sh >> /home/ubuntu/backup.log 2>&1
#
# NOTE: these copies live on the same disk as the database. They protect against
# mistakes and corruption, NOT against losing the server — also copy them off the
# machine (e.g. to S3) once AWS access is set up.
set -euo pipefail

SRC="${SRC:-/home/ubuntu/glysmork/db.sqlite3}"
DEST="${DEST:-/home/ubuntu/db-backups}"
KEEP="${KEEP:-14}"
PY="${PY:-/home/ubuntu/venv/bin/python}"

mkdir -p "$DEST"
chmod 700 "$DEST"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
OUT="$DEST/db-$STAMP.sqlite3"

"$PY" - "$SRC" "$OUT" <<'PYEOF'
import sqlite3, sys
src, out = sys.argv[1], sys.argv[2]
s = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=60)
d = sqlite3.connect(out)
s.backup(d)
ok = d.execute("PRAGMA integrity_check").fetchone()[0]
d.close(); s.close()
if ok != "ok":
    raise SystemExit(f"backup integrity check failed: {ok}")
PYEOF

gzip -9 "$OUT"
chmod 600 "$OUT.gz"
# keep only the newest $KEEP backups
ls -1t "$DEST"/db-*.sqlite3.gz 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -f
echo "$(date -u +%FT%TZ) backup ok: $OUT.gz ($(du -h "$OUT.gz" | cut -f1)); kept $(ls -1 "$DEST"/db-*.sqlite3.gz | wc -l)"
