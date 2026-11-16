#!/usr/bin/env bash
# Nightly backup of the production Postgres database (replaces backup_sqlite.sh
# since the move to Postgres). pg_dump takes a consistent snapshot while the site
# is running. Keeps the newest $KEEP dumps in $DEST.
#
# Restore into an empty database:
#   pg_restore --no-owner -h 127.0.0.1 -U glysmork -d glysmork db-YYYYmmdd-HHMMSS.dump
#
# Cron (installed on the server):
#   30 3 * * * /home/ubuntu/glysmork/deploy/backup_postgres.sh >> /home/ubuntu/backup.log 2>&1
#
# The password comes from ~/.pgpass (chmod 600), so it never shows up in `ps`.
# NOTE: these copies live on the same disk as the database. They protect against
# mistakes and corruption, NOT against losing the server — also copy them off the
# machine (e.g. to S3) once AWS access is set up.
set -euo pipefail

DEST="${DEST:-/home/ubuntu/db-backups}"
KEEP="${KEEP:-14}"
PGHOST="${PGHOST:-127.0.0.1}"
PGUSER="${PGUSER:-glysmork}"
PGDATABASE="${PGDATABASE:-glysmork}"
export PGHOST PGUSER PGDATABASE

mkdir -p "$DEST"
chmod 700 "$DEST"
umask 077
OUT="$DEST/db-$(date -u +%Y%m%d-%H%M%S).dump"

pg_dump --format=custom --compress=9 --no-password --file="$OUT.part"
# make sure the dump is readable before keeping it
pg_restore --list "$OUT.part" > /dev/null
mv "$OUT.part" "$OUT"

# keep only the newest $KEEP Postgres dumps (old SQLite backups are left alone)
ls -1t "$DEST"/db-*.dump 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -f
echo "$(date -u +%FT%TZ) backup ok: $OUT ($(du -h "$OUT" | cut -f1)); kept $(ls -1 "$DEST"/db-*.dump | wc -l)"
