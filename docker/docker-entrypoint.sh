#!/bin/bash
set -e

if [ "$AUTO_MIGRATE" = "true" ]; then
    echo "Running database migrations..."
    alembic upgrade head
    echo "Migrations completed successfully"
fi

exec "$@"
