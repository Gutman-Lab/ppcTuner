#!/bin/sh
# Development entrypoint script to fix permissions for Vite
# This script runs as root (from Dockerfile) to fix permissions

# Get the user ID from the environment or default to 1000
USER_ID=${USER_ID:-1000}
GROUP_ID=${GROUP_ID:-1000}

# Fix permissions for node_modules (needed for Vite temp files)
# This must be done as root since node_modules might be owned by root or node user
if [ -d "/app/node_modules" ]; then
    # Force ownership change - node_modules might be owned by node:node from the base image
    chown -R ${USER_ID}:${GROUP_ID} /app/node_modules
    chmod -R u+w /app/node_modules
    mkdir -p /app/node_modules/.vite-temp
    chown -R ${USER_ID}:${GROUP_ID} /app/node_modules/.vite-temp
    chmod -R u+w /app/node_modules/.vite-temp
fi

# Also ensure the app directory is owned by the correct user
chown -R ${USER_ID}:${GROUP_ID} /app 2>/dev/null || true

# Switch to appuser using gosu and execute the command
exec gosu ${USER_ID}:${GROUP_ID} "$@"
