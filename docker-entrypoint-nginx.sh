#!/bin/sh
set -e

# Create nginx cache directories if they do not exist
mkdir -p /var/cache/nginx/client_temp
mkdir -p /var/cache/nginx/proxy_temp
mkdir -p /var/cache/nginx/fastcgi_temp
mkdir -p /var/cache/nginx/uwsgi_temp
mkdir -p /var/cache/nginx/scgi_temp
mkdir -p /var/log/nginx
mkdir -p /var/run/nginx

# Set permissions (nginx runs as root, so this is fine)
chmod -R 755 /var/cache/nginx /var/log/nginx /var/run/nginx 2>/dev/null || true

# Execute the original nginx entrypoint
exec /docker-entrypoint.sh "$@"
