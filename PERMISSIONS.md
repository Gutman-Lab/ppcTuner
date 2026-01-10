# User Permissions Setup

This document explains how user permissions are handled in the PPC Tuner V2 Docker setup.

## Overview

The Docker setup uses user ID mapping to ensure files created in containers have the correct permissions on the host system. This prevents permission issues when mounting volumes.

## Finding Your User ID

To find your user and group IDs:

```bash
id -u  # Your user ID (e.g., 501)
id -g  # Your group ID (e.g., 20)
```

## Configuration

Set these in your `.env` file or as environment variables:

```env
USER_ID=501
GROUP_ID=20
```

## Service-Specific Setup

### Backend (FastAPI)

- **Runs as**: Non-root user (`appuser`) with your specified UID/GID
- **Why**: Ensures files created in mounted volumes (`./backend:/app`) have correct permissions
- **Dockerfile**: Creates `appuser` with matching UID/GID, switches to this user before starting
- **docker-compose**: Sets `user: "${USER_ID:-1000}:${GROUP_ID:-1000}"`

### Frontend Production (Nginx)

- **Runs as**: Root (standard for nginx containers)
- **Why**: Nginx needs root privileges to bind to port 80
- **Safety**: Production containers only have read-only mounts, so this is safe
- **Cache directories**: Created with proper permissions via entrypoint script
- **docker-compose**: No `user` directive (runs as root)

### Frontend Development (Vite)

- **Runs as**: Non-root user with your specified UID/GID
- **Why**: Ensures files created during development have correct permissions
- **Dockerfile.dev**: Creates `appuser` with matching UID/GID
- **docker-compose.dev**: Sets `user: "${USER_ID:-1000}:${GROUP_ID:-1000}"`

## Troubleshooting

### Permission Denied Errors

If you see permission errors:

1. **Check your user ID**:
   ```bash
   id -u
   id -g
   ```

2. **Set in `.env` file**:
   ```env
   USER_ID=501
   GROUP_ID=20
   ```

3. **Rebuild containers**:
   ```bash
   docker-compose down
   docker-compose build --no-cache
   docker-compose up
   ```

### Files Created as Root

If files are created as root:

- **Backend**: Ensure `user:` directive is set in docker-compose.yml
- **Frontend Dev**: Ensure `user:` directive is set in docker-compose.dev.yml
- **Frontend Prod**: This is expected (nginx runs as root, but only serves static files)

### Nginx Cache Directory Errors

If nginx can't create cache directories:

- The entrypoint script (`docker-entrypoint-nginx.sh`) should handle this
- If issues persist, check that the script is executable and copied correctly
- Nginx runs as root, so it should have permissions to create these directories

## Best Practices

1. **Always set USER_ID and GROUP_ID** to match your host user
2. **Use `.env` file** to avoid hardcoding values
3. **Rebuild after changing user IDs** to ensure proper setup
4. **Check file ownership** after container operations:
   ```bash
   ls -la ./backend  # Should show files owned by your user
   ```
