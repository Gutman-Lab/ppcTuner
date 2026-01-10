# Multi-stage build for Vite React app
# Specify platform for consistent builds (amd64 for Linux deployment)
FROM --platform=linux/amd64 node:20-alpine AS builder

# Install git (needed for GitHub dependencies)
RUN apk add --no-cache git

WORKDIR /app

# Copy package files (package-lock.json is optional, will be generated if missing)
COPY package.json ./

# Install dependencies (including bdsa-react-components from npm)
# Note: bdsa-react-components has a peer dependency from GitHub that requires git
RUN npm install

# Copy source code
COPY . .

# Build the app
RUN npm run build

# Production stage
# Specify platform for consistent builds (amd64 for Linux deployment)
FROM --platform=linux/amd64 nginx:alpine

# Build arguments for user ID mapping
ARG USER_ID=1000
ARG GROUP_ID=1000

# Create nginx cache directories with proper permissions
# These need to be writable by the nginx user
RUN mkdir -p /var/cache/nginx/client_temp \
    /var/cache/nginx/proxy_temp \
    /var/cache/nginx/fastcgi_temp \
    /var/cache/nginx/uwsgi_temp \
    /var/cache/nginx/scgi_temp \
    /var/log/nginx \
    /var/run/nginx && \
    chmod -R 755 /var/cache/nginx \
    /var/log/nginx \
    /var/run/nginx

# Copy custom entrypoint script
COPY docker-entrypoint-nginx.sh /docker-entrypoint-nginx.sh
RUN chmod +x /docker-entrypoint-nginx.sh

# Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Set ownership of web files (nginx user can read, but we want proper permissions)
RUN chown -R nginx:nginx /usr/share/nginx/html /etc/nginx/conf.d/default.conf

EXPOSE 80

# Use custom entrypoint
ENTRYPOINT ["/docker-entrypoint-nginx.sh"]
CMD ["nginx", "-g", "daemon off;"]
