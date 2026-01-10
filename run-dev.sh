#!/bin/bash
# Docker development startup script for PPC Tuner V2

set -e

# Check for force rebuild flag
FORCE_REBUILD=false
if [[ "$1" == "--rebuild" ]] || [[ "$1" == "-r" ]]; then
    FORCE_REBUILD=true
    echo "Force rebuild requested..."
fi

echo "PPC Tuner V2 - Docker Development Stack"
echo "======================================="
echo ""

# Get current user's UID and GID for proper file permissions
# This ensures files created in containers match host user ownership
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)

echo "Using user ID: $USER_ID, group ID: $GROUP_ID"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found"
    echo "You may want to create one with:"
    echo "  DSAKEY=your_api_key_here"
    echo "  DSA_START_FOLDER=695d6a148c871f3a02969b00"
    echo "  DSA_START_FOLDERTYPE=collection"
    echo ""
fi

# Check if images exist (simple check - docker-compose will handle smart rebuilding)
COMPOSE_CMD="docker-compose -f docker-compose.yml -f docker-compose.dev.yml"

# Check if images exist for our services
BACKEND_IMAGE_EXISTS=false
FRONTEND_IMAGE_EXISTS=false

# Get project name (docker-compose uses directory name)
PROJECT_NAME=$(basename "$(pwd)" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g')

# Check if backend image exists
if docker images --format "{{.Repository}}" | grep -q "^${PROJECT_NAME}_backend$"; then
    BACKEND_IMAGE_EXISTS=true
fi

# Check if frontend image exists  
if docker images --format "{{.Repository}}" | grep -q "^${PROJECT_NAME}_frontend$"; then
    FRONTEND_IMAGE_EXISTS=true
fi

# Determine if we need to build
NEEDS_BUILD=false
if [ "$FORCE_REBUILD" = true ]; then
    NEEDS_BUILD=true
    echo "Force rebuild requested"
elif [ "$BACKEND_IMAGE_EXISTS" = false ] || [ "$FRONTEND_IMAGE_EXISTS" = false ]; then
    NEEDS_BUILD=true
    if [ "$BACKEND_IMAGE_EXISTS" = false ]; then
        echo "Backend image not found, will build"
    fi
    if [ "$FRONTEND_IMAGE_EXISTS" = false ]; then
        echo "Frontend image not found, will build"
    fi
else
    echo "Images exist - docker-compose will use cached layers if nothing changed"
    echo "(Use './run-dev.sh --rebuild' to force a full rebuild)"
fi
echo ""

echo "Starting in DEVELOPMENT mode..."
echo "Frontend: http://localhost:5174 (Vite dev server)"
echo "Backend: http://localhost:8001 (FastAPI)"
echo "API Docs: http://localhost:8001/docs"
echo ""
echo "Files are bind-mounted for hot reload:"
echo "  - ./src -> /app/src (frontend)"
echo "  - ./backend -> /app (backend)"
echo ""

# Start containers
# Use --build only if images don't exist or force rebuild requested
# Docker-compose is smart about layer caching, so this will be fast if nothing changed
if [ "$NEEDS_BUILD" = true ]; then
    echo "Building and starting containers..."
    $COMPOSE_CMD up --build
else
    echo "Starting containers (using existing images)..."
    $COMPOSE_CMD up
fi
