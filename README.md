# PPC Tuner V2

A Docker-based widget for loading images from the DSA server and generating thumbnail images of slides stained with aBeta. Eventually will support tuning parameters for running positive pixel count (PPC) algorithm.

## Architecture

- **Backend**: FastAPI (Python) - handles image processing and PPC algorithm
- **Frontend**: React + Vite - uses `bdsa-react-components` for DSA integration
- **Docker**: Docker Compose setup for easy development and deployment

## Authentication Pattern

This application implements a **two-step authentication pattern** for DSA:

1. **Backend**: Uses `girder_client` to authenticate with the API key and obtain a token
2. **Frontend**: Receives the token (not the API key) from the backend and uses it for all DSA requests

This pattern:
- ✅ Keeps the API key secure on the backend (never exposed to frontend)
- ✅ Follows DSA/Girder authentication best practices
- ✅ Allows token refresh/re-authentication on the backend if needed

**Implementation:**
- Backend: `DSAClient` service (`backend/app/services/dsa_client.py`) handles API key → token exchange
- Backend endpoint: `/api/config` returns the token (not the API key)
- Frontend: Uses the token in `Girder-Token` header for all DSA API calls

**Note:** The `bdsa-react-components` library's `DsaAuthManager` component doesn't currently handle API key → token exchange automatically. This pattern could be a useful addition to the library for applications that need to authenticate with API keys.

See [`BDSA_REACT_COMPONENTS_IMPROVEMENTS.md`](./BDSA_REACT_COMPONENTS_IMPROVEMENTS.md) for a comprehensive list of suggested improvements to the library based on this implementation.

## Authentication Pattern

This application uses a **two-step authentication pattern** for DSA:

1. **Backend**: Uses `girder_client` to authenticate with the API key and obtain a token
2. **Frontend**: Receives the token (not the API key) from the backend and uses it for all DSA requests

This pattern:
- ✅ Keeps the API key secure on the backend (never exposed to frontend)
- ✅ Follows DSA/Girder authentication best practices
- ✅ Allows token refresh/re-authentication on the backend if needed

**Implementation:**
- Backend: `DSAClient` service (`backend/app/services/dsa_client.py`) handles API key → token exchange
- Backend endpoint: `/api/config` returns the token (not the API key)
- Frontend: Uses the token in `Girder-Token` header for all DSA API calls

## Prerequisites

- Docker and Docker Compose
- Node.js (for local development, optional)
- Python 3.11+ (for local development, optional)

## Setup

### 1. Link bdsa-react-components (if using local version)

If you're using a local version of `bdsa-react-components`:

```bash
# In the bdsaReactComponents directory
npm link

# In this project directory
npm link bdsa-react-components
```

### 2. Environment Variables

Create a `.env` file in the project root (optional):

```env
DSA_BASE_URL=http://bdsa.pathology.emory.edu:8080/api/v1
DSAKEY=your_dsa_key_here
DSA_START_FOLDER=695d6a148c871f3a02969b00
DSA_START_FOLDERTYPE=collection
USER_ID=1000
GROUP_ID=1000
```

**DSA Start Folder Configuration:**
- `DSA_START_FOLDER`: The ID of the collection or folder to start browsing from
- `DSA_START_FOLDERTYPE`: Either `"collection"` or `"folder"` (defaults to `"collection"`)

**Important:** Set `USER_ID` and `GROUP_ID` to match your host user to avoid permission issues with mounted volumes. Check your IDs with:
```bash
id -u  # Your user ID
id -g  # Your group ID
```

**Note on Permissions:**
- **Backend**: Runs as the specified user (appuser) to ensure proper file permissions on mounted volumes
- **Frontend (Production)**: Nginx runs as root (standard practice) since it needs to bind to port 80. This is safe as production containers only have read-only mounts.
- **Frontend (Development)**: Vite dev server runs as the specified user for proper file permissions

### 3. Build and Run with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

The application will be available at:
- Frontend: http://localhost:8080 (host port, no conflict with viteReg on port 80)
- Backend API: http://localhost:8001 (host port, no conflict with viteReg on port 8000)
- API Docs: http://localhost:8001/docs

**Note on Ports**: 
- Frontend container listens on port 80 internally (standard nginx), mapped to host port 8080
- Backend container listens on port 8000 internally, mapped to host port 8001
- This ensures no conflicts with viteReg which uses host ports 80 and 8000

### 4. Development Mode

For development with hot reload (recommended):

```bash
# Use the development script (sets up bind mounts for hot reload)
./run-dev.sh

# Or manually:
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

This will:
- Start backend on http://localhost:8001 with auto-reload
- Start frontend dev server on http://localhost:5174 with hot reload
- Bind mount source files so changes are reflected immediately
- Use your host user ID/GID for proper file permissions

**Alternative:** Run frontend locally (without Docker):
```bash
# Start backend in Docker
docker-compose up backend

# In another terminal, run frontend locally
npm install
npm run dev
# Frontend dev server will be at http://localhost:5174
```

## Project Structure

```
ppcTunerV2/
├── backend/
│   ├── app/
│   │   ├── api/          # API routes
│   │   │   ├── images.py # Image endpoints
│   │   │   └── ppc.py     # PPC algorithm endpoints
│   │   └── core/
│   │       └── config.py # Configuration
│   ├── main.py           # FastAPI app entry point
│   ├── requirements.txt  # Python dependencies
│   └── Dockerfile        # Backend Docker image
├── src/
│   ├── App.tsx           # Main React component
│   ├── main.tsx          # React entry point
│   └── ...
├── docker-compose.yml    # Docker Compose configuration
├── Dockerfile            # Frontend Docker image
├── nginx.conf            # Nginx configuration
└── package.json          # Node.js dependencies
```

## Features

### Current
- Load images from DSA server using `FolderBrowser` component
- Display thumbnails using `ThumbnailGrid` component from `bdsa-react-components`
- Basic FastAPI backend structure

### Planned
- Generate thumbnail images of aBeta-stained slides
- Tune PPC algorithm parameters
- Process images with PPC algorithm
- View and compare results

## API Endpoints

### Images
- `GET /api/images/` - List images
- `GET /api/images/{image_id}` - Get image info
- `GET /api/images/{image_id}/thumbnail` - Get thumbnail

### PPC
- `POST /api/ppc/process` - Process image with PPC
- `GET /api/ppc/{image_id}/result` - Get PPC result

## Development

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
npm install
npm run dev
```

## Troubleshooting

### bdsa-react-components not found
- Ensure `npm link bdsa-react-components` has been run
- Check that the library is built: `cd /path/to/bdsaReactComponents && npm run build`

### Docker permissions issues
- Set `USER_ID` and `GROUP_ID` in `.env` to match your user
- Or run `docker-compose` with appropriate permissions

### Port conflicts
- Change ports in `docker-compose.yml` if 80 or 8000 are already in use

## License

[Add your license here]
