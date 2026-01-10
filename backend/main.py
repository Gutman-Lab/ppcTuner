"""
FastAPI backend for PPC Tuner application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api import images, ppc
from app.core.config import settings
from app.services.dsa_client import DSAClient

# Initialize DSA client (will authenticate on startup)
dsa_client = DSAClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown tasks"""
    # Startup
    print("Starting FastAPI backend for PPC Tuner...")
    yield
    # Shutdown
    print("Shutting down FastAPI backend...")


app = FastAPI(
    title="PPC Tuner API",
    description="API for Positive Pixel Count tuning and thumbnail generation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(images.router, prefix="/api/images", tags=["images"])
app.include_router(ppc.router, prefix="/api/ppc", tags=["ppc"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "PPC Tuner API", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/api/config")
async def get_config():
    """Get application configuration for frontend"""
    # Get the token (not the API key) - token is obtained by authenticating with API key
    token = dsa_client.get_token() if dsa_client else None
    return {
        "dsaBaseUrl": settings.DSA_BASE_URL,
        "startFolder": settings.DSA_START_FOLDER,
        "startFolderType": settings.DSA_START_FOLDERTYPE,
        "dsaToken": token,  # Token obtained from API key authentication
    }
