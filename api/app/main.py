"""
NTSB Aviation Accident API

FastAPI application for querying 64 years of NTSB aviation accident data (1962-2025).

Documentation:
- OpenAPI docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.config import settings
from app.routers import events, statistics, search, health, geospatial
from app.database import test_database_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="NTSB Aviation Accident API",
    description="""
    REST API for 64 years of NTSB aviation accident data (1962-2025).

    **Features**:
    - 179,809 events across 64 years
    - Full-text search across accident narratives
    - Statistical summaries and trends
    - Geospatial queries (PostGIS)
    - Pagination and filtering

    **Data Source**: National Transportation Safety Board (NTSB)

    **Database**: PostgreSQL 18.0 with PostGIS

    **Coverage**:
    - Events: 1962-2025
    - States: All 50 US states + territories
    - Aircraft: 94,533 aircraft across 971 makes/models
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "NTSB Aviation Database Project",
        "url": "https://github.com/your-repo/ntsb-aviation-database",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("Starting NTSB Aviation Accident API v1.0.0")
    logger.info(f"Environment: {settings.log_level}")
    logger.info(f"CORS origins: {settings.cors_origins_list}")

    # Test database connection
    db_connected = test_database_connection()
    if db_connected:
        logger.info("Database connection successful")
    else:
        logger.error("Database connection failed - API may not function correctly")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Shutting down NTSB Aviation Accident API")


# Include routers
app.include_router(
    health.router,
    prefix="/api/v1/health",
    tags=["Health"],
)

app.include_router(
    events.router,
    prefix="/api/v1/events",
    tags=["Events"],
)

app.include_router(
    statistics.router,
    prefix="/api/v1/statistics",
    tags=["Statistics"],
)

app.include_router(
    search.router,
    prefix="/api/v1/search",
    tags=["Search"],
)

app.include_router(
    geospatial.router,
    prefix="/api/v1/geospatial",
    tags=["Geospatial"],
)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    API root endpoint.

    Returns basic API information and links to documentation.
    """
    return {
        "name": "NTSB Aviation Accident API",
        "version": "1.0.0",
        "description": "REST API for 64 years of NTSB aviation accident data",
        "documentation": {
            "openapi": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json",
        },
        "endpoints": {
            "health": "/api/v1/health",
            "events": "/api/v1/events",
            "statistics": "/api/v1/statistics",
            "search": "/api/v1/search",
            "geospatial": "/api/v1/geospatial",
        },
        "database": {
            "events": "179,809",
            "years": "1962-2025",
            "size": "801 MB",
        },
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
