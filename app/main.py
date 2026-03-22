import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routers import analyze, health
from app.utils.version import APP_VERSION

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Animal Eye Metrology API",
    description=(
        "Image segmentation + metrology pipeline for measuring "
        "inter-ocular and inter-animal eye distances using COCO dataset."
    ),
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Ensure output directory exists
settings.output_dir.mkdir(parents=True, exist_ok=True)

# Mount static files for serving annotated images
app.mount("/outputs", StaticFiles(directory=str(settings.output_dir)), name="outputs")

# Register routers
app.include_router(health.router)
app.include_router(analyze.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        reload_dirs=["app"] if settings.debug else None,
    )
