from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.utils.version import APP_VERSION

router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", version=APP_VERSION)
