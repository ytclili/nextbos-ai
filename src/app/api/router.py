from fastapi import APIRouter

from app.api.routes.chat import router as chat_router
from app.api.routes.conversations import router as conversations_router
from app.api.routes.health import router as health_router

router = APIRouter()
router.include_router(health_router)
router.include_router(chat_router, prefix="/api/v1")
router.include_router(conversations_router, prefix="/api/v1")
