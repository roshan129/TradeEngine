from fastapi import FastAPI

from tradeengine.api.health import router as health_router
from tradeengine.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)
    app.include_router(health_router)
    return app


app = create_app()

