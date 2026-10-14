"""FastAPI application factory."""

import platform

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from oratio_api import __version__, analysis, judges
from oratio_api.settings import Settings, get_settings


class Health(BaseModel):
    status: str
    version: str
    arch: str
    image_digest: str


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="Oratio API", version=__version__, docs_url="/api/docs", redoc_url=None)

    if settings.cors_origins or settings.cors_origin_regex:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_origin_regex=settings.cors_origin_regex,
            allow_methods=["GET", "POST"],
            allow_headers=["content-type"],
        )

    @app.get("/api/health")
    def health() -> Health:
        return Health(
            status="ok",
            version=__version__,
            arch=platform.machine(),
            image_digest=settings.image_digest,
        )

    app.include_router(analysis.build_router(settings), prefix="/api")
    app.include_router(judges.build_router(settings), prefix="/api")

    if settings.enable_spike:
        from oratio_api.spike.router import build_router

        app.include_router(build_router(settings), prefix="/api/spike")

    return app


app = create_app()
