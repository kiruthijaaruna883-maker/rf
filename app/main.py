"""FastAPI application entry point for Regulatory Affairs Assistant."""

from fastapi import FastAPI
from fastapi.routing import request_response
from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Regulatory Affairs Assistant API",
        description="Backend API for Regulatory Affairs Assistant.",
        version="0.1.0",
    )

    # Include routers
    app.include_router(auth_router)
    app.include_router(documents_router)
    app.include_router(chat_router)

    # In FastAPI >= 0.141, include_router wraps routes in _IncludedRouter.
    # Expose the underlying APIRoutes directly in app.router.routes for
    # seamless introspection and compatibility with route inspections.
    flattened_routes = []
    for r in app.routes:
        if hasattr(r, "effective_route_contexts"):
            for ctx in r.effective_route_contexts():
                route = ctx.original_route
                route.dependency_overrides_provider = app
                route.app = request_response(route.get_route_handler())
                flattened_routes.append(route)
        else:
            flattened_routes.append(r)
    app.router.routes = flattened_routes

    # Root endpoint for health and verification
    @app.get("/", tags=["Health"])
    def root():
        return {
            "status": "online",
            "message": "Regulatory Affairs Assistant API",
        }

    return app


app = create_app()
