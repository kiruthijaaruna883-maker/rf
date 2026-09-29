"""FastAPI application entry point for Regulatory Affairs Assistant."""

from fastapi import FastAPI
from app.api.auth import router as auth_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Regulatory Affairs Assistant API",
        description="Backend API for Regulatory Affairs Assistant.",
        version="0.1.0",
    )

    # Include authentication router
    app.include_router(auth_router)

    # In FastAPI >= 0.141, include_router wraps routes in _IncludedRouter.
    # Expose the underlying APIRoutes directly in app.router.routes for
    # seamless introspection and compatibility with route inspections.
    flattened_routes = []
    for r in app.routes:
        if hasattr(r, "effective_route_contexts"):
            for ctx in r.effective_route_contexts():
                flattened_routes.append(ctx.original_route)
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
