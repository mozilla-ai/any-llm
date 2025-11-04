from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from any_llm.gateway import __version__
from any_llm.gateway.db import get_db

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check() -> dict[str, str]:
    """General health check endpoint.

    Returns basic health status. For infrastructure monitoring,
    use /health/readiness or /health/liveness instead.
    """
    return {"status": "healthy"}


@router.get("/liveness")
async def health_liveness() -> str:
    """Liveness probe endpoint.

    Simple check to verify the process is alive and responding.
    Used by Kubernetes/container orchestrators for liveness probes.

    Returns:
        Plain text "I'm alive!" message

    """
    return "I'm alive!"


@router.get("/readiness")
async def health_readiness() -> dict[str, Any]:
    """Readiness probe endpoint.

    Checks if the gateway is ready to serve requests by validating:
    - Database connectivity
    - Service availability

    Used by Kubernetes/container orchestrators for readiness probes.
    Returns HTTP 503 if any dependency is unavailable.

    Returns:
        dict: Status object with health details

    Raises:
        HTTPException: 503 if service is not ready

    """
    try:
        db_gen = get_db()
        db = next(db_gen)
        try:
            db.execute(text("SELECT 1"))
            db_status = "connected"
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "database": "error",
                "error": str(e),
                "version": __version__,
            },
        ) from e
    return {
        "status": "healthy",
        "database": db_status,
        "version": __version__,
    }
