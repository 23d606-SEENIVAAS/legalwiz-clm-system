# error_handlers.py - Global Exception Handlers
"""
Structured error responses and global exception handling.
Catches unhandled errors and returns consistent JSON.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import psycopg2
import traceback
import logging

logger = logging.getLogger("legalwiz")


def register_error_handlers(app: FastAPI):
    """Register all global exception handlers on the FastAPI app."""

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        """Pydantic validation errors → user-friendly messages."""
        errors = []
        for error in exc.errors():
            field = " → ".join(str(loc) for loc in error.get("loc", []))
            errors.append({
                "field": field,
                "message": error.get("msg", "Invalid value"),
                "type": error.get("type", "unknown"),
            })

        return JSONResponse(
            status_code=422,
            content={
                "detail": "Validation error",
                "errors": errors,
                "hint": "Check field names and types in your request body",
            },
        )

    @app.exception_handler(psycopg2.OperationalError)
    async def db_connection_handler(request: Request, exc: psycopg2.OperationalError):
        """Database connection failures."""
        logger.error(f"Database connection error: {exc}")
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Database unavailable. Please try again later.",
                "error_type": "database_connection",
            },
        )

    @app.exception_handler(psycopg2.IntegrityError)
    async def db_integrity_handler(request: Request, exc: psycopg2.IntegrityError):
        """Constraint violations (duplicate keys, FK violations)."""
        msg = str(exc)
        if "unique" in msg.lower() or "duplicate" in msg.lower():
            detail = "A record with this value already exists"
        elif "foreign" in msg.lower():
            detail = "Referenced record does not exist"
        elif "not-null" in msg.lower() or "null" in msg.lower():
            detail = "A required field is missing"
        else:
            detail = "Data integrity error"

        return JSONResponse(
            status_code=409,
            content={
                "detail": detail,
                "error_type": "integrity_error",
            },
        )

    @app.exception_handler(Exception)
    async def global_error_handler(request: Request, exc: Exception):
        """Catch-all for unhandled exceptions."""
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(
            f"[{request_id}] Unhandled error on {request.method} {request.url.path}: "
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        )

        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "error_type": type(exc).__name__,
                "request_id": request_id,
                "hint": "Check server logs for details",
            },
        )
