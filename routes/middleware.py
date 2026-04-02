# middleware.py - Rate Limiting & Request ID Middleware
"""
In-memory rate limiter for FastAPI using pure ASGI middleware.
Uses a sliding window counter per IP address.

Rate limits:
- Default:  60 req/min   (general API)
- AI:       20 req/min   (LLM-powered endpoints)
- Export:   10 req/min   (PDF/DOCX generation)
"""

import time
import uuid
import json
from collections import defaultdict
from typing import Dict, Tuple


class RateLimiter:
    """Thread-safe in-memory sliding window rate limiter."""

    def __init__(self):
        self._windows: Dict[str, list] = defaultdict(list)
        self._window_size = 60  # 1 minute

    def is_allowed(self, key: str, max_requests: int) -> Tuple[bool, int, int]:
        """
        Check if a request is allowed.
        Returns (allowed, remaining, retry_after_seconds).
        """
        now = time.time()
        window_start = now - self._window_size

        # Clean old entries
        self._windows[key] = [
            t for t in self._windows[key] if t > window_start
        ]

        current_count = len(self._windows[key])

        if current_count >= max_requests:
            oldest = min(self._windows[key]) if self._windows[key] else now
            retry_after = int(oldest + self._window_size - now) + 1
            return False, 0, max(retry_after, 1)

        self._windows[key].append(now)
        remaining = max_requests - current_count - 1
        return True, remaining, 0


# Path-based rate limit configuration
RATE_LIMITS = {
    "/recommendations": 20,
    "/customize": 20,
    "/risk-analysis": 20,
    "/chat": 20,
    "/export/pdf": 10,
    "/export/docx": 10,
    "/api/auth/login": 10,
    "/api/auth/register": 5,
}

DEFAULT_RATE_LIMIT = 60

rate_limiter = RateLimiter()


def _get_rate_limit(path: str) -> int:
    """Match a request path to its rate limit."""
    for pattern, limit in RATE_LIMITS.items():
        if pattern in path:
            return limit
    return DEFAULT_RATE_LIMIT


def _get_client_ip(scope) -> str:
    """Get client IP from ASGI scope."""
    client = scope.get("client")
    return client[0] if client else "unknown"


def setup_middleware(app):
    """
    Register all middleware on the FastAPI app using @app.middleware("http").
    This approach avoids the BaseHTTPMiddleware hanging issues.
    """

    @app.middleware("http")
    async def request_id_middleware(request, call_next):
        """Add X-Request-ID to all requests/responses."""
        request_id = request.headers.get("x-request-id", str(uuid.uuid4())[:8])
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.middleware("http")
    async def rate_limit_middleware(request, call_next):
        """Rate limiting with headers."""
        path = request.url.path

        # Skip for docs
        if path in ("/docs", "/redoc", "/openapi.json", "/health"):
            return await call_next(request)

        client_ip = _get_client_ip(request.scope)
        max_requests = _get_rate_limit(path)
        key = f"{client_ip}:{path}"

        allowed, remaining, retry_after = rate_limiter.is_allowed(key, max_requests)

        if not allowed:
            from starlette.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests. Please try again later.",
                    "retry_after": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(max_requests),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
