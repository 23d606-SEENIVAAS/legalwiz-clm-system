# auth_middleware.py - JWT Authentication Middleware
"""
JWT-based authentication dependency for FastAPI.
Replaces the mock get_current_user() in main.py.

Usage in routes:
    from auth_middleware import get_current_user, require_role
    
    @router.get("/protected")
    async def protected_route(user = Depends(get_current_user)):
        return {"user_id": user["id"]}
    
    @router.delete("/admin-only")
    async def admin_route(user = Depends(require_role("admin"))):
        ...

Set AUTH_REQUIRED=false in .env to bypass auth during development.
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

from config import DB_CONFIG
import psycopg2
from psycopg2.extras import RealDictCursor


# ==================== CONFIG ====================

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "legalwiz-dev-secret-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_EXPIRE_DAYS", "7"))
AUTH_REQUIRED = os.getenv("AUTH_REQUIRED", "false").lower() == "true"

# Mock user for when auth is disabled
MOCK_USER = {
    "id": "11111111-1111-1111-1111-111111111111",
    "email": "dev@legalwiz.local",
    "full_name": "Dev User",
    "role": "admin",
}

security = HTTPBearer(auto_error=False)


# ==================== TOKEN UTILITIES ====================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """Create a JWT refresh token (longer lived)."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ==================== DEPENDENCIES ====================

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    """
    FastAPI dependency that extracts and validates the current user from JWT.
    
    If AUTH_REQUIRED=false, returns a mock dev user (for development).
    """
    # Dev mode: skip auth
    if not AUTH_REQUIRED:
        return MOCK_USER

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(credentials.credentials)

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type. Use an access token.",
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing user ID",
        )

    # Look up user in DB
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, email, full_name, role, is_active
                FROM users WHERE id = %s
            """, (user_id,))
            user = cur.fetchone()
    finally:
        conn.close()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    return {
        "id": str(user["id"]),
        "email": user["email"],
        "full_name": user["full_name"],
        "role": user["role"],
    }


def require_role(*allowed_roles: str):
    """
    Factory that returns a dependency requiring specific roles.
    
    Usage: Depends(require_role("admin", "user"))
    """
    async def role_checker(
        user: Dict = Depends(get_current_user),
    ) -> Dict:
        if not AUTH_REQUIRED:
            return user
        if user.get("role") not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {', '.join(allowed_roles)}. You have: {user.get('role')}",
            )
        return user
    return role_checker
