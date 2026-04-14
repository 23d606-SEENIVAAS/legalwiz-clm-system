# auth_routes.py - User Registration & Login
"""
JWT-based authentication endpoints.
Auto-creates `users` table on first use.

Endpoints:
- POST /api/auth/register  — Create user account
- POST /api/auth/login     — Get access + refresh tokens
- GET  /api/auth/me        — Get current user profile
- POST /api/auth/refresh   — Refresh access token
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
import uuid

from passlib.context import CryptContext

from config import get_db, DB_CONFIG
from auth_middleware import ACCESS_TOKEN_EXPIRE_MINUTES
import psycopg2
from psycopg2.extras import RealDictCursor

from auth_middleware import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    REFRESH_TOKEN_EXPIRE_DAYS,
)

router = APIRouter(prefix="/api/auth", tags=["authentication"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ==================== TABLE AUTO-CREATION ====================


def _ensure_table():
    """Create users + revoked_tokens tables if they don't exist. Called once at startup."""

    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    is_active BOOLEAN NOT NULL DEFAULT true,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            # Create index for email lookups
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)
            """)
            # Revoked tokens table (for refresh token invalidation)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS revoked_tokens (
                    jti TEXT PRIMARY KEY,
                    revoked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    expires_at TIMESTAMPTZ NOT NULL
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_revoked_tokens_expires
                ON revoked_tokens(expires_at)
            """)
            conn.commit()
    finally:
        conn.close()


# ==================== PYDANTIC MODELS ====================

class RegisterRequest(BaseModel):
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., min_length=8, max_length=128, description="Min 8 characters")
    full_name: str = Field(..., min_length=2, max_length=100)
    # Role is always 'user' on self-registration. Admin promotion is a separate operation.


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


class RefreshRequest(BaseModel):
    refresh_token: str


class UserProfile(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime


# ==================== HELPERS ====================



# ==================== ROUTES ====================

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest):
    """
    Create a new user account.
    Returns access + refresh tokens on success.
    """

    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check if email already exists
            cur.execute("SELECT id FROM users WHERE email = %s", (request.email.lower(),))
            if cur.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Email already registered"
                )

            # Hash password
            password_hash = pwd_context.hash(request.password)

            # Insert user
            user_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO users (id, email, password_hash, full_name, role)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, email, full_name, role, is_active, created_at
            """, (user_id, request.email.lower(), password_hash, request.full_name, "user"))

            user = cur.fetchone()
            conn.commit()

        # Generate tokens
        token_data = {"sub": str(user["id"]), "email": user["email"], "role": user["role"]}
        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "user": {
                "id": str(user["id"]),
                "email": user["email"],
                "full_name": user["full_name"],
                "role": user["role"],
            }
        }
    finally:
        conn.close()


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Authenticate with email + password.
    Returns access + refresh tokens.
    """

    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, email, password_hash, full_name, role, is_active
                FROM users WHERE email = %s
            """, (request.email.lower(),))
            user = cur.fetchone()
    finally:
        conn.close()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated"
        )

    # Verify password
    if not pwd_context.verify(request.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Generate tokens
    token_data = {"sub": str(user["id"]), "email": user["email"], "role": user["role"]}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "user": {
            "id": str(user["id"]),
            "email": user["email"],
            "full_name": user["full_name"],
            "role": user["role"],
        }
    }


@router.get("/me", response_model=UserProfile)
async def get_profile(user=Depends(get_current_user)):
    """Get current user's profile. Requires authentication."""


    # In dev mode, return mock user
    if user.get("email") == "dev@legalwiz.local":
        return {
            **user,
            "is_active": True,
            "created_at": datetime.now(),
        }

    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, email, full_name, role, is_active, created_at
                FROM users WHERE id = %s
            """, (user["id"],))
            profile = cur.fetchone()
            if not profile:
                raise HTTPException(status_code=404, detail="User not found")
        return {**dict(profile), "id": str(profile["id"])}
    finally:
        conn.close()


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshRequest):
    """
    Get a new access token using a valid refresh token.
    """
    payload = decode_token(request.refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type. Provide a refresh token."
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    # Look up user to ensure still active
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check if this refresh token has been revoked
            jti = payload.get("jti")
            if not jti:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token (missing jti)"
                )
            cur.execute(
                "SELECT 1 FROM revoked_tokens WHERE jti = %s", (jti,)
            )
            if cur.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Refresh token has been revoked"
                )

            cur.execute("""
                SELECT id, email, full_name, role, is_active
                FROM users WHERE id = %s
            """, (user_id,))
            user = cur.fetchone()
    finally:
        conn.close()

    if not user or not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated"
        )

    # Generate new tokens
    token_data = {"sub": str(user["id"]), "email": user["email"], "role": user["role"]}
    new_access = create_access_token(token_data)
    new_refresh = create_refresh_token(token_data)

    return {
        "access_token": new_access,
        "refresh_token": new_refresh,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "user": {
            "id": str(user["id"]),
            "email": user["email"],
            "full_name": user["full_name"],
            "role": user["role"],
        }
    }


@router.post("/logout")
async def logout(request: RefreshRequest):
    """
    Revoke a refresh token so it can no longer be used.
    Call this on user logout to prevent stolen-token replay.
    """


    payload = decode_token(request.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide a refresh token"
        )

    jti = payload.get("jti")
    if not jti:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid refresh token (missing jti)"
        )
    exp = payload.get("exp")
    from datetime import timezone as tz, timedelta
    expires_at = datetime.fromtimestamp(exp, tz=tz.utc) if exp else (datetime.now(tz.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))

    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO revoked_tokens (jti, expires_at)
                VALUES (%s, %s)
                ON CONFLICT (jti) DO NOTHING
            """, (jti, expires_at))
            conn.commit()
    finally:
        conn.close()

    return {"message": "Token revoked successfully"}
