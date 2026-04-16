# config.py - Centralized Configuration
import os
import secrets
import threading
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Neo4j Configuration
NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI"),
    "username": os.getenv("NEO4J_USERNAME"),
    "password": os.getenv("NEO4J_PASSWORD"),
    "database": os.getenv("NEO4J_DATABASE", "neo4j")
}

# Supabase PostgreSQL Configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "sslmode": os.getenv("DB_SSLMODE", "require")
}

# API Configuration
API_PORT = int(os.getenv("API_PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# LLM Configuration (optional — AI features degrade gracefully without it)
LLM_CONFIG = {
    "provider": os.getenv("LLM_PROVIDER", "gemini"),
    "api_key": os.getenv("LLM_API_KEY"),
    "model": os.getenv("LLM_MODEL", "gemini-2.0-flash"),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "4096")),
}

# Validation: Ensure critical configs are set
WEAK_JWT_SECRETS = {
    "",
    "change-this-to-a-random-secret-in-production",
    "legalwiz-dev-secret-change-in-production",
}

def validate_config():
    """Validate that all required environment variables are set"""
    required_vars = [
        "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD",
        "DB_HOST", "DB_USER", "DB_PASSWORD",
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please create a .env file based on .env.example"
        )
    
    # Warn (but don't block) if JWT secret is weak — only block in production
    jwt_secret = os.getenv("JWT_SECRET_KEY", "")
    auth_required = os.getenv("AUTH_REQUIRED", "false").lower() == "true"
    if auth_required and jwt_secret in WEAK_JWT_SECRETS:
        raise EnvironmentError(
            "JWT_SECRET_KEY must be set to a strong random value when AUTH_REQUIRED=true.\n"
            f"Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
        )

# Run validation on import
validate_config()


# ==================== CONNECTION POOL ====================
# ThreadedConnectionPool is safe across threads (Gunicorn workers each get
# their own process, so the pool is per-process as intended).
_pg_pool = None
_pg_pool_lock = threading.Lock()


def _get_pool():
    """Return (or lazily create) the shared Postgres connection pool."""
    global _pg_pool
    if _pg_pool is None:
        with _pg_pool_lock:
            if _pg_pool is None:  # double-checked locking
                from psycopg2 import pool as pg_pool
                _pg_pool = pg_pool.ThreadedConnectionPool(
                    minconn=2,
                    maxconn=10,
                    **DB_CONFIG
                )
    return _pg_pool


@contextmanager
def get_connection():
    """
    Context manager that borrows a connection from the pool and returns it
    when the block exits (even on error).  Use like:

        with get_connection() as conn:
            with conn.cursor() as cur:
                ...
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ==================== NEO4J SINGLETON DRIVER ====================
_neo4j_driver = None
_neo4j_driver_lock = threading.Lock()


def get_neo4j_driver():
    """
    Return (or lazily create) the shared Neo4j driver.
    The driver manages its own connection pool internally; do NOT call
    driver.close() in request handlers — only call session.close().
    """
    global _neo4j_driver
    if _neo4j_driver is None:
        with _neo4j_driver_lock:
            if _neo4j_driver is None:
                from neo4j import GraphDatabase
                _neo4j_driver = GraphDatabase.driver(
                    NEO4J_CONFIG["uri"],
                    auth=(NEO4J_CONFIG["username"], NEO4J_CONFIG["password"]),
                )
    return _neo4j_driver


def close_connections():
    """Graceful shutdown — call from app lifespan/atexit handler."""
    global _pg_pool, _neo4j_driver
    if _pg_pool is not None:
        _pg_pool.closeall()
        _pg_pool = None
    if _neo4j_driver is not None:
        _neo4j_driver.close()
        _neo4j_driver = None


# ==================== LEGACY HELPER ====================

def get_db():
    """Legacy per-request connection. Prefer get_connection() context manager."""
    import psycopg2 as _pg
    return _pg.connect(**DB_CONFIG)


# ==================== OWNERSHIP HELPER ====================

def verify_contract_ownership(contract_id: str, user_id: str):
    """
    Verify a contract exists and belongs to the given user.
    Returns the contract row dict, or raises HTTP 404.
    """
    from fastapi import HTTPException
    from psycopg2.extras import RealDictCursor

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM contracts WHERE id = %s AND created_by = %s",
                (contract_id, user_id),
            )
            contract = cur.fetchone()
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    return dict(contract)
