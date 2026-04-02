# config.py - Centralized Configuration
import os
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
def validate_config():
    """Validate that all required environment variables are set"""
    required_vars = [
        "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD",
        "DB_HOST", "DB_USER", "DB_PASSWORD"
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please create a .env file based on .env.example"
        )

# Run validation on import
validate_config()
