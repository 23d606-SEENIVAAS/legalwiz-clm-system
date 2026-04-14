# main.py - FastAPI Backend for Contracts (main table)
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum
from psycopg2.extras import RealDictCursor
from datetime import datetime
import uuid
from contextlib import asynccontextmanager
from config import DB_CONFIG, NEO4J_CONFIG, get_connection, close_connections

from party_routes import router as party_routes
from neo4j_routes import router as neo4j_routes
from parameters_routes import router as parameters_router
from contract_generation_routes import router as generation_router

# AI Feature Routes (Graph RAG)
from recommendation_routes import router as recommendation_router
from customization_routes import router as customization_router
from risk_routes import router as risk_router
from chatbot_routes import router as chatbot_router

# Export & Utility Routes
from export_routes import router as export_router
from template_routes import router as template_router
from version_routes import router as version_router
from esign_routes import router as esign_router

# Auth
from auth_routes import router as auth_router
from auth_middleware import get_current_user

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create all required tables/columns once (not per-worker)
    import logging
    logger = logging.getLogger("legalwiz.startup")
    logger.info("Running startup table creation...")
    try:
        from auth_routes import _ensure_table as ensure_users_table
        ensure_users_table()
    except Exception as e:
        logger.warning(f"Users table setup: {e}")
    try:
        from chatbot_routes import _ensure_chat_table as ensure_chat_table
        ensure_chat_table()
    except Exception as e:
        logger.warning(f"Chat table setup: {e}")
    try:
        from template_routes import _ensure_table as ensure_template_table
        ensure_template_table()
    except Exception as e:
        logger.warning(f"Template table setup: {e}")
    try:
        from esign_routes import _ensure_columns as ensure_esign_columns
        ensure_esign_columns()
    except Exception as e:
        logger.warning(f"E-sign columns setup: {e}")
    logger.info("Startup table creation complete.")
    yield
    # Shutdown: cleanly close Neo4j singleton driver + Postgres pool
    try:
        from graph_rag_engine import close_driver
        close_driver()
    except Exception:
        pass
    close_connections()

app = FastAPI(
    title="LegalWiz CLM API",
    version="3.0.0",
    description="Contract Lifecycle Management with Graph RAG AI",
    lifespan=lifespan,
)

# CORS — restrict to known frontend origins
# Add your production domain(s) here when deploying
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

# Security Middleware (rate limiting + request ID)
from middleware import setup_middleware
setup_middleware(app)

# Global Error Handlers
from error_handlers import register_error_handlers
register_error_handlers(app)

# Core Routes
app.include_router(party_routes)
app.include_router(neo4j_routes)
app.include_router(parameters_router)
app.include_router(generation_router)

# AI Feature Routes (Graph RAG)
app.include_router(recommendation_router)
app.include_router(customization_router)
app.include_router(risk_router)
app.include_router(chatbot_router)

# Export & Utility Routes
app.include_router(export_router)
app.include_router(template_router)
app.include_router(version_router)
app.include_router(esign_router)

# Auth
app.include_router(auth_router)


# Pydantic Models (matches your contracts table)
class ContractType(str, Enum):
    EMPLOYMENT_NDA = "employment_nda"
    SAAS_SERVICE_AGREEMENT = "saas_service_agreement"
    CONSULTING_SERVICE_AGREEMENT = "consulting_service_agreement"
    SOFTWARE_LICENSE_AGREEMENT = "software_license_agreement"
    DATA_PROCESSING_AGREEMENT = "data_processing_agreement"
    VENDOR_AGREEMENT = "vendor_agreement"
    PARTNERSHIP_AGREEMENT = "partnership_agreement"

class ContractStatus(str, Enum):
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    SIGNED = "signed"
    ACTIVE = "active"
    TERMINATED = "terminated"

class Jurisdiction(str, Enum):
    INDIA = "India"
    USA = "USA"
    UK = "UK"

class CreateContract(BaseModel):
    title: str
    contract_type: ContractType
    jurisdiction: Jurisdiction = Jurisdiction.INDIA
    description: Optional[str] = None
    tags: Optional[List[str]] = None

class UpdateContract(BaseModel):
    title: Optional[str] = None
    status: Optional[ContractStatus] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None

class ContractResponse(BaseModel):
    id: str
    title: str
    contract_type: ContractType
    jurisdiction: Jurisdiction
    status: ContractStatus
    created_by: str
    created_at: datetime
    updated_at: datetime
    description: Optional[str] = None
    tags: Optional[List[str]] = None

# (No get_db() needed — use `get_connection()` from config directly)


# ==================== OWNERSHIP HELPER ====================

def verify_contract_ownership(contract_id: str, user_id: str, cur) -> Dict:
    """
    Verify the current user owns a contract. Returns the contract row.
    Raises 404 if not found or not owned by user.
    """
    cur.execute("""
        SELECT * FROM contracts WHERE id = %s AND created_by = %s
    """, (contract_id, user_id))
    result = cur.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Contract not found")
    return result


# ROUTES FOR CONTRACTS (MAIN TABLE)
@app.post("/api/contracts", response_model=ContractResponse, status_code=201)
async def create_contract(request: CreateContract, user=Depends(get_current_user)):
    """Create new contract"""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO contracts (
                    title, contract_type, jurisdiction, description, tags, created_by
                ) VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING *
            """, (
                request.title,
                request.contract_type,
                request.jurisdiction,
                request.description,
                request.tags,
                user["id"]
            ))
            result = cur.fetchone()
            conn.commit()
            return result

@app.get("/api/contracts/{contract_id}", response_model=ContractResponse)
async def get_contract(contract_id: str, user=Depends(get_current_user)):
    """Get single contract (must be owned by current user)"""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            return verify_contract_ownership(contract_id, user["id"], cur)

@app.get("/api/contracts", response_model=List[ContractResponse])
async def list_contracts(
    limit: int = 50,
    offset: int = 0,
    contract_type: Optional[ContractType] = None,
    status: Optional[ContractStatus] = None,
    user=Depends(get_current_user),
):
    """List contracts with filters (scoped to current user)"""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = "SELECT * FROM contracts WHERE created_by = %s"
            params = [user["id"]]
            
            if contract_type:
                query += " AND contract_type = %s"
                params.append(contract_type)
            
            if status:
                query += " AND status = %s"
                params.append(status)
            
            query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            
            cur.execute(query, params)
            return cur.fetchall()


@app.put("/api/contracts/{contract_id}", response_model=ContractResponse)
async def update_contract(contract_id: str, request: UpdateContract, user=Depends(get_current_user)):
    """Update contract (must be owned by current user)"""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            updates = []
            params = []
            
            if request.title is not None:
                updates.append("title = %s")
                params.append(request.title)
            
            if request.status is not None:
                updates.append("status = %s")
                params.append(request.status)
            
            if request.description is not None:
                updates.append("description = %s")
                params.append(request.description)
            
            if request.tags is not None:
                updates.append("tags = %s")
                params.append(request.tags)
            
            if not updates:
                raise HTTPException(status_code=400, detail="No fields to update")
            
            params.extend([user["id"], contract_id])
            
            cur.execute(f"""
                UPDATE contracts 
                SET {', '.join(updates)}, updated_at = NOW()
                WHERE created_by = %s AND id = %s
                RETURNING *
            """, params)
            
            result = cur.fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="Contract not found")
            
            conn.commit()
            return result


@app.delete("/api/contracts/{contract_id}")
async def delete_contract(contract_id: str, user=Depends(get_current_user)):
    """Delete draft contract (must be owned by current user)"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM contracts 
                WHERE id = %s AND created_by = %s AND status = 'draft'
            """, (contract_id, user["id"]))
            
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Contract not found or cannot delete")
            
            conn.commit()
            return {"message": "Contract deleted successfully"}

# Health check
@app.get("/api/health")
async def health():
    from llm_config import LLM_CONFIG as llm_cfg
    llm_ready = bool(llm_cfg.get("api_key"))
    return {
        "status": "healthy",
        "version": "3.0.0",
        "tables": "contracts ready",
        "ai_features": {
            "llm_configured": llm_ready,
            "llm_provider": llm_cfg.get("provider", "none"),
            "llm_model": llm_cfg.get("model", "none") if llm_ready else "not configured",
            "endpoints": {
                "recommendations": "/api/contracts/{id}/recommendations",
                "customization": "/api/contracts/{id}/clauses/{clause_id}/customize",
                "risk_analysis": "/api/contracts/{id}/risk-analysis",
                "chatbot": "/api/contracts/{id}/chat"
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
