# customization_routes.py - Phase 3: Smart Clause Customization (Graph RAG)
"""
AI-powered clause customization with placeholder preservation and risk impact analysis.

Endpoints:
- POST /{contract_id}/clauses/{clause_db_id}/customize — get AI customization suggestion
- POST /{contract_id}/clauses/{clause_db_id}/apply-customization — save customization
- POST /{contract_id}/clauses/{clause_db_id}/revert-customization — revert to original
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

from graph_rag_engine import retriever, llm_client, validator
from llm_config import SYSTEM_PROMPT_CUSTOMIZATION, CUSTOMIZATION_PROMPT
from config import DB_CONFIG

import psycopg2
from psycopg2.extras import RealDictCursor

router = APIRouter(prefix="/api/contracts", tags=["customization"])


# ==================== PYDANTIC MODELS ====================

class CustomizeRequest(BaseModel):
    instruction: str  # User's natural language customization request

class CustomizationResult(BaseModel):
    clause_id: str
    clause_type: str
    original_text: str
    customized_text: str
    changes_summary: str
    risk_impact: str  # "lower" | "same" | "higher"
    risk_explanation: str
    preserved_placeholders: List[str]
    legal_notes: Optional[str] = None
    validation: Dict[str, Any]

class ApplyCustomizationRequest(BaseModel):
    customized_text: str

class RevertResponse(BaseModel):
    clause_id: str
    original_text: str
    message: str


# ==================== HELPERS ====================

def get_db():
    return psycopg2.connect(**DB_CONFIG)


def _format_variants(variants: List[Dict]) -> str:
    """Format variant data for LLM prompt."""
    lines = []
    for v in variants:
        text_preview = v["raw_text"][:300] + "..." if len(v["raw_text"]) > 300 else v["raw_text"]
        lines.append(
            f"### {v['variant']} Variant (risk: {v['risk_level']})\n"
            f"ID: {v['clause_id']}\n"
            f"Text:\n{text_preview}\n"
        )
    return "\n".join(lines)


def _format_parameters(params: List[Dict]) -> str:
    """Format parameter data for LLM prompt."""
    if not params:
        return "No parameters in this clause."
    
    lines = []
    for p in params:
        req = "REQUIRED" if p["is_required"] else "optional"
        lines.append(f"- {p['parameter_name']} (type: {p['data_type']}, {req})")
    return "\n".join(lines)


# ==================== ROUTES ====================

@router.post("/{contract_id}/clauses/{clause_db_id}/customize", response_model=CustomizationResult)
async def customize_clause(
    contract_id: str, clause_db_id: int, request: CustomizeRequest
):
    """
    Get an AI-powered customization suggestion for a clause.
    
    Graph RAG flow:
    1. Get clause text + all variants + parameters from Neo4j
    2. LLM customizes based on user instruction + graph context
    3. Validator checks all {{PLACEHOLDERS}} are preserved
    4. Returns diff-ready result with risk impact
    """
    # Get clause metadata from Supabase
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM contract_clauses
                WHERE id = %s AND contract_id = %s
            """, (clause_db_id, contract_id))
            clause_record = cur.fetchone()
            
            if not clause_record:
                raise HTTPException(status_code=404, detail="Clause not found in contract")
    finally:
        conn.close()
    
    clause_id = clause_record["clause_id"]
    
    # --- GRAPH RETRIEVAL ---
    graph_context = retriever.get_customization_context(clause_id)
    if not graph_context:
        raise HTTPException(
            status_code=404,
            detail=f"Clause '{clause_id}' not found in knowledge graph"
        )
    
    clause_data = graph_context["clause"]
    original_text = clause_record.get("overridden_text") or clause_data["raw_text"]
    
    # Check if LLM is configured
    if not llm_client.is_configured():
        raise HTTPException(
            status_code=503,
            detail="LLM not configured. Set LLM_API_KEY in .env to enable AI customization."
        )
    
    # --- LLM GENERATION ---
    prompt = CUSTOMIZATION_PROMPT.format(
        clause_id=clause_id,
        clause_type=clause_data["clause_type"],
        variant=clause_data["variant"],
        risk_level=clause_data["risk_level"],
        clause_text=original_text,
        all_variants=_format_variants(graph_context["all_variants"]),
        parameters=_format_parameters(graph_context["parameters"]),
        user_instruction=request.instruction
    )
    
    try:
        llm_response = llm_client.generate(prompt, SYSTEM_PROMPT_CUSTOMIZATION)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM generation failed: {str(e)}"
        )
    
    customized_text = llm_response.get("customized_text", "")
    
    # --- VALIDATION ---
    placeholder_check = validator.validate_placeholders(original_text, customized_text)
    
    # If placeholders are missing, reject the customization
    if not placeholder_check["valid"]:
        # Try to repair: re-add missing placeholders as a note
        missing = placeholder_check["missing_placeholders"]
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Customization failed validation: missing placeholders",
                "missing_placeholders": missing,
                "hint": "The AI removed required placeholders. Try being more specific in your instruction."
            }
        )
    
    return {
        "clause_id": clause_id,
        "clause_type": clause_data["clause_type"],
        "original_text": original_text,
        "customized_text": customized_text,
        "changes_summary": llm_response.get("changes_summary", ""),
        "risk_impact": llm_response.get("risk_impact", "same"),
        "risk_explanation": llm_response.get("risk_explanation", ""),
        "preserved_placeholders": placeholder_check["preserved_placeholders"],
        "legal_notes": llm_response.get("legal_notes"),
        "validation": {
            "placeholders_valid": placeholder_check["valid"],
            "preservation_rate": placeholder_check["preservation_rate"],
            "all_placeholders": placeholder_check["original_placeholders"]
        }
    }


@router.post("/{contract_id}/clauses/{clause_db_id}/apply-customization")
async def apply_customization(
    contract_id: str, clause_db_id: int, request: ApplyCustomizationRequest
):
    """
    Apply a customization by saving the customized text to overridden_text.
    The user reviews the AI suggestion first, then applies it.
    """
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                UPDATE contract_clauses
                SET overridden_text = %s, is_customized = true, updated_at = NOW()
                WHERE id = %s AND contract_id = %s
                RETURNING *
            """, (request.customized_text, clause_db_id, contract_id))
            
            result = cur.fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="Clause not found")
            
            conn.commit()
            
            # Auto-version: snapshot after customization
            try:
                from version_routes import create_version_snapshot
                create_version_snapshot(
                    contract_id,
                    f"Customized clause {result.get('clause_type', clause_db_id)}"
                )
            except Exception:
                pass  # Don't fail the main operation if versioning fails
            
            return {
                "message": "Customization applied successfully",
                "clause": dict(result)
            }
    finally:
        conn.close()


@router.post("/{contract_id}/clauses/{clause_db_id}/revert-customization", response_model=RevertResponse)
async def revert_customization(contract_id: str, clause_db_id: int):
    """
    Revert a customized clause back to its original Neo4j text.
    Clears overridden_text and resets is_customized.
    """
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get current clause to find its Neo4j ID
            cur.execute("""
                SELECT clause_id FROM contract_clauses
                WHERE id = %s AND contract_id = %s
            """, (clause_db_id, contract_id))
            clause_record = cur.fetchone()
            
            if not clause_record:
                raise HTTPException(status_code=404, detail="Clause not found")
            
            # Clear customization
            cur.execute("""
                UPDATE contract_clauses
                SET overridden_text = NULL, is_customized = false, updated_at = NOW()
                WHERE id = %s AND contract_id = %s
            """, (clause_db_id, contract_id))
            
            conn.commit()
            
            # Get original text from Neo4j
            context = retriever.get_customization_context(clause_record["clause_id"])
            original_text = context["clause"]["raw_text"] if context else "Original text unavailable"
            
            return {
                "clause_id": clause_record["clause_id"],
                "original_text": original_text,
                "message": "Customization reverted to original text"
            }
    finally:
        conn.close()
