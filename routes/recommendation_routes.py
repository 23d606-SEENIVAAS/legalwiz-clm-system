# recommendation_routes.py - Phase 2: Smart Clause Recommendations (Graph RAG)
"""
AI-powered clause recommendations using Graph RAG.
The GRAPH decides what to recommend, the LLM explains why.

Endpoints:
- GET  /{contract_id}/recommendations — get AI recommendations
- POST /{contract_id}/recommendations/apply — apply a recommendation
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

from graph_rag_engine import retriever, llm_client, validator
from llm_config import SYSTEM_PROMPT_RECOMMENDATIONS, RECOMMENDATION_PROMPT
from config import DB_CONFIG

import psycopg2
from psycopg2.extras import RealDictCursor

router = APIRouter(prefix="/api/contracts", tags=["recommendations"])


# ==================== PYDANTIC MODELS ====================

class Recommendation(BaseModel):
    type: str  # "variant_upgrade" | "missing_clause" | "optional_addition"
    clause_type: str
    current_clause_id: Optional[str] = None
    recommended_clause_id: Optional[str] = None
    title: str
    reason: str
    benefit: str
    priority: str  # "high" | "medium" | "low"
    recommendation_strength: Optional[str] = None

class RecommendationResponse(BaseModel):
    contract_id: str
    recommendations: List[Recommendation]
    summary: str
    total_recommendations: int
    grounding_validation: Dict[str, Any]
    generated_at: datetime

class ApplyRecommendationRequest(BaseModel):
    recommendation_type: str  # "variant_upgrade" | "missing_clause" | "optional_addition"
    clause_type: str
    clause_id: str  # The clause_id to activate/add


# ==================== HELPERS ====================

def get_db():
    return psycopg2.connect(**DB_CONFIG)


def _format_alternatives(alternatives: List[Dict]) -> str:
    """Format alternatives data for LLM prompt."""
    if not alternatives:
        return "No alternative variants found."
    
    lines = []
    for alt in alternatives:
        lines.append(
            f"- Current: {alt['current_clause_id']} ({alt['current_variant']}, risk={alt['current_risk']})\n"
            f"  Recommended: {alt['recommended_clause_id']} ({alt['recommended_variant']}, risk={alt['recommended_risk']})\n"
            f"  Clause Type: {alt['clause_type']}\n"
            f"  Reason: {alt['reason']}\n"
            f"  Benefit: {alt['benefit']}\n"
            f"  Strength: {alt['strength']}\n"
        )
    return "\n".join(lines)


def _format_requires(requires: List[Dict]) -> str:
    """Format requires data for LLM prompt."""
    if not requires:
        return "No missing dependencies detected."
    
    lines = []
    for req in requires:
        lines.append(
            f"- {req['source_name']} REQUIRES {req['required_name']}\n"
            f"  Dependency Type: {req['dependency_type']}\n"
            f"  Is Critical: {req['is_critical']}\n"
            f"  Reason: {req['reason']}\n"
        )
    return "\n".join(lines)


def _format_optional_gaps(gaps: List[Dict]) -> str:
    """Format optional gaps for LLM prompt."""
    if not gaps:
        return "All optional clause types are covered."
    
    lines = []
    for gap in gaps:
        lines.append(
            f"- {gap['clause_type_name']} (category: {gap['category']})\n"
            f"  Importance: {gap['importance_level']}\n"
            f"  Description: {gap['description']}\n"
        )
    return "\n".join(lines)


# ==================== ROUTES ====================

@router.get("/{contract_id}/recommendations", response_model=RecommendationResponse)
async def get_recommendations(contract_id: str):
    """
    Get AI-powered clause recommendations for a contract.
    
    Uses Graph RAG:
    1. Retrieves ALTERNATIVE_TO relationships for variant upgrades
    2. Retrieves REQUIRES relationships for missing dependencies
    3. Identifies unselected optional clause types
    4. LLM explains and prioritizes the graph-derived recommendations
    5. Validator verifies all recommended clause_ids exist in Neo4j
    """
    # Get contract info
    contract = retriever.get_contract_info(contract_id)
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    # Get active clause IDs
    active_clauses = retriever.get_active_clause_ids(contract_id)
    if not active_clauses:
        raise HTTPException(
            status_code=400,
            detail="No active clauses found. Complete Step 3 first."
        )
    
    active_clause_ids = [c["clause_id"] for c in active_clauses]
    
    # --- GRAPH RETRIEVAL ---
    graph_context = retriever.get_recommendation_context(
        contract["contract_type"],
        contract["jurisdiction"],
        active_clause_ids
    )
    
    # If no recommendations from graph, return empty
    if (not graph_context["alternatives"] and 
        not graph_context["requires"] and 
        not graph_context["optional_gaps"]):
        return {
            "contract_id": contract_id,
            "recommendations": [],
            "summary": "Your contract configuration looks complete — no recommendations at this time.",
            "total_recommendations": 0,
            "grounding_validation": {"valid": True, "grounding_rate": 1.0},
            "generated_at": datetime.now()
        }
    
    # Check if LLM is configured
    if not llm_client.is_configured():
        # Return graph data directly without LLM interpretation
        raw_recs = []
        for alt in graph_context["alternatives"]:
            raw_recs.append({
                "type": "variant_upgrade",
                "clause_type": alt["clause_type"],
                "current_clause_id": alt["current_clause_id"],
                "recommended_clause_id": alt["recommended_clause_id"],
                "title": f"Upgrade {alt['clause_type']} to {alt['recommended_variant']}",
                "reason": alt["reason"],
                "benefit": alt["benefit"],
                "priority": "high" if alt["strength"] == "high" else "medium",
                "recommendation_strength": alt["strength"]
            })
        for req in graph_context["requires"]:
            raw_recs.append({
                "type": "missing_clause",
                "clause_type": req["required_clause_type"],
                "current_clause_id": None,
                "recommended_clause_id": None,
                "title": f"Missing required: {req['required_name']}",
                "reason": req["reason"],
                "benefit": f"Required by {req['source_name']} ({req['dependency_type']})",
                "priority": "high" if req["is_critical"] else "medium",
                "recommendation_strength": "high" if req["is_critical"] else "medium"
            })
        
        return {
            "contract_id": contract_id,
            "recommendations": raw_recs,
            "summary": "Recommendations from knowledge graph (LLM not configured for detailed analysis).",
            "total_recommendations": len(raw_recs),
            "grounding_validation": {"valid": True, "grounding_rate": 1.0},
            "generated_at": datetime.now()
        }
    
    # --- LLM GENERATION ---
    prompt = RECOMMENDATION_PROMPT.format(
        contract_type=contract["contract_type"],
        jurisdiction=contract["jurisdiction"],
        active_clauses=", ".join(
            f"{c['clause_type']}:{c['variant']}({c['clause_id']})" for c in active_clauses
        ),
        alternatives_data=_format_alternatives(graph_context["alternatives"]),
        requires_data=_format_requires(graph_context["requires"]),
        optional_gaps=_format_optional_gaps(graph_context["optional_gaps"])
    )
    
    try:
        llm_response = llm_client.generate(prompt, SYSTEM_PROMPT_RECOMMENDATIONS)
    except Exception as e:
        # Fall back to raw graph recommendations on LLM failure
        raw_recs = []
        for alt in graph_context["alternatives"]:
            raw_recs.append({
                "type": "variant_upgrade",
                "clause_type": alt["clause_type"],
                "current_clause_id": alt["current_clause_id"],
                "recommended_clause_id": alt["recommended_clause_id"],
                "title": f"Upgrade {alt['clause_type']} to {alt['recommended_variant']}",
                "reason": alt["reason"],
                "benefit": alt["benefit"],
                "priority": "high" if alt.get("strength") == "high" else "medium",
                "recommendation_strength": alt.get("strength")
            })
        for req in graph_context["requires"]:
            raw_recs.append({
                "type": "missing_clause",
                "clause_type": req["required_clause_type"],
                "current_clause_id": None,
                "recommended_clause_id": None,
                "title": f"Missing required: {req['required_name']}",
                "reason": req["reason"],
                "benefit": f"Required by {req['source_name']} ({req['dependency_type']})",
                "priority": "high" if req["is_critical"] else "medium",
                "recommendation_strength": "high" if req["is_critical"] else "medium"
            })
        
        return {
            "contract_id": contract_id,
            "recommendations": raw_recs,
            "summary": f"Recommendations from knowledge graph (LLM unavailable: {str(e)[:100]}).",
            "total_recommendations": len(raw_recs),
            "grounding_validation": {"valid": True, "grounding_rate": 1.0, "llm_fallback": True},
            "generated_at": datetime.now()
        }
    
    recommendations = llm_response.get("recommendations", [])
    
    # --- VALIDATION ---
    grounding = validator.validate_recommendations(recommendations, graph_context)
    
    # Filter out ungrounded recommendations (hallucinated)
    if not grounding["valid"]:
        recommendations = [
            r for r in recommendations 
            if r not in grounding.get("ungrounded_recommendations", [])
        ]
    
    return {
        "contract_id": contract_id,
        "recommendations": recommendations,
        "summary": llm_response.get("summary", ""),
        "total_recommendations": len(recommendations),
        "grounding_validation": {
            "valid": grounding["valid"],
            "grounding_rate": grounding["grounding_rate"],
            "filtered_hallucinations": grounding["ungrounded_count"]
        },
        "generated_at": datetime.now()
    }


@router.post("/{contract_id}/recommendations/apply")
async def apply_recommendation(contract_id: str, request: ApplyRecommendationRequest):
    """
    Apply a recommendation: switch variant or add a clause.
    """
    # Validate the clause_id exists in Neo4j
    if not retriever.verify_clause_exists(request.clause_id):
        raise HTTPException(
            status_code=404,
            detail=f"Clause '{request.clause_id}' not found in knowledge graph"
        )
    
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if request.recommendation_type == "variant_upgrade":
                # Deactivate current variant of this clause_type
                cur.execute("""
                    UPDATE contract_clauses
                    SET is_active = false, updated_at = NOW()
                    WHERE contract_id = %s AND clause_type = %s
                """, (contract_id, request.clause_type))
                
                # Activate the recommended variant
                cur.execute("""
                    UPDATE contract_clauses
                    SET is_active = true, updated_at = NOW()
                    WHERE contract_id = %s AND clause_id = %s
                    RETURNING *
                """, (contract_id, request.clause_id))
                
                result = cur.fetchone()
                if not result:
                    raise HTTPException(
                        status_code=404,
                        detail="Clause not found in contract. Generate clauses first."
                    )
                
                conn.commit()
                return {"message": "Variant switched successfully", "clause": dict(result)}
            
            elif request.recommendation_type in ("missing_clause", "optional_addition"):
                # Check if clause already exists in contract
                cur.execute("""
                    SELECT id FROM contract_clauses
                    WHERE contract_id = %s AND clause_id = %s
                """, (contract_id, request.clause_id))
                
                if cur.fetchone():
                    # Just activate it
                    cur.execute("""
                        UPDATE contract_clauses
                        SET is_active = true, updated_at = NOW()
                        WHERE contract_id = %s AND clause_id = %s
                        RETURNING *
                    """, (contract_id, request.clause_id))
                    result = cur.fetchone()
                else:
                    # Get next sequence number
                    cur.execute("""
                        SELECT COALESCE(MAX(sequence), 0) + 1 AS next_seq
                        FROM contract_clauses WHERE contract_id = %s
                    """, (contract_id,))
                    next_seq = cur.fetchone()["next_seq"]
                    
                    # Insert the clause
                    cur.execute("""
                        INSERT INTO contract_clauses (
                            contract_id, clause_id, clause_type, variant,
                            sequence, is_mandatory, is_customized, is_active
                        ) VALUES (%s, %s, %s, %s, %s, false, false, true)
                        RETURNING *
                    """, (
                        contract_id, request.clause_id,
                        request.clause_type,
                        "Moderate",  # Default variant
                        next_seq
                    ))
                    result = cur.fetchone()
                
                conn.commit()
                return {"message": "Clause added successfully", "clause": dict(result)}
            
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown recommendation type: {request.recommendation_type}"
                )
    
            # Auto-version: snapshot after recommendation applied
            try:
                from version_routes import create_version_snapshot
                create_version_snapshot(
                    contract_id,
                    f"Applied recommendation: {request.recommendation_type} for {request.clause_type}"
                )
            except Exception:
                pass  # Don't fail the main operation if versioning fails

    finally:
        conn.close()
