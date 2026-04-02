# risk_routes.py - Phase 4: Risk Analysis (Graph RAG)
"""
AI-powered contract risk analysis using Graph RAG.
All risk data comes from the graph — the LLM explains, never invents.

Endpoints:
- GET  /{contract_id}/risk-analysis — full risk analysis dashboard
- GET  /{contract_id}/risk-analysis/quick — lightweight risk summary (no LLM)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

from graph_rag_engine import retriever, llm_client, validator
from llm_config import SYSTEM_PROMPT_RISK, RISK_ANALYSIS_PROMPT
from config import DB_CONFIG

import psycopg2
from psycopg2.extras import RealDictCursor

router = APIRouter(prefix="/api/contracts", tags=["risk-analysis"])


# ==================== PYDANTIC MODELS ====================

class ClauseRisk(BaseModel):
    clause_id: str
    clause_type: str
    risk_level: float
    explanation: Optional[str] = None
    mitigation: Optional[str] = None

class ConflictItem(BaseModel):
    clause_a: str
    clause_b: str
    severity: str
    description: str
    resolution: str

class GapItem(BaseModel):
    missing_clause_type: str
    reason: str
    impact: Optional[str] = None
    is_critical: bool

class RiskAnalysisResponse(BaseModel):
    contract_id: str
    overall_risk_score: float
    overall_risk_label: str  # "Low" | "Medium" | "High" | "Critical"
    summary: str
    clause_risks: List[ClauseRisk]
    conflicts: List[ConflictItem]
    gaps: List[GapItem]
    action_items: List[str]
    validation: Dict[str, Any]
    generated_at: datetime

class QuickRiskResponse(BaseModel):
    contract_id: str
    overall_risk_score: float
    overall_risk_label: str
    total_clauses: int
    conflict_count: int
    gap_count: int
    high_risk_clauses: List[Dict[str, Any]]
    generated_at: datetime


# ==================== HELPERS ====================

def _compute_risk_score(clause_risks: List[Dict], conflicts: List[Dict]) -> float:
    """
    Compute overall risk score from graph data. Pure math, no LLM.
    
    Formula:
    - Base = weighted average of clause risk levels (by importance)
    - Conflict penalty = +0.5 per high severity, +0.3 per medium, +0.1 per low
    - Capped at 10.0
    """
    if not clause_risks:
        return 0.0
    
    # Importance weights
    importance_weights = {"Critical": 3.0, "High": 2.0, "Medium": 1.5, "Low": 1.0}
    
    total_weight = 0
    weighted_sum = 0
    for cr in clause_risks:
        importance = cr.get("importance_level", "Medium")
        weight = importance_weights.get(importance, 1.0)
        weighted_sum += cr.get("risk_level", 5.0) * weight
        total_weight += weight
    
    base_score = weighted_sum / total_weight if total_weight > 0 else 5.0
    
    # Conflict penalty
    severity_penalty = {"high": 0.5, "medium": 0.3, "low": 0.1}
    conflict_penalty = sum(
        severity_penalty.get(c.get("severity", "low"), 0.1) 
        for c in conflicts
    )
    
    return min(10.0, round(base_score + conflict_penalty, 1))


def _get_risk_label(score: float) -> str:
    """Map numeric risk score to label."""
    if score <= 3.5:
        return "Low"
    elif score <= 5.5:
        return "Medium"
    elif score <= 7.5:
        return "High"
    else:
        return "Critical"


def _format_clause_risks(risks: List[Dict]) -> str:
    """Format clause risk data for LLM prompt."""
    lines = []
    for r in risks:
        lines.append(
            f"- {r['clause_id']} ({r.get('clause_type_name', r.get('clause_type', 'Unknown'))}): "
            f"variant={r['variant']}, risk_level={r['risk_level']}/10, "
            f"importance={r.get('importance_level', 'Medium')}, "
            f"category={r.get('category', 'General')}"
        )
    return "\n".join(lines) or "No clause risks found."


def _format_conflicts(conflicts: List[Dict]) -> str:
    """Format conflict data for LLM prompt."""
    if not conflicts:
        return "No conflicts detected between active clauses."
    
    lines = []
    for c in conflicts:
        lines.append(
            f"- CONFLICT: {c['clause_a_id']} ({c['clause_a_type']}/{c['clause_a_variant']}) "
            f"↔ {c['clause_b_id']} ({c['clause_b_type']}/{c['clause_b_variant']})\n"
            f"  Severity: {c['severity']}\n"
            f"  Type: {c['conflict_type']}\n"
            f"  Reason: {c['reason']}\n"
            f"  Resolution Advice: {c['resolution_advice']}"
        )
    return "\n".join(lines)


def _format_missing_deps(deps: List[Dict]) -> str:
    """Format missing dependency data for LLM prompt."""
    if not deps:
        return "No missing dependencies detected."
    
    lines = []
    for d in deps:
        critical = "CRITICAL" if d["is_critical"] else "recommended"
        lines.append(
            f"- {d['source_name']} REQUIRES {d['missing_name']} ({critical})\n"
            f"  Dependency Type: {d['dependency_type']}\n"
            f"  Reason: {d['reason']}"
        )
    return "\n".join(lines)


def _format_gaps(gaps: List[Dict]) -> str:
    """Format gap data for LLM prompt."""
    if not gaps:
        return "No clause type gaps in contract template."
    
    lines = []
    for g in gaps:
        lines.append(
            f"- Missing: {g['clause_type_name']} "
            f"(importance: {g['importance_level']}, {g.get('description', '')})"
        )
    return "\n".join(lines)


# ==================== ROUTES ====================

@router.get("/{contract_id}/risk-analysis/quick", response_model=QuickRiskResponse)
async def get_quick_risk(contract_id: str):
    """
    Lightweight risk summary — no LLM needed.
    Pure graph computation for quick dashboard display.
    """
    contract = retriever.get_contract_info(contract_id)
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    active_clauses = retriever.get_active_clause_ids(contract_id)
    if not active_clauses:
        raise HTTPException(status_code=400, detail="No active clauses found")
    
    active_clause_ids = [c["clause_id"] for c in active_clauses]
    
    # Get risk context from graph
    risk_ctx = retriever.get_risk_context(
        contract["contract_type"],
        contract["jurisdiction"],
        active_clause_ids
    )
    
    # Compute risk score (pure math)
    risk_score = _compute_risk_score(
        risk_ctx["clause_risks"], risk_ctx["conflicts"]
    )
    risk_label = _get_risk_label(risk_score)
    
    # High risk clauses
    high_risk = [
        {"clause_id": r["clause_id"], "clause_type": r.get("clause_type", ""), 
         "risk_level": r["risk_level"]}
        for r in risk_ctx["clause_risks"]
        if r["risk_level"] >= 7
    ]
    
    return {
        "contract_id": contract_id,
        "overall_risk_score": risk_score,
        "overall_risk_label": risk_label,
        "total_clauses": len(active_clause_ids),
        "conflict_count": len(risk_ctx["conflicts"]),
        "gap_count": len(risk_ctx["gaps"]) + len(risk_ctx["missing_dependencies"]),
        "high_risk_clauses": high_risk,
        "generated_at": datetime.now()
    }


@router.get("/{contract_id}/risk-analysis", response_model=RiskAnalysisResponse)
async def get_risk_analysis(contract_id: str):
    """
    Full AI-powered risk analysis dashboard.
    
    Graph RAG flow:
    1. Retrieve risk levels, conflicts (CONFLICTS_WITH), dependencies (REQUIRES), and gaps
    2. Compute overall risk score from graph data (pure math)
    3. LLM explains risks in plain language and suggests mitigations
    4. Validator verifies risk scores match graph data
    """
    contract = retriever.get_contract_info(contract_id)
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    active_clauses = retriever.get_active_clause_ids(contract_id)
    if not active_clauses:
        raise HTTPException(status_code=400, detail="No active clauses found")
    
    active_clause_ids = [c["clause_id"] for c in active_clauses]
    
    # --- GRAPH RETRIEVAL ---
    risk_ctx = retriever.get_risk_context(
        contract["contract_type"],
        contract["jurisdiction"],
        active_clause_ids
    )
    
    # Compute risk score (pure math, no LLM)
    risk_score = _compute_risk_score(
        risk_ctx["clause_risks"], risk_ctx["conflicts"]
    )
    risk_label = _get_risk_label(risk_score)
    
    # If LLM is not configured, return raw graph analysis
    if not llm_client.is_configured():
        return _build_raw_risk_response(
            contract_id, risk_score, risk_label, risk_ctx
        )
    
    # --- LLM GENERATION ---
    ct_info = risk_ctx.get("contract_type_info", {})
    prompt = RISK_ANALYSIS_PROMPT.format(
        contract_type=contract["contract_type"],
        contract_description=ct_info.get("description", ""),
        jurisdiction=contract["jurisdiction"],
        total_clauses=len(active_clause_ids),
        clause_risks=_format_clause_risks(risk_ctx["clause_risks"]),
        conflicts=_format_conflicts(risk_ctx["conflicts"]),
        missing_dependencies=_format_missing_deps(risk_ctx["missing_dependencies"]),
        gaps=_format_gaps(risk_ctx["gaps"])
    )
    
    try:
        llm_response = llm_client.generate(prompt, SYSTEM_PROMPT_RISK)
    except Exception as e:
        # Fall back to raw graph response on LLM failure
        return _build_raw_risk_response(
            contract_id, risk_score, risk_label, risk_ctx
        )
    
    # --- VALIDATION ---
    # Override LLM risk scores with graph-truth scores
    llm_clause_risks = llm_response.get("clause_risks", [])
    risk_validation = validator.validate_risk_scores(
        llm_clause_risks, risk_ctx["clause_risks"]
    )
    
    # Force-correct any mismatched risk scores to graph values
    graph_risk_map = {r["clause_id"]: r["risk_level"] for r in risk_ctx["clause_risks"]}
    for llm_risk in llm_clause_risks:
        cid = llm_risk.get("clause_id")
        if cid in graph_risk_map:
            llm_risk["risk_level"] = graph_risk_map[cid]
    
    # Override overall score with our math-computed one (not LLM's)
    return {
        "contract_id": contract_id,
        "overall_risk_score": risk_score,  # Graph-computed, NOT LLM
        "overall_risk_label": risk_label,
        "summary": llm_response.get("summary", ""),
        "clause_risks": llm_clause_risks,
        "conflicts": llm_response.get("conflicts", []),
        "gaps": llm_response.get("gaps", []),
        "action_items": llm_response.get("action_items", []),
        "validation": {
            "risk_scores_from_graph": True,
            "overall_score_formula": "weighted average + conflict penalty",
            "risk_score_accuracy": risk_validation["accuracy_rate"],
            "scores_corrected": len(risk_validation["mismatches"])
        },
        "generated_at": datetime.now()
    }


def _build_raw_risk_response(
    contract_id: str, risk_score: float, risk_label: str, risk_ctx: Dict
) -> Dict:
    """Build risk response from raw graph data (no LLM)."""
    clause_risks = []
    for r in risk_ctx["clause_risks"]:
        clause_risks.append({
            "clause_id": r["clause_id"],
            "clause_type": r.get("clause_type", ""),
            "risk_level": r["risk_level"],
            "explanation": f"Risk level {r['risk_level']}/10 ({r['variant']} variant)",
            "mitigation": "Consider switching to a lower-risk variant" if r["risk_level"] >= 7 else None
        })
    
    conflicts = []
    for c in risk_ctx["conflicts"]:
        conflicts.append({
            "clause_a": c["clause_a_id"],
            "clause_b": c["clause_b_id"],
            "severity": c["severity"],
            "description": c["reason"],
            "resolution": c["resolution_advice"]
        })
    
    gaps = []
    for g in risk_ctx["missing_dependencies"]:
        gaps.append({
            "missing_clause_type": g["missing_type"],
            "reason": g["reason"],
            "impact": f"Required by {g['source_name']}",
            "is_critical": g["is_critical"]
        })
    for g in risk_ctx["gaps"]:
        gaps.append({
            "missing_clause_type": g["clause_type_id"],
            "reason": f"Mandatory clause type for this contract template",
            "impact": g.get("description", ""),
            "is_critical": True
        })
    
    action_items = []
    if conflicts:
        action_items.append(f"Resolve {len(conflicts)} clause conflict(s)")
    if gaps:
        critical_gaps = [g for g in gaps if g["is_critical"]]
        if critical_gaps:
            action_items.append(f"Add {len(critical_gaps)} missing critical clause type(s)")
    high_risk = [r for r in clause_risks if r["risk_level"] >= 7]
    if high_risk:
        action_items.append(f"Review {len(high_risk)} high-risk clause(s)")
    
    return {
        "contract_id": contract_id,
        "overall_risk_score": risk_score,
        "overall_risk_label": risk_label,
        "summary": f"Contract risk score: {risk_score}/10 ({risk_label}). "
                   f"{len(conflicts)} conflicts, {len(gaps)} gaps detected.",
        "clause_risks": clause_risks,
        "conflicts": conflicts,
        "gaps": gaps,
        "action_items": action_items,
        "validation": {
            "risk_scores_from_graph": True,
            "llm_used": False,
            "note": "LLM not configured — showing raw graph analysis"
        },
        "generated_at": datetime.now()
    }
