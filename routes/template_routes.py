# template_routes.py - Contract Templates
"""
Save and reuse contract configurations as templates.
Templates store clause selections + parameter defaults.

Endpoints:
- POST /api/templates                              — Create template from contract
- GET  /api/templates                              — List templates (filter by type)
- GET  /api/templates/{id}                         — Get single template
- DELETE /api/templates/{id}                       — Delete template
- POST /api/contracts/{id}/apply-template/{tid}    — Apply template to contract
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from config import DB_CONFIG
import psycopg2
from psycopg2.extras import RealDictCursor

router = APIRouter(tags=["templates"])


# ==================== TABLE AUTO-CREATION ====================

_table_created = False

def _ensure_table():
    """Create contract_templates table if it doesn't exist."""
    global _table_created
    if _table_created:
        return

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS contract_templates (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    contract_type TEXT NOT NULL,
                    jurisdiction TEXT NOT NULL,
                    clause_config JSONB NOT NULL DEFAULT '[]',
                    parameter_defaults JSONB NOT NULL DEFAULT '{}',
                    created_by UUID,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    is_public BOOLEAN NOT NULL DEFAULT true
                )
            """)
            conn.commit()
        _table_created = True
    finally:
        conn.close()


# ==================== PYDANTIC MODELS ====================

class CreateTemplateRequest(BaseModel):
    contract_id: str  # Source contract to create template from
    name: str
    description: Optional[str] = None
    is_public: bool = True


class TemplateResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    contract_type: str
    jurisdiction: str
    clause_count: int
    parameter_count: int
    created_at: datetime
    is_public: bool


class TemplateDetailResponse(TemplateResponse):
    clause_config: List[Dict[str, Any]]
    parameter_defaults: Dict[str, Any]


# ==================== HELPERS ====================

def get_db():
    return psycopg2.connect(**DB_CONFIG)


def _snapshot_contract(contract_id: str) -> dict:
    """
    Capture the current state of a contract's clause selections
    and parameter values as a template snapshot.
    """
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get contract info
            cur.execute("""
                SELECT id, title, contract_type, jurisdiction
                FROM contracts WHERE id = %s
            """, (contract_id,))
            contract = cur.fetchone()
            if not contract:
                raise HTTPException(status_code=404, detail="Contract not found")

            # Get active clause selections
            cur.execute("""
                SELECT clause_id, clause_type, variant, sequence,
                       is_mandatory, is_active, is_customized, overridden_text
                FROM contract_clauses
                WHERE contract_id = %s
                ORDER BY sequence,
                    CASE variant
                        WHEN 'Standard' THEN 1
                        WHEN 'Moderate' THEN 2
                        WHEN 'Strict' THEN 3
                        ELSE 4
                    END
            """, (contract_id,))
            clauses = [dict(r) for r in cur.fetchall()]

            # Get parameter values
            cur.execute("""
                SELECT cp.parameter_id, cp.value_text, cp.value_integer,
                       cp.value_decimal, cp.value_date, cp.value_currency
                FROM contract_parameters cp
                WHERE cp.contract_id = %s
            """, (contract_id,))
            params = {}
            for row in cur.fetchall():
                pid = row["parameter_id"]
                # Store whichever value column is populated
                val = (row["value_text"] or row["value_integer"] or
                       row["value_decimal"] or
                       (str(row["value_date"]) if row["value_date"] else None) or
                       (json.dumps(row["value_currency"]) if row["value_currency"] else None))
                if val is not None:
                    params[pid] = val

        return {
            "contract_type": contract["contract_type"],
            "jurisdiction": contract["jurisdiction"],
            "clause_config": clauses,
            "parameter_defaults": params,
        }
    finally:
        conn.close()


# ==================== ROUTES ====================

@router.post("/api/templates", response_model=TemplateResponse)
async def create_template(request: CreateTemplateRequest):
    """
    Create a reusable template from an existing contract's configuration.
    Captures clause selections (which variants are active) and parameter values.
    """
    _ensure_table()

    snapshot = _snapshot_contract(request.contract_id)

    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO contract_templates
                    (name, description, contract_type, jurisdiction,
                     clause_config, parameter_defaults, is_public)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING *
            """, (
                request.name,
                request.description,
                snapshot["contract_type"],
                snapshot["jurisdiction"],
                json.dumps(snapshot["clause_config"], default=str),
                json.dumps(snapshot["parameter_defaults"], default=str),
                request.is_public,
            ))
            template = cur.fetchone()
            conn.commit()

        return {
            **dict(template),
            "clause_count": len(snapshot["clause_config"]),
            "parameter_count": len(snapshot["parameter_defaults"]),
        }
    finally:
        conn.close()


@router.get("/api/templates", response_model=List[TemplateResponse])
async def list_templates(
    contract_type: Optional[str] = Query(None, description="Filter by contract type"),
    jurisdiction: Optional[str] = Query(None, description="Filter by jurisdiction"),
):
    """List available contract templates, optionally filtered."""
    _ensure_table()

    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = "SELECT * FROM contract_templates WHERE 1=1"
            params = []

            if contract_type:
                query += " AND contract_type = %s"
                params.append(contract_type)
            if jurisdiction:
                query += " AND jurisdiction = %s"
                params.append(jurisdiction)

            query += " ORDER BY created_at DESC"
            cur.execute(query, params)
            templates = cur.fetchall()

        return [
            {
                **dict(t),
                "clause_count": len(t.get("clause_config") or []),
                "parameter_count": len(t.get("parameter_defaults") or {}),
            }
            for t in templates
        ]
    finally:
        conn.close()


@router.get("/api/templates/{template_id}", response_model=TemplateDetailResponse)
async def get_template(template_id: int):
    """Get a single template with full clause and parameter details."""
    _ensure_table()

    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM contract_templates WHERE id = %s", (template_id,))
            template = cur.fetchone()
            if not template:
                raise HTTPException(status_code=404, detail="Template not found")

        return {
            **dict(template),
            "clause_count": len(template.get("clause_config") or []),
            "parameter_count": len(template.get("parameter_defaults") or {}),
        }
    finally:
        conn.close()


@router.delete("/api/templates/{template_id}")
async def delete_template(template_id: int):
    """Delete a template."""
    _ensure_table()

    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM contract_templates WHERE id = %s RETURNING id", (template_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Template not found")
            conn.commit()
        return {"message": "Template deleted", "template_id": template_id}
    finally:
        conn.close()


@router.post("/api/contracts/{contract_id}/apply-template/{template_id}")
async def apply_template(contract_id: str, template_id: int):
    """
    Apply a template to an existing contract.

    This will:
    1. Set active clauses to match the template's clause selections
    2. Optionally pre-fill parameter values from the template defaults

    Only affects clause active/inactive state — does not delete clauses.
    """
    _ensure_table()

    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get template
            cur.execute("SELECT * FROM contract_templates WHERE id = %s", (template_id,))
            template = cur.fetchone()
            if not template:
                raise HTTPException(status_code=404, detail="Template not found")

            # Get contract
            cur.execute("SELECT id, contract_type FROM contracts WHERE id = %s", (contract_id,))
            contract = cur.fetchone()
            if not contract:
                raise HTTPException(status_code=404, detail="Contract not found")

            # Get the template's active clause IDs
            clause_config = template["clause_config"] or []
            active_clause_ids = [
                c["clause_id"] for c in clause_config
                if c.get("is_active", False)
            ]

            if not active_clause_ids:
                raise HTTPException(
                    status_code=400,
                    detail="Template has no active clauses to apply"
                )

            # Step 1: Deactivate all clauses
            cur.execute("""
                UPDATE contract_clauses
                SET is_active = false, updated_at = NOW()
                WHERE contract_id = %s
            """, (contract_id,))
            deactivated = cur.rowcount

            # Step 2: Activate the template's selected clauses
            cur.execute("""
                UPDATE contract_clauses
                SET is_active = true, updated_at = NOW()
                WHERE contract_id = %s AND clause_id = ANY(%s)
            """, (contract_id, active_clause_ids))
            activated = cur.rowcount

            # Step 3: Apply parameter defaults (if any matching params exist)
            param_defaults = template.get("parameter_defaults") or {}
            params_applied = 0
            for param_id, value in param_defaults.items():
                cur.execute("""
                    UPDATE contract_parameters
                    SET value_text = %s, updated_at = NOW()
                    WHERE contract_id = %s AND parameter_id = %s
                    AND (value_text IS NULL OR value_text = '')
                """, (str(value), contract_id, param_id))
                params_applied += cur.rowcount

            conn.commit()

        return {
            "message": "Template applied successfully",
            "contract_id": contract_id,
            "template_id": template_id,
            "template_name": template["name"],
            "clauses_deactivated": deactivated,
            "clauses_activated": activated,
            "parameters_applied": params_applied,
        }
    finally:
        conn.close()
