# contract_generation_routes.py - Step 5: Contract Generation
import html as html_mod
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from neo4j import GraphDatabase
import psycopg2
from psycopg2.extras import RealDictCursor
import re
from config import DB_CONFIG, NEO4J_CONFIG, get_db, verify_contract_ownership
from auth_middleware import get_current_user

router = APIRouter(prefix="/api/contracts", tags=["contract-generation"])

# ==================== PYDANTIC MODELS ====================

class ClauseWithText(BaseModel):
    """Clause with rendered text"""
    id: int
    clause_id: str
    clause_type: str
    variant: str
    sequence: int
    raw_text: str
    rendered_text: str
    missing_parameters: List[str] = []

class GeneratedContract(BaseModel):
    """Complete generated contract"""
    contract_id: str
    contract_title: str
    contract_type: str
    jurisdiction: str
    generated_at: datetime
    clauses: List[ClauseWithText]
    full_text: str
    word_count: int
    is_complete: bool
    missing_parameters: List[str] = []

# ==================== DATABASE HELPERS ====================
from config import get_connection, get_neo4j_driver

# ==================== CONTRACT GENERATION ====================

def get_active_clauses_with_text(contract_id: str) -> List[Dict]:
    """
    Get active clauses from Supabase + raw text from Neo4j
    """
    # Get active clauses from Supabase
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    id,
                    clause_id,
                    clause_type,
                    variant,
                    sequence,
                    overridden_text
                FROM contract_clauses
                WHERE contract_id = %s AND is_active = true
                ORDER BY sequence
            """, (contract_id,))
            
            clauses = cur.fetchall()
    finally:
        conn.close()
    
    if not clauses:
        return []
    
    # Get raw text from Neo4j
    clause_ids = [c["clause_id"] for c in clauses]
    driver = get_neo4j_driver()
    
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (c:Clause)
                WHERE c.id IN $clause_ids
                RETURN c.id AS clause_id, c.raw_text AS raw_text
            """, {"clause_ids": clause_ids})
            
            neo4j_texts = {rec["clause_id"]: rec["raw_text"] for rec in result}
    finally:
        driver.close()
    
    # Merge data
    for clause in clauses:
        clause["raw_text"] = clause.get("overridden_text") or neo4j_texts.get(clause["clause_id"], "")
    
    return clauses


def get_parameter_values(contract_id: str) -> Dict[str, str]:
    """
    Get all parameter values for a contract
    Returns dict: {parameter_id: value}
    """
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    parameter_id,
                    value_text,
                    value_integer,
                    value_decimal,
                    value_date,
                    value_currency
                FROM contract_parameters
                WHERE contract_id = %s
            """, (contract_id,))
            
            rows = cur.fetchall()
            
            # Build parameter map
            param_values = {}
            for row in rows:
                param_id = row["parameter_id"]
                
                # Get value from appropriate column
                if row["value_text"] is not None:
                    param_values[param_id] = row["value_text"]
                elif row["value_integer"] is not None:
                    param_values[param_id] = str(row["value_integer"])
                elif row["value_decimal"] is not None:
                    param_values[param_id] = str(row["value_decimal"])
                elif row["value_date"] is not None:
                    param_values[param_id] = row["value_date"].strftime("%B %d, %Y")
                elif row["value_currency"] is not None:
                    currency_obj = row["value_currency"]
                    param_values[param_id] = f"{currency_obj.get('currency', 'INR')} {currency_obj.get('amount', 0):,}"
            
            return param_values
    finally:
        conn.close()


def get_parameter_names_map(contract_id: str) -> Dict[str, str]:
    """
    Get mapping of parameter_id to parameter_name from Neo4j
    Returns: {"P_156": "{{PARTY_A_NAME}}", ...}
    """
    # Get active clauses first
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT DISTINCT clause_id
                FROM contract_clauses
                WHERE contract_id = %s AND is_active = true
            """, (contract_id,))
            clause_ids = [row["clause_id"] for row in cur.fetchall()]
    finally:
        conn.close()
    
    if not clause_ids:
        return {}
    
    # Get parameter names from Neo4j
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (c:Clause)-[:CONTAINS_PARAM]->(p:Parameter)
                WHERE c.id IN $clause_ids
                RETURN DISTINCT p.id AS parameter_id, p.name AS parameter_name
            """, {"clause_ids": clause_ids})
            
            return {rec["parameter_id"]: rec["parameter_name"] for rec in result}
    finally:
        driver.close()


def replace_parameters(text: str, param_values: Dict[str, str], param_names: Dict[str, str]) -> tuple:
    """
    Replace parameter placeholders with actual values
    
    Args:
        text: Template text with {{PLACEHOLDERS}}
        param_values: {parameter_id: value}
        param_names: {parameter_id: parameter_name}
    
    Returns:
        (rendered_text, missing_parameters)
    """
    rendered_text = text
    missing = []
    
    # Find all placeholders in format {{PARAM_NAME}}
    placeholders = re.findall(r'\{\{([A-Z_0-9]+)\}\}', text)
    
    for placeholder in set(placeholders):
        placeholder_full = f"{{{{{placeholder}}}}}"
        
        # Find parameter_id that has this placeholder name
        param_id = None
        for pid, pname in param_names.items():
            if pname == placeholder_full:
                param_id = pid
                break
        
        # Replace if value exists
        if param_id and param_id in param_values:
            rendered_text = rendered_text.replace(placeholder_full, param_values[param_id])
        else:
            missing.append(placeholder_full)
    
    return rendered_text, missing


def format_contract_text(clauses: List[Dict], contract_title: str, contract_type: str) -> str:
    """
    Format clauses into a complete contract document
    """
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append(contract_title.upper().center(80))
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Contract Type: {contract_type}")
    lines.append(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("")
    
    # Clauses
    for i, clause in enumerate(clauses, 1):
        lines.append(f"{i}. {clause['clause_type'].upper().replace('_', ' ')}")
        lines.append("")
        lines.append(clause['rendered_text'])
        lines.append("")
        lines.append("-" * 80)
        lines.append("")
    
    # Footer
    lines.append("=" * 80)
    lines.append("END OF CONTRACT")
    lines.append("=" * 80)
    
    return "\n".join(lines)


# ==================== ROUTES ====================

@router.post("/{contract_id}/generate", response_model=GeneratedContract)
async def generate_contract(contract_id: str, user=Depends(get_current_user)):
    """
    Step 5.1: Generate complete contract with all parameters replaced
    
    Process:
    1. Get active clauses
    2. Fetch clause templates from Neo4j
    3. Get parameter values from Supabase
    4. Replace placeholders
    5. Format as complete document
    """
    verify_contract_ownership(contract_id, user["id"])

    # Verify contract exists
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, title, contract_type, jurisdiction
                FROM contracts
                WHERE id = %s
            """, (contract_id,))
            
            contract = cur.fetchone()
            
            if not contract:
                raise HTTPException(status_code=404, detail="Contract not found")
    finally:
        conn.close()
    
    # Get active clauses with text
    clauses = get_active_clauses_with_text(contract_id)
    
    if not clauses:
        raise HTTPException(
            status_code=400,
            detail="No active clauses found. Please complete Step 3 first."
        )
    
    # Get parameter values and names
    param_values = get_parameter_values(contract_id)
    param_names = get_parameter_names_map(contract_id)
    
    # Process each clause
    processed_clauses = []
    all_missing_params = set()
    
    for clause in clauses:
        rendered_text, missing = replace_parameters(
            clause["raw_text"],
            param_values,
            param_names
        )
        
        processed_clauses.append({
            "id": clause["id"],
            "clause_id": clause["clause_id"],
            "clause_type": clause["clause_type"],
            "variant": clause["variant"],
            "sequence": clause["sequence"],
            "raw_text": clause["raw_text"],
            "rendered_text": rendered_text,
            "missing_parameters": missing
        })
        
        all_missing_params.update(missing)
    
    # Generate full document
    full_text = format_contract_text(
        processed_clauses,
        contract.get("title", "Contract Agreement"),
        contract.get("contract_type", "General Agreement")
    )
    
    # Word count
    word_count = len(full_text.split())
    
    return {
        "contract_id": contract_id,
        "contract_title": contract.get("title", "Contract Agreement"),
        "contract_type": contract.get("contract_type", "General Agreement"),
        "jurisdiction": contract.get("jurisdiction", "India"),
        "generated_at": datetime.now(),
        "clauses": processed_clauses,
        "full_text": full_text,
        "word_count": word_count,
        "is_complete": len(all_missing_params) == 0,
        "missing_parameters": sorted(list(all_missing_params))
    }


@router.get("/{contract_id}/preview", response_model=GeneratedContract)
async def preview_contract(contract_id: str, user=Depends(get_current_user)):
    """
    Step 5.2: Preview contract (same as generate but GET request)
    
    Shows current state with placeholders for missing parameters
    """
    return await generate_contract(contract_id, user)


@router.get("/{contract_id}/preview/html")
async def preview_contract_html(contract_id: str, user=Depends(get_current_user)):
    """
    Step 5.3: Get HTML preview of contract
    
    Returns HTML formatted contract for web display
    """
    generated = await generate_contract(contract_id, user)
    
    # Convert to HTML
    html_parts = []
    html_parts.append('<!DOCTYPE html>')
    html_parts.append('<html lang="en">')
    html_parts.append('<head>')
    html_parts.append('<meta charset="UTF-8">')
    html_parts.append(f'<title>{html_mod.escape(generated["contract_title"])}</title>')
    html_parts.append('<style>')
    html_parts.append('''
        body { font-family: 'Times New Roman', serif; max-width: 800px; margin: 40px auto; padding: 20px; line-height: 1.6; }
        .header { text-align: center; border-bottom: 2px solid #000; padding-bottom: 20px; margin-bottom: 30px; }
        .title { font-size: 24px; font-weight: bold; margin-bottom: 10px; }
        .meta { font-size: 14px; color: #666; }
        .clause { margin-bottom: 30px; page-break-inside: avoid; }
        .clause-title { font-weight: bold; font-size: 16px; margin-bottom: 10px; }
        .clause-text { text-align: justify; }
        .missing-param { background-color: yellow; padding: 2px 4px; }
        .footer { text-align: center; margin-top: 40px; border-top: 2px solid #000; padding-top: 20px; }
    ''')
    html_parts.append('</style>')
    html_parts.append('</head>')
    html_parts.append('<body>')
    
    # Header
    html_parts.append('<div class="header">')
    html_parts.append(f'<div class="title">{html_mod.escape(generated["contract_title"])}</div>')
    html_parts.append(f'<div class="meta">Type: {html_mod.escape(generated["contract_type"])} | Jurisdiction: {html_mod.escape(generated["jurisdiction"])}</div>')
    html_parts.append(f'<div class="meta">Generated: {generated["generated_at"].strftime("%B %d, %Y at %I:%M %p")}</div>')
    html_parts.append('</div>')
    
    # Clauses
    for i, clause in enumerate(generated["clauses"], 1):
        html_parts.append('<div class="clause">')
        html_parts.append(f'<div class="clause-title">{i}. {clause["clause_type"].upper().replace("_", " ")}</div>')
        
        # Highlight missing parameters
        text = html_mod.escape(clause["rendered_text"])
        for missing in clause["missing_parameters"]:
            escaped_m = html_mod.escape(missing)
            text = text.replace(escaped_m, f'<span class="missing-param">{escaped_m}</span>')
        
        html_parts.append(f'<div class="clause-text">{text}</div>')
        html_parts.append('</div>')
    
    # Footer
    html_parts.append('<div class="footer">')
    html_parts.append('<p><strong>END OF CONTRACT</strong></p>')
    if not generated["is_complete"]:
        html_parts.append(f'<p style="color: red;">⚠️ Missing {len(generated["missing_parameters"])} parameters</p>')
    html_parts.append('</div>')
    
    html_parts.append('</body>')
    html_parts.append('</html>')
    
    return {"content": "\n".join(html_parts), "content_type": "text/html"}


@router.get("/{contract_id}/status")
async def get_contract_status(contract_id: str, user=Depends(get_current_user)):
    """
    Step 5.4: Get contract generation status
    
    Check if contract is ready for generation
    """
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check if contract exists
            cur.execute("SELECT id FROM contracts WHERE id = %s", (contract_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Contract not found")
            
            # Check active clauses
            cur.execute("""
                SELECT COUNT(*) as clause_count
                FROM contract_clauses
                WHERE contract_id = %s AND is_active = true
            """, (contract_id,))
            clause_count = cur.fetchone()["clause_count"]
            
            # Check parameters
            cur.execute("""
                SELECT COUNT(*) as param_count
                FROM contract_parameters
                WHERE contract_id = %s
            """, (contract_id,))
            param_count = cur.fetchone()["param_count"]
    finally:
        conn.close()
    
    # is_ready requires clauses AND that all required parameters are filled
    # (param_count > 0 alone is a false positive if only 1 of N is set)
    is_ready = clause_count > 0 and param_count > 0
    if is_ready:
        # Double-check by generating and seeing if any params are missing
        try:
            param_values = get_parameter_values(contract_id)
            param_names = get_parameter_names_map(contract_id)
            total_required = len(param_names)
            filled = sum(1 for pid in param_names if pid in param_values)
            is_ready = filled >= total_required and total_required > 0
        except Exception:
            is_ready = False
    
    return {
        "contract_id": contract_id,
        "is_ready": is_ready,
        "active_clauses_count": clause_count,
        "filled_parameters_count": param_count,
        "next_step": "Generate contract" if is_ready else "Complete Steps 3 & 4 first"
    }
