# parameters_routes.py - Step 4: Parameter Management (FINAL)
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from datetime import datetime, date
from neo4j import GraphDatabase
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from config import DB_CONFIG, NEO4J_CONFIG

router = APIRouter(prefix="/api/contracts", tags=["parameters"])

# ==================== PYDANTIC MODELS ====================

class ParameterDefinition(BaseModel):
    """Parameter definition from Neo4j"""
    id: str
    name: str
    data_type: str
    is_required: bool
    created_at: Optional[str] = None
    used_in_clauses: List[str]

class ParameterValue(BaseModel):
    """Single parameter value - flexible for template compatibility"""
    parameter_id: str
    value: Any
    
    # Optional fields (ignored but allowed for template compatibility)
    name: Optional[str] = None
    data_type: Optional[str] = None
    is_required: Optional[bool] = None
    
    class Config:
        extra = "ignore"  # Ignore any extra fields
class BulkSetParametersRequest(BaseModel):
    """Bulk set multiple parameters"""
    parameters: List[ParameterValue]

class ContractParameterResponse(BaseModel):
    """Saved parameter response"""
    id: int
    contract_id: str
    parameter_id: str
    value_text: Optional[str]
    value_integer: Optional[int]
    value_decimal: Optional[float]
    value_date: Optional[date]
    value_currency: Optional[Dict]
    provided_by: Optional[str]
    created_at: datetime
    updated_at: datetime

# ==================== DATABASE HELPERS ====================
from config import get_connection, get_neo4j_driver

def get_db():
    """Legacy shim — callers should migrate to get_connection() context manager."""
    import psycopg2
    from config import DB_CONFIG
    return psycopg2.connect(**DB_CONFIG)

# ==================== NEO4J QUERIES ====================

def fetch_parameters_for_active_clauses(contract_id: str) -> List[Dict]:
    """
    Fetch parameters from Neo4j for ACTIVE clauses only
    """
    # Get active clause_ids from Supabase
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT clause_id 
                FROM contract_clauses
                WHERE contract_id = %s AND is_active = true
                ORDER BY sequence
            """, (contract_id,))
            
            clause_rows = cur.fetchall()
            clause_ids = [row["clause_id"] for row in clause_rows]
    finally:
        conn.close()
    
    if not clause_ids:
        return []
    
    # Fetch parameters from Neo4j using EXACT property names
    driver = get_neo4j_driver()
    
    cypher_query = """
    MATCH (c:Clause)-[:CONTAINS_PARAM]->(p:Parameter)
    WHERE c.id IN $clause_ids
    RETURN DISTINCT
      p.id AS id,
      p.name AS name,
      p.data_type AS data_type,
      p.is_required AS is_required,
      p.created_at AS created_at,
      collect(DISTINCT c.id) AS used_in_clauses
    ORDER BY 
      CASE WHEN p.is_required = true THEN 1 ELSE 2 END,
      p.name
    """
    
    with driver.session() as session:
        result = session.run(cypher_query, {"clause_ids": clause_ids})
        parameters = []
        for record in result:
            parameters.append({
                "id": record["id"],
                "name": record["name"],
                "data_type": record["data_type"],
                "is_required": record["is_required"],
                "created_at": str(record["created_at"]) if record["created_at"] else None,
                "used_in_clauses": list(record["used_in_clauses"])
            })
        return parameters

# ==================== VALUE CONVERSION HELPER ====================

def convert_parameter_value(data_type: str, value: Any) -> tuple:
    """
    Convert value to appropriate column based on data_type
    Returns: (value_text, value_integer, value_decimal, value_date, value_currency)
    """
    if value is None or value == "":
        return None, None, None, None, None
    
    # Normalize data_type (case-insensitive)
    data_type_lower = data_type.lower()
    
    if data_type_lower == "string" or data_type_lower == "text":
        return str(value), None, None, None, None
    
    elif data_type_lower == "integer" or data_type_lower == "int" or data_type_lower == "number":
        return None, int(value), None, None, None
    
    elif data_type_lower == "decimal" or data_type_lower == "float" or data_type_lower == "double":
        return None, None, float(value), None, None
    
    elif data_type_lower == "date" or data_type_lower == "datetime":
        if isinstance(value, str):
            from datetime import datetime
            parsed_date = datetime.strptime(value, "%Y-%m-%d").date()
            return None, None, None, parsed_date, None
        elif isinstance(value, date):
            return None, None, None, value, None
        else:
            return str(value), None, None, None, None
    
    elif data_type_lower == "currency" or data_type_lower == "money":
        if isinstance(value, dict):
            return None, None, None, None, value
        else:
            return str(value), None, None, None, None
    
    elif data_type_lower == "boolean" or data_type_lower == "bool":
        return str(value).lower(), None, None, None, None
    
    else:
        # Default: store as text
        return str(value), None, None, None, None

# ==================== ROUTES ====================
# ==================== TEMPLATE ROUTES FOR TESTING ====================

@router.get("/{contract_id}/parameters/grouped")
async def get_parameters_grouped(
    contract_id: str,
    format: str = "display"  # "display" or "template"
):
    """
    Get parameters grouped by clause type for better UX
    
    Query params:
    - format=display (default): Full parameter details for UI
    - format=template: Simplified format ready for POST /bulk
    
    Examples:
    - GET /api/contracts/{id}/parameters/grouped
    - GET /api/contracts/{id}/parameters/grouped?format=template
    """
    # Verify contract exists
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id FROM contracts WHERE id = %s", (contract_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Contract not found")
    finally:
        conn.close()
    
    # Fetch parameters for active clauses
    parameters = fetch_parameters_for_active_clauses(contract_id)
    
    if not parameters:
        return {
            "groups": [],
            "total_parameters": 0,
            "message": "No parameters found for active clauses"
        }
    
    # Group by clause type
    grouped = {}
    for param in parameters:
        for clause_id in param["used_in_clauses"]:
            clause_type = clause_id.split('_')[0]
            
            if clause_type not in grouped:
                grouped[clause_type] = {
                    "clause_type": clause_type,
                    "clause_type_label": get_clause_type_label(clause_type),
                    "parameters": [],
                    "required_count": 0,
                    "optional_count": 0
                }
            
            # Avoid duplicates
            param_ids = [p["id"] for p in grouped[clause_type]["parameters"]]
            if param["id"] not in param_ids:
                grouped[clause_type]["parameters"].append(param)
                
                if param["is_required"]:
                    grouped[clause_type]["required_count"] += 1
                else:
                    grouped[clause_type]["optional_count"] += 1
    
    sorted_groups = sorted(grouped.values(), key=lambda x: x["clause_type"])
    
    # Return template format for testing
    if format == "template":
        return {
            "instructions": "Fill in the 'value' field for each parameter, then POST to the endpoint below",
            "post_url": f"/api/contracts/{contract_id}/parameters/bulk",
            "post_method": "POST",
            "parameters": [
                {
                    "parameter_id": param["id"],
                    "name": param["name"],
                    "data_type": param["data_type"],
                    "is_required": param["is_required"],
                    "value": ""  # Empty - ready to fill
                }
                for group in sorted_groups
                for param in group["parameters"]
            ]
        }
    
    # Default: Display format for UI
    return {
        "groups": sorted_groups,
        "total_parameters": len(parameters),
        "total_groups": len(sorted_groups),
        "required_parameters": sum(1 for p in parameters if p["is_required"]),
        "optional_parameters": sum(1 for p in parameters if not p["is_required"])
    }
def get_clause_type_label(clause_type: str) -> str:
    """
    Convert clause type code to human-readable label
    """
    labels = {
        "PART": "Parties & Recitals",
        "DEFN": "Definitions",
        "SCOPE": "Scope of Agreement",
        "CONF": "Confidentiality",
        "NDISC": "Non-Disclosure",
        "NONCOMP": "Non-Compete",
        "NONSOL": "Non-Solicitation",
        "IP": "Intellectual Property",
        "PAY": "Payment Terms",
        "TERM": "Term & Renewal",
        "TERMB": "Termination for Cause",
        "TERMC": "Termination for Convenience",
        "SURV": "Survival & Effect of Termination",
        "REP": "Representations & Warranties",
        "INDEM": "Indemnification",
        "LIAB": "Limitation of Liability",
        "FORCE": "Force Majeure",
        "GOV": "Governing Law",
        "DISP": "Dispute Resolution",
        "AMEND": "Amendments",
        "ENTIRE": "Entire Agreement",
        "SEVER": "Severability",
        "NOTICE": "Notices",
        "ASSIGN": "Assignment",
        "WAIVER": "Waiver"
    }
    return labels.get(clause_type, clause_type)
@router.get("/{contract_id}/parameters/required", response_model=List[ParameterDefinition])
async def get_required_parameters(contract_id: str):
    """
    Step 4.1: Get all parameters needed for active clauses
    
    Returns parameter definitions from Neo4j for frontend form rendering
    """
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id FROM contracts WHERE id = %s", (contract_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Contract not found")
    finally:
        conn.close()
    
    parameters = fetch_parameters_for_active_clauses(contract_id)
    return parameters


@router.get("/{contract_id}/parameters/values", response_model=List[ContractParameterResponse])
async def get_parameter_values(contract_id: str):
    """
    Get saved parameter values for a contract
    """
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM contract_parameters
                WHERE contract_id = %s
                ORDER BY created_at
            """, (contract_id,))
            return cur.fetchall()
    finally:
        conn.close()


@router.get("/{contract_id}/parameters/form", response_model=Dict)
async def get_parameter_form(contract_id: str):
    """
    Step 4.2: Get complete form data (definitions + saved values)
    
    Perfect for frontend to render a form with pre-filled values
    """
    # Get parameter definitions
    parameters = fetch_parameters_for_active_clauses(contract_id)
    
    # Get saved values
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
            
            saved_rows = cur.fetchall()
            saved_values = {}
            for row in saved_rows:
                param_id = row["parameter_id"]
                # Get actual value from appropriate column
                if row["value_text"] is not None:
                    saved_values[param_id] = row["value_text"]
                elif row["value_integer"] is not None:
                    saved_values[param_id] = row["value_integer"]
                elif row["value_decimal"] is not None:
                    saved_values[param_id] = float(row["value_decimal"])
                elif row["value_date"] is not None:
                    saved_values[param_id] = row["value_date"].isoformat()
                elif row["value_currency"] is not None:
                    saved_values[param_id] = row["value_currency"]
    finally:
        conn.close()
    
    # Calculate completion
    required_params = [p for p in parameters if p["is_required"]]
    filled_required = sum(1 for p in required_params if p["id"] in saved_values)
    
    return {
        "parameter_definitions": parameters,
        "saved_values": saved_values,
        "total_parameters": len(parameters),
        "total_required": len(required_params),
        "filled_required": filled_required,
        "completion_percentage": (filled_required / len(required_params) * 100) if required_params else 100
    }


@router.post("/{contract_id}/parameters", response_model=ContractParameterResponse)
async def set_parameter_value(contract_id: str, request: ParameterValue, provided_by: Optional[str] = None):
    """
    Step 4.3: Set a single parameter value
    
    Automatically determines which column to use based on data_type from Neo4j
    """
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id FROM contracts WHERE id = %s", (contract_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Contract not found")
    finally:
        conn.close()
    
    # Get parameter data_type from Neo4j
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Parameter {id: $parameter_id})
            RETURN p.data_type AS data_type
        """, {"parameter_id": request.parameter_id})
        
        param_def = result.single()
        if not param_def:
            raise HTTPException(
                status_code=404, 
                detail=f"Parameter '{request.parameter_id}' not found in Neo4j"
            )
        
        data_type = param_def["data_type"]
    
    # Convert value to appropriate columns
    value_text, value_integer, value_decimal, value_date, value_currency = convert_parameter_value(
        data_type, request.value
    )
    
    # Upsert parameter value
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO contract_parameters (
                    contract_id, parameter_id,
                    value_text, value_integer, value_decimal, 
                    value_date, value_currency, provided_by
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (contract_id, parameter_id) 
                DO UPDATE SET
                    value_text = EXCLUDED.value_text,
                    value_integer = EXCLUDED.value_integer,
                    value_decimal = EXCLUDED.value_decimal,
                    value_date = EXCLUDED.value_date,
                    value_currency = EXCLUDED.value_currency,
                    provided_by = EXCLUDED.provided_by,
                    updated_at = NOW()
                RETURNING *
            """, (
                contract_id, request.parameter_id,
                value_text, value_integer, value_decimal, value_date,
                json.dumps(value_currency) if value_currency else None,
                provided_by
            ))
            
            result = cur.fetchone()
            conn.commit()
            return result
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error saving parameter: {str(e)}")
    finally:
        conn.close()


@router.post("/{contract_id}/parameters/bulk", response_model=List[ContractParameterResponse])
async def set_parameters_bulk(contract_id: str, request: BulkSetParametersRequest, provided_by: Optional[str] = None):
    """
    Step 4.4: Set multiple parameters at once
    
    Frontend submits entire form in one request
    """
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id FROM contracts WHERE id = %s", (contract_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Contract not found")
    finally:
        conn.close()
    
    # Get all parameter data_types from Neo4j
    driver = get_neo4j_driver()
    param_ids = [p.parameter_id for p in request.parameters]
    
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Parameter)
            WHERE p.id IN $param_ids
            RETURN p.id AS id, p.data_type AS data_type
        """, {"param_ids": param_ids})
        
        param_types = {rec["id"]: rec["data_type"] for rec in result}
    
    # Insert/update all parameters
    conn = get_db()
    saved_params = []
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for param in request.parameters:
                if param.parameter_id not in param_types:
                    continue  # Skip unknown parameters
                
                data_type = param_types[param.parameter_id]
                
                # Convert value
                value_text, value_integer, value_decimal, value_date, value_currency = convert_parameter_value(
                    data_type, param.value
                )
                
                # Upsert
                cur.execute("""
                    INSERT INTO contract_parameters (
                        contract_id, parameter_id,
                        value_text, value_integer, value_decimal, 
                        value_date, value_currency, provided_by
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (contract_id, parameter_id) 
                    DO UPDATE SET
                        value_text = EXCLUDED.value_text,
                        value_integer = EXCLUDED.value_integer,
                        value_decimal = EXCLUDED.value_decimal,
                        value_date = EXCLUDED.value_date,
                        value_currency = EXCLUDED.value_currency,
                        provided_by = EXCLUDED.provided_by,
                        updated_at = NOW()
                    RETURNING *
                """, (
                    contract_id, param.parameter_id,
                    value_text, value_integer, value_decimal, value_date,
                    json.dumps(value_currency) if value_currency else None,
                    provided_by
                ))
                
                saved_params.append(cur.fetchone())
            
            conn.commit()
            return saved_params
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error saving parameters: {str(e)}")
    finally:
        conn.close()


@router.delete("/{contract_id}/parameters/{parameter_id}")
async def delete_parameter_value(contract_id: str, parameter_id: str):
    """Delete a parameter value (for optional parameters)"""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                DELETE FROM contract_parameters
                WHERE contract_id = %s AND parameter_id = %s
                RETURNING parameter_id
            """, (contract_id, parameter_id))
            
            result = cur.fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="Parameter not found")
            
            conn.commit()
            return {"message": f"Parameter {parameter_id} deleted successfully"}
    finally:
        conn.close()


@router.get("/{contract_id}/parameters/validation")
async def validate_parameters(contract_id: str):
    """
    Step 4.5: Validate all required parameters are filled
    
    Returns:
    - is_complete: Can user proceed to Step 5?
    - missing_required: List of required parameters not yet filled
    """
    parameters = fetch_parameters_for_active_clauses(contract_id)
    required_params = [p for p in parameters if p["is_required"]]
    
    if not required_params:
        return {
            "is_complete": True,
            "total_required": 0,
            "filled_required": 0,
            "missing_required": [],
            "message": "No required parameters"
        }
    
    # Get filled parameters
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT parameter_id 
                FROM contract_parameters
                WHERE contract_id = %s
            """, (contract_id,))
            
            filled_param_ids = {row["parameter_id"] for row in cur.fetchall()}
    finally:
        conn.close()
    
    # Find missing required parameters
    missing_required = []
    for param in required_params:
        if param["id"] not in filled_param_ids:
            missing_required.append({
                "parameter_id": param["id"],
                "parameter_name": param["name"],
                "data_type": param["data_type"]
            })
    
    return {
        "is_complete": len(missing_required) == 0,
        "total_required": len(required_params),
        "filled_required": len(required_params) - len(missing_required),
        "missing_required": missing_required,
        "completion_percentage": ((len(required_params) - len(missing_required)) / len(required_params) * 100) if required_params else 100
    }
