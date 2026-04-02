# neo4j_routes.py - Step 3 Backend Routes with Variant Management
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from neo4j import GraphDatabase
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
from config import DB_CONFIG, NEO4J_CONFIG

# Database configs
# DB_CONFIG = {
#     "host": "db.wjbijphzxqizbbgpbacg.supabase.co",
#     "port": 5432,
#     "dbname": "postgres",
#     "user": "postgres",
#     "password": "Sapvoyagers@1234",
#     "sslmode": "require"
# }

# NEO4J_CONFIG = {
#     "uri": "neo4j+s://3c5b6f0d.databases.neo4j.io",
#     "username": "neo4j",
#     "password": "CgG5iYCxef1ExRTXTenDiO6wtXzQPgiDECdWDdmJi38",
#     "database": "neo4j"
# }

router = APIRouter(prefix="/api/contracts", tags=["clauses"])

# ==================== PYDANTIC MODELS ====================

class ClauseResponse(BaseModel):
    id: int
    contract_id: str
    clause_id: str
    clause_type: str
    variant: str
    sequence: int
    is_mandatory: bool
    is_customized: bool = False
    is_active: bool = True  # NEW
    overridden_text: Optional[str] = None
    parameters_bound: Optional[Dict] = None
    created_at: datetime
    updated_at: datetime

class UpdateClauseRequest(BaseModel):
    sequence: Optional[int] = None
    is_customized: Optional[bool] = None
    overridden_text: Optional[str] = None
    parameters_bound: Optional[Dict] = None

class AddOptionalClauseRequest(BaseModel):
    clause_id: str
    clause_type: str
    variant: str
    sequence: int

class SwitchVariantRequest(BaseModel):
    clause_type: str
    new_variant: str

# ==================== DATABASE HELPERS ====================

def get_db():
    return psycopg2.connect(**DB_CONFIG)

def get_neo4j_driver():
    return GraphDatabase.driver(
        NEO4J_CONFIG["uri"],
        auth=(NEO4J_CONFIG["username"], NEO4J_CONFIG["password"]),
        database=NEO4J_CONFIG["database"]
    )

# ==================== NEO4J QUERIES ====================

def fetch_clauses_from_neo4j(contract_type: str, jurisdiction: str):
    """Fetch ALL clauses (all variants) from Neo4j"""
    driver = get_neo4j_driver()
    neo4j_ct_id = contract_type.replace("_", "-")
    
    cypher_query = """
    MATCH (ct:ContractType {id: $contract_type})
          -[rel:CONTAINS_CLAUSE]->(clauseType:ClauseType)
          -[hv:HAS_VARIANT]->(c:Clause)
    WHERE c.jurisdiction = $jurisdiction
    RETURN
      c.id AS clause_id,
      clauseType.id AS clause_type,
      c.variant AS variant,
      c.raw_text AS raw_text,
      rel.sequence AS sequence,
      rel.mandatory AS is_mandatory,
      rel.description AS clause_description
    ORDER BY rel.sequence, c.variant
    """
    
    try:
        with driver.session() as session:
            result = session.run(cypher_query, {
                "contract_type": neo4j_ct_id,
                "jurisdiction": jurisdiction
            })
            
            clauses = []
            for record in result:
                clauses.append({
                    "clause_id": record["clause_id"],
                    "clause_type": record["clause_type"],
                    "variant": record["variant"],
                    "raw_text": record["raw_text"],
                    "sequence": int(record["sequence"]),
                    "is_mandatory": bool(record["is_mandatory"]),
                    "clause_description": record["clause_description"]
                })
            return clauses
    finally:
        driver.close()

# ==================== ROUTES ====================

@router.post("/{contract_id}/clauses/generate", response_model=List[ClauseResponse])
async def generate_clauses(contract_id: str, default_variant: str = "Moderate"):
    """
    Step 3: Generate clauses from Neo4j → contract_clauses table
    
    - Inserts ALL variants (Standard, Moderate, Strict) for all clause types
    - Marks only default_variant as active (is_active = true)
    - Other variants stored as inactive (is_active = false)
    
    Query params:
      - default_variant: "Standard" | "Moderate" | "Strict" (default: Moderate)
    """
    pg_conn = get_db()
    try:
        with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Verify contract exists & get type/jurisdiction
            cur.execute("""
                SELECT id, contract_type, jurisdiction
                FROM contracts WHERE id = %s
            """, (contract_id,))
            contract = cur.fetchone()
            
            if not contract:
                raise HTTPException(status_code=404, detail="Contract not found")
            
            # Check if clauses already exist
            cur.execute("""
                SELECT COUNT(*) as count FROM contract_clauses
                WHERE contract_id = %s
            """, (contract_id,))
            if cur.fetchone()["count"] > 0:
                raise HTTPException(
                    status_code=400, 
                    detail="Clauses already generated. Use DELETE /clauses to regenerate."
                )
        
        # Fetch ALL clauses (all variants) from Neo4j
        neo4j_clauses = fetch_clauses_from_neo4j(
            contract["contract_type"],
            contract["jurisdiction"]
        )
        
        if not neo4j_clauses:
            raise HTTPException(
                status_code=404,
                detail=f"No clauses for {contract['contract_type']} in {contract['jurisdiction']}"
            )
        
        # Insert ALL variants into database
        with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            inserted_clauses = []
            for clause in neo4j_clauses:
                # Mark only default_variant as active
                is_active = (clause["variant"] == default_variant)
                
                cur.execute("""
                    INSERT INTO contract_clauses (
                        contract_id, clause_id, clause_type, variant,
                        sequence, is_mandatory, is_customized, is_active
                    ) VALUES (%s, %s, %s, %s, %s, %s, false, %s)
                    RETURNING *
                """, (
                    contract_id,
                    clause["clause_id"],
                    clause["clause_type"],
                    clause["variant"],
                    clause["sequence"],
                    clause["is_mandatory"],
                    is_active
                ))
                inserted_clauses.append(cur.fetchone())
            
            pg_conn.commit()
            return inserted_clauses
    
    except HTTPException:
        raise
    except Exception as e:
        pg_conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        pg_conn.close()


@router.get("/{contract_id}/clauses/active", response_model=List[Dict])
async def get_active_clauses(contract_id: str):
    """
    Get active clauses with full text from Neo4j
    Shows all variant options with their texts for comparison
    """
    conn = get_db()
    driver = get_neo4j_driver()
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get ALL clauses (active + variants)
            cur.execute("""
                SELECT * FROM contract_clauses
                WHERE contract_id = %s
                ORDER BY sequence, 
                  CASE variant 
                    WHEN 'Standard' THEN 1
                    WHEN 'Moderate' THEN 2
                    WHEN 'Strict' THEN 3
                    ELSE 4
                  END
            """, (contract_id,))
            
            all_clauses = cur.fetchall()
            
            if not all_clauses:
                raise HTTPException(
                    status_code=404, 
                    detail="No clauses found. Please generate clauses first."
                )
            
            # Collect all clause_ids
            clause_ids = [c["clause_id"] for c in all_clauses]
        
        # Batch fetch all texts from Neo4j in ONE query
        with driver.session() as session:
            result = session.run("""
                MATCH (c:Clause)
                WHERE c.id IN $clause_ids
                OPTIONAL MATCH (ct:ClauseType)-[:HAS_VARIANT]->(c)
                RETURN
                  c.id AS clause_id,
                  c.raw_text AS raw_text,
                  c.risk_level AS risk_level,
                  ct.name AS clause_type_name
            """, {"clause_ids": clause_ids})
            
            # Build lookup dict
            neo4j_data = {}
            for rec in result:
                neo4j_data[rec["clause_id"]] = {
                    "raw_text": rec.get("raw_text"),
                    "risk_level": rec.get("risk_level"),
                    "clause_type_name": rec.get("clause_type_name")
                }
        
        # Group clauses by clause_type
        clause_groups = {}
        for clause in all_clauses:
            clause_type = clause["clause_type"]
            if clause_type not in clause_groups:
                clause_groups[clause_type] = []
            
            # Merge Supabase data + Neo4j data
            clause_dict = dict(clause)
            neo4j_info = neo4j_data.get(clause["clause_id"], {})
            clause_dict["raw_text"] = neo4j_info.get("raw_text")
            clause_dict["risk_level"] = neo4j_info.get("risk_level")
            clause_dict["clause_type_name"] = neo4j_info.get("clause_type_name")
            
            clause_groups[clause_type].append(clause_dict)
        
        # Build result: active clauses with variants
        result = []
        for clause_type, variants in clause_groups.items():
            # Find active variant
            active = next((v for v in variants if v["is_active"]), None)
            
            if active:
                result.append({
                    **active,
                    "available_variants": variants
                })
        
        # Sort by sequence
        result.sort(key=lambda x: x["sequence"])
        
        return result
    
    finally:
        conn.close()
        driver.close()



@router.get("/{contract_id}/clauses", response_model=List[ClauseResponse])
async def get_clauses(contract_id: str, is_active: Optional[bool] = None):
    """
    Get clauses for contract
    
    Query params:
      - is_active: true (only active) | false (only inactive) | null (all clauses)
    """
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if is_active is not None:
                cur.execute("""
                    SELECT * FROM contract_clauses
                    WHERE contract_id = %s AND is_active = %s
                    ORDER BY sequence
                """, (contract_id, is_active))
            else:
                cur.execute("""
                    SELECT * FROM contract_clauses
                    WHERE contract_id = %s
                    ORDER BY sequence, variant
                """, (contract_id,))
            
            return cur.fetchall()
    finally:
        conn.close()


@router.put("/{contract_id}/clauses/switch-variant", response_model=ClauseResponse)
async def switch_clause_variant(contract_id: str, request: SwitchVariantRequest):
    """
    Switch active variant for a clause type
    
    Example: User changes Confidentiality from Moderate → Strict
    
    Request body:
    {
      "clause_type": "confidentiality",
      "new_variant": "Strict"
    }
    """
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Deactivate all variants of this clause_type
            cur.execute("""
                UPDATE contract_clauses
                SET is_active = false, updated_at = NOW()
                WHERE contract_id = %s 
                  AND clause_type = %s
            """, (contract_id, request.clause_type))
            
            # Activate the new variant
            cur.execute("""
                UPDATE contract_clauses
                SET is_active = true, updated_at = NOW()
                WHERE contract_id = %s 
                  AND clause_type = %s 
                  AND variant = %s
                RETURNING *
            """, (contract_id, request.clause_type, request.new_variant))
            
            result = cur.fetchone()
            
            if not result:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Variant '{request.new_variant}' not found for clause type '{request.clause_type}'"
                )
            
            conn.commit()
            
            # Auto-version: snapshot after variant switch
            try:
                from version_routes import create_version_snapshot
                create_version_snapshot(
                    contract_id,
                    f"Switched {request.clause_type} to {request.new_variant} variant"
                )
            except Exception:
                pass  # Don't fail the main operation if versioning fails
            
            return result
    
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        conn.close()


@router.get("/{contract_id}/clauses/{clause_db_id}", response_model=Dict)
async def get_clause_detail(contract_id: str, clause_db_id: int):
    """Get clause detail + full text from Neo4j"""
    pg_conn = get_db()
    try:
        with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Metadata from Postgres
            cur.execute("""
                SELECT * FROM contract_clauses
                WHERE id = %s AND contract_id = %s
            """, (clause_db_id, contract_id))
            
            clause = cur.fetchone()
            if not clause:
                raise HTTPException(status_code=404, detail="Clause not found")
        
        # Text from Neo4j
        driver = get_neo4j_driver()
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH (c:Clause {id: $clause_id})
                    OPTIONAL MATCH (ct:ClauseType)-[:HAS_VARIANT]->(c)
                    RETURN
                      c.raw_text AS raw_text,
                      ct.name AS clause_type_name,
                      c.risk_level AS risk_level
                """, {"clause_id": clause["clause_id"]})
                
                neo4j_data = result.single()
                if neo4j_data:
                    neo4j_dict = {
                        "raw_text": neo4j_data.get("raw_text"),
                        "clause_type_name": neo4j_data.get("clause_type_name"),
                        "risk_level": neo4j_data.get("risk_level")
                    }
                else:
                    neo4j_dict = {}
                
                return {
                    **dict(clause),
                    **neo4j_dict
                }
        finally:
            driver.close()
    finally:
        pg_conn.close()



@router.put("/{contract_id}/clauses/{clause_db_id}", response_model=ClauseResponse)
async def update_clause(contract_id: str, clause_db_id: int, request: UpdateClauseRequest):
    """
    Update clause (sequence, customization, parameters)
    """
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            updates = []
            params = []
            
            if request.sequence is not None:
                updates.append("sequence = %s")
                params.append(request.sequence)
            
            if request.is_customized is not None:
                updates.append("is_customized = %s")
                params.append(request.is_customized)
            
            if request.overridden_text is not None:
                updates.append("overridden_text = %s")
                params.append(request.overridden_text)
            
            if request.parameters_bound is not None:
                updates.append("parameters_bound = %s")
                params.append(json.dumps(request.parameters_bound))
            
            if not updates:
                raise HTTPException(status_code=400, detail="No fields to update")
            
            params.extend([clause_db_id, contract_id])
            
            cur.execute(f"""
                UPDATE contract_clauses
                SET {', '.join(updates)}, updated_at = NOW()
                WHERE id = %s AND contract_id = %s
                RETURNING *
            """, params)
            
            result = cur.fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="Clause not found")
            
            conn.commit()
            return result
    finally:
        conn.close()


@router.delete("/{contract_id}/clauses/{clause_db_id}")
async def delete_clause(contract_id: str, clause_db_id: int):
    """Delete optional clause only"""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT is_mandatory FROM contract_clauses
                WHERE id = %s AND contract_id = %s
            """, (clause_db_id, contract_id))
            
            clause = cur.fetchone()
            if not clause:
                raise HTTPException(status_code=404, detail="Clause not found")
            
            if clause["is_mandatory"]:
                raise HTTPException(status_code=400, detail="Cannot delete mandatory clause")
            
            cur.execute("""
                DELETE FROM contract_clauses
                WHERE id = %s AND contract_id = %s
            """, (clause_db_id, contract_id))
            
            conn.commit()
            return {"message": "Clause deleted successfully"}
    finally:
        conn.close()


@router.delete("/{contract_id}/clauses")
async def delete_all_clauses(contract_id: str):
    """Delete all clauses (to regenerate with different default variant)"""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                DELETE FROM contract_clauses
                WHERE contract_id = %s
                RETURNING clause_id
            """, (contract_id,))
            
            deleted = cur.fetchall()
            conn.commit()
            return {
                "message": f"Deleted {len(deleted)} clauses",
                "deleted_count": len(deleted)
            }
    finally:
        conn.close()


@router.post("/{contract_id}/clauses/add-optional", response_model=ClauseResponse)
async def add_optional_clause(contract_id: str, request: AddOptionalClauseRequest):
    """Manually add optional clause"""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Verify contract
            cur.execute("SELECT id FROM contracts WHERE id = %s", (contract_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Contract not found")
        
        # Verify Neo4j clause exists
        driver = get_neo4j_driver()
        try:
            with driver.session() as session:
                result = session.run(
                    "MATCH (c:Clause {id: $id}) RETURN c.id",
                    {"id": request.clause_id}
                )
                if not result.single():
                    raise HTTPException(status_code=404, detail="Clause not found in Neo4j")
        finally:
            driver.close()
        
        # Insert as optional
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO contract_clauses (
                    contract_id, clause_id, clause_type, variant,
                    sequence, is_mandatory, is_customized, is_active
                ) VALUES (%s, %s, %s, %s, %s, false, false, true)
                RETURNING *
            """, (
                contract_id, request.clause_id, request.clause_type,
                request.variant, request.sequence
            ))
            
            result = cur.fetchone()
            conn.commit()
            return result
    finally:
        conn.close()
