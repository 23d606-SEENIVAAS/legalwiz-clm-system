# parties_routes.py - Add these routes to your FastAPI app
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor
from config import get_db, DB_CONFIG, NEO4J_CONFIG

router = APIRouter(prefix="/api/contracts", tags=["parties"])

# Pydantic Models (matching your party_table.py schema)
class PartyRole(str, Enum):
    PARTY_A = "party_a"
    PARTY_B = "party_b"
    PARTY_C = "party_c"
    WITNESS_1 = "witness_1"
    WITNESS_2 = "witness_2"

class CreateParty(BaseModel):
    party_role: PartyRole
    party_name: str
    legal_entity_type: Optional[str] = None  # company/LLP/individual
    address_line1: Optional[str] = None
    address_line2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = "India"
    contact_person: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

class UpdateParty(BaseModel):
    party_name: Optional[str] = None
    legal_entity_type: Optional[str] = None
    address_line1: Optional[str] = None
    address_line2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None
    contact_person: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

class PartyResponse(BaseModel):
    id: int
    contract_id: str
    party_role: PartyRole
    party_name: str
    legal_entity_type: Optional[str]
    address_line1: Optional[str]
    address_line2: Optional[str]
    city: Optional[str]
    state: Optional[str]
    postal_code: Optional[str]
    country: Optional[str]
    contact_person: Optional[str]
    email: Optional[str]
    phone: Optional[str]


# ROUTES
@router.post("/{contract_id}/parties", response_model=PartyResponse, status_code=201)
async def add_party(contract_id: str, request: CreateParty):
    """Add party to contract (Party A, B, witnesses)"""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Verify contract exists
            cur.execute("SELECT id FROM contracts WHERE id = %s", (contract_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Contract not found")
            
            # Check if party_role already exists
            cur.execute("""
                SELECT id FROM contract_parties 
                WHERE contract_id = %s AND party_role = %s
            """, (contract_id, request.party_role))
            
            if cur.fetchone():
                raise HTTPException(
                    status_code=400, 
                    detail=f"{request.party_role} already exists for this contract"
                )
            
            # Insert party
            cur.execute("""
                INSERT INTO contract_parties (
                    contract_id, party_role, party_name, legal_entity_type,
                    address_line1, address_line2, city, state, postal_code, country,
                    contact_person, email, phone
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
            """, (
                contract_id, request.party_role, request.party_name, 
                request.legal_entity_type, request.address_line1, request.address_line2,
                request.city, request.state, request.postal_code, request.country,
                request.contact_person, request.email, request.phone
            ))
            
            result = cur.fetchone()
            conn.commit()
            return result
    finally:
        conn.close()

@router.get("/{contract_id}/parties", response_model=List[PartyResponse])
async def get_parties(contract_id: str):
    """Get all parties for contract"""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM contract_parties 
                WHERE contract_id = %s 
                ORDER BY CASE party_role 
                    WHEN 'party_a' THEN 1 
                    WHEN 'party_b' THEN 2 
                    WHEN 'party_c' THEN 3 
                    ELSE 4 
                END
            """, (contract_id,))
            return cur.fetchall()
    finally:
        conn.close()

@router.get("/{contract_id}/parties/{party_id}", response_model=PartyResponse)
async def get_party(contract_id: str, party_id: int):
    """Get specific party"""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM contract_parties 
                WHERE id = %s AND contract_id = %s
            """, (party_id, contract_id))
            
            result = cur.fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="Party not found")
            return result
    finally:
        conn.close()

@router.put("/{contract_id}/parties/{party_id}", response_model=PartyResponse)
async def update_party(contract_id: str, party_id: int, request: UpdateParty):
    """Update party details"""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            updates = []
            params = []
            
            if request.party_name:
                updates.append("party_name = %s")
                params.append(request.party_name)
            if request.legal_entity_type is not None:
                updates.append("legal_entity_type = %s")
                params.append(request.legal_entity_type)
            if request.address_line1 is not None:
                updates.append("address_line1 = %s")
                params.append(request.address_line1)
            if request.address_line2 is not None:
                updates.append("address_line2 = %s")
                params.append(request.address_line2)
            if request.city is not None:
                updates.append("city = %s")
                params.append(request.city)
            if request.state is not None:
                updates.append("state = %s")
                params.append(request.state)
            if request.postal_code is not None:
                updates.append("postal_code = %s")
                params.append(request.postal_code)
            if request.country is not None:
                updates.append("country = %s")
                params.append(request.country)
            if request.contact_person is not None:
                updates.append("contact_person = %s")
                params.append(request.contact_person)
            if request.email is not None:
                updates.append("email = %s")
                params.append(request.email)
            if request.phone is not None:
                updates.append("phone = %s")
                params.append(request.phone)
            
            if not updates:
                raise HTTPException(status_code=400, detail="No fields to update")
            
            params.extend([party_id, contract_id])
            
            cur.execute(f"""
                UPDATE contract_parties 
                SET {', '.join(updates)}, updated_at = NOW()
                WHERE id = %s AND contract_id = %s
                RETURNING *
            """, params)
            
            result = cur.fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="Party not found")
            
            conn.commit()
            return result
    finally:
        conn.close()

@router.delete("/{contract_id}/parties/{party_id}")
async def delete_party(contract_id: str, party_id: int):
    """Delete party from contract"""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM contract_parties 
                WHERE id = %s AND contract_id = %s
            """, (party_id, contract_id))
            
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Party not found")
            
            conn.commit()
            return {"message": "Party deleted successfully"}
    finally:
        conn.close()
