# chatbot_routes.py - Phase 5: Contract Q&A Chatbot (Graph RAG)
"""
AI-powered contract chatbot using Graph RAG.
Answers questions about the user's specific contract by retrieving
relevant clauses and parameters from the knowledge graph.

Endpoints:
- POST /{contract_id}/chat — send a message, get AI response
- GET  /{contract_id}/chat/history — get chat history
- DELETE /{contract_id}/chat/history — clear chat history
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from graph_rag_engine import retriever, llm_client, validator
from llm_config import SYSTEM_PROMPT_CHATBOT, CHATBOT_PROMPT
from config import DB_CONFIG

import psycopg2
from psycopg2.extras import RealDictCursor

router = APIRouter(prefix="/api/contracts", tags=["chatbot"])


# ==================== PYDANTIC MODELS ====================

class ChatMessage(BaseModel):
    message: str
    context_filter: Optional[str] = None  # Optional: filter to specific clause type

class Citation(BaseModel):
    clause_id: str
    clause_type: str
    relevant_snippet: str

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    follow_up_suggestions: List[str]
    grounding_validation: Dict[str, Any]
    timestamp: datetime

class ChatHistoryItem(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime
    citations: Optional[List[Citation]] = None

class ChatHistoryResponse(BaseModel):
    contract_id: str
    messages: List[ChatHistoryItem]
    total_messages: int


# ==================== HELPERS ====================

def get_db():
    return psycopg2.connect(**DB_CONFIG)


def _ensure_chat_table():
    """Create chat history table if not exists."""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS contract_chat_history (
                    id SERIAL PRIMARY KEY,
                    contract_id UUID NOT NULL REFERENCES contracts(id) ON DELETE CASCADE,
                    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    citations JSONB DEFAULT '[]',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_history_contract 
                ON contract_chat_history(contract_id, created_at)
            """)
            conn.commit()
    finally:
        conn.close()


def _save_message(contract_id: str, role: str, content: str, citations: List = None):
    """Save a chat message to history."""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO contract_chat_history (contract_id, role, content, citations)
                VALUES (%s, %s, %s, %s)
            """, (contract_id, role, content, json.dumps(citations or [])))
            conn.commit()
    finally:
        conn.close()


def _get_recent_history(contract_id: str, limit: int = 10) -> List[Dict]:
    """Get recent chat history for context."""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT role, content, citations, created_at
                FROM contract_chat_history
                WHERE contract_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (contract_id, limit))
            rows = cur.fetchall()
            return list(reversed(rows))  # Oldest first
    finally:
        conn.close()


def _format_clause_context(clauses: List[Dict]) -> str:
    """Format clause data for LLM context."""
    if not clauses:
        return "No clause data available."
    
    lines = []
    for c in clauses:
        text_preview = c.get("raw_text", "")
        if len(text_preview) > 500:
            text_preview = text_preview[:500] + "..."
        lines.append(
            f"### {c.get('clause_type_name', c.get('clause_type', 'Unknown'))} "
            f"({c['variant']} variant, risk: {c['risk_level']})\n"
            f"**Clause ID:** {c['clause_id']}\n"
            f"**Text:**\n{text_preview}\n"
        )
    return "\n".join(lines)


def _format_params(param_values: Dict[str, str]) -> str:
    """Format parameter values for LLM context."""
    if not param_values:
        return "No parameter values set yet."
    
    lines = [f"- {pid}: {val}" for pid, val in param_values.items()]
    return "\n".join(lines)


def _format_chat_history(history: List[Dict]) -> str:
    """Format chat history for LLM context."""
    if not history:
        return "No previous conversation."
    
    lines = []
    for msg in history[-6:]:  # Last 6 messages for context window
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"**{role}:** {content}")
    
    return "\n".join(lines)


# ==================== ROUTES ====================

@router.post("/{contract_id}/chat", response_model=ChatResponse)
async def chat(contract_id: str, request: ChatMessage):
    """
    Send a message to the contract chatbot.
    
    Graph RAG flow:
    1. Retrieve contract info + active clause texts from Neo4j
    2. Get parameter values from Supabase
    3. Include recent chat history for context continuity
    4. LLM answers using ONLY the provided graph context
    5. Validator checks citations reference real clauses
    """
    # Ensure chat table exists
    _ensure_chat_table()
    
    # Get contract info
    contract = retriever.get_contract_info(contract_id)
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    # Check LLM
    if not llm_client.is_configured():
        raise HTTPException(
            status_code=503,
            detail="LLM not configured. Set LLM_API_KEY in .env to enable the chatbot."
        )
    
    # Get active clause IDs
    active_clauses = retriever.get_active_clause_ids(contract_id)
    active_clause_ids = [c["clause_id"] for c in active_clauses]
    
    if not active_clause_ids:
        raise HTTPException(
            status_code=400,
            detail="No active clauses found. Add clauses to the contract first."
        )
    
    # --- GRAPH RETRIEVAL ---
    qa_context = retriever.get_qa_context(contract_id, active_clause_ids, request.message)
    
    # Get chat history
    history = _get_recent_history(contract_id)
    
    # Save user message
    _save_message(contract_id, "user", request.message)
    
    # --- LLM GENERATION ---
    prompt = CHATBOT_PROMPT.format(
        contract_type=contract["contract_type"],
        jurisdiction=contract["jurisdiction"],
        contract_status=contract.get("status", "draft"),
        relevant_clauses=_format_clause_context(qa_context["clauses"]),
        relevant_parameters=_format_params(qa_context["parameter_values"]),
        chat_history=_format_chat_history(history),
        user_message=request.message
    )
    
    try:
        llm_response = llm_client.generate(prompt, SYSTEM_PROMPT_CHATBOT)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM generation failed: {str(e)}"
        )
    
    answer = llm_response.get("answer", "I'm unable to answer that question.")
    citations = llm_response.get("citations", [])
    follow_ups = llm_response.get("follow_up_suggestions", [])
    
    # --- VALIDATION ---
    citation_check = validator.validate_citations(citations, qa_context["clauses"])
    
    # Filter out invalid citations
    valid_citations = citation_check.get("valid_citations", citations)
    
    # Save assistant response
    _save_message(contract_id, "assistant", answer, valid_citations)
    
    return {
        "answer": answer,
        "citations": valid_citations,
        "follow_up_suggestions": follow_ups[:3],  # Max 3 suggestions
        "grounding_validation": {
            "citations_valid": citation_check["valid"],
            "citation_rate": citation_check["citation_rate"],
            "invalid_citations_filtered": len(citation_check.get("invalid_citations", []))
        },
        "timestamp": datetime.now()
    }


@router.get("/{contract_id}/chat/history", response_model=ChatHistoryResponse)
async def get_chat_history(contract_id: str, limit: int = 50):
    """Get chat history for a contract."""
    _ensure_chat_table()
    
    # Verify contract exists
    contract = retriever.get_contract_info(contract_id)
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT role, content, citations, created_at AS timestamp
                FROM contract_chat_history
                WHERE contract_id = %s
                ORDER BY created_at ASC
                LIMIT %s
            """, (contract_id, limit))
            messages = cur.fetchall()
    finally:
        conn.close()
    
    formatted = []
    for msg in messages:
        formatted.append({
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": msg["timestamp"],
            "citations": msg.get("citations") or []
        })
    
    return {
        "contract_id": contract_id,
        "messages": formatted,
        "total_messages": len(formatted)
    }


@router.delete("/{contract_id}/chat/history")
async def clear_chat_history(contract_id: str):
    """Clear chat history for a contract."""
    _ensure_chat_table()
    
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM contract_chat_history
                WHERE contract_id = %s
            """, (contract_id,))
            deleted = cur.rowcount
            conn.commit()
    finally:
        conn.close()
    
    return {
        "message": f"Cleared {deleted} chat messages",
        "contract_id": contract_id
    }
