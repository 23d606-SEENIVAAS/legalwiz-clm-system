# llm_config.py - LLM Configuration & Prompt Templates
"""
Provider-agnostic LLM configuration for LegalWiz CLM.
Currently supports: Google Gemini (free tier)
Designed for easy swap to OpenAI, Anthropic, etc.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ==================== LLM PROVIDER CONFIG ====================

LLM_CONFIG = {
    "provider": os.getenv("LLM_PROVIDER", "gemini"),
    "api_key": os.getenv("LLM_API_KEY"),
    "model": os.getenv("LLM_MODEL", "gemini-2.0-flash"),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),  # Low for grounding
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "4096")),
}


# ==================== SYSTEM PROMPTS ====================

SYSTEM_PROMPT_BASE = """You are LegalWiz AI, a legal contract assistant. You help users understand, 
customize, and analyze legal contracts. You are embedded in a Contract Lifecycle Management system.

CRITICAL RULES:
1. You must ONLY use information provided in the context. NEVER invent legal language, clause IDs, or facts.
2. When referencing clauses, always cite the clause_id (e.g., CONF_MOD_001).
3. Risk levels, conflict data, and recommendations come from the knowledge graph — do NOT override them.
4. All your outputs must be valid JSON matching the requested schema.
5. If you don't have enough information to answer, say so honestly.
"""

SYSTEM_PROMPT_RECOMMENDATIONS = SYSTEM_PROMPT_BASE + """
You are helping the user select the best clauses and variants for their contract.
Your recommendations are based on data from the knowledge graph — you explain and prioritize them.
You do NOT invent new recommendations. You only explain what the graph provides.
"""

SYSTEM_PROMPT_CUSTOMIZATION = SYSTEM_PROMPT_BASE + """
You are helping the user customize a legal clause.
CRITICAL: You MUST preserve ALL {{PLACEHOLDER}} tokens in the clause text. 
Never remove, rename, or alter any {{PLACEHOLDER}}.
Your customization must maintain legal validity and enforceability.
"""

SYSTEM_PROMPT_RISK = SYSTEM_PROMPT_BASE + """
You are analyzing the risk profile of a contract.
Risk scores, conflict data, and gap analysis come from the knowledge graph.
You explain these risks in plain language and suggest actionable mitigations.
Do NOT invent new risks — only explain what the graph provides.
"""

SYSTEM_PROMPT_CHATBOT = SYSTEM_PROMPT_BASE + """
You are answering questions about the user's specific contract.
Always quote relevant clause text in your answers and cite clause IDs.
If a question cannot be answered from the provided context, say so clearly.
Be concise but thorough. Use bullet points for clarity.
"""


# ==================== PROMPT TEMPLATES ====================

RECOMMENDATION_PROMPT = """
## Contract Context
- Contract Type: {contract_type}
- Jurisdiction: {jurisdiction}
- Current active clauses: {active_clauses}

## Graph-Derived Recommendations

### Alternative Clauses (from ALTERNATIVE_TO relationships):
{alternatives_data}

### Missing Required Clause Types (from REQUIRES relationships):
{requires_data}

### Unselected Optional Clause Types:
{optional_gaps}

## Your Task
Based on the above graph data, create a prioritized recommendation summary:
1. List the TOP recommendations (most impactful first)
2. For each, explain WHY using the graph-provided reason/benefit
3. Classify each as: "variant_upgrade" | "missing_clause" | "optional_addition"
4. Do NOT invent recommendations beyond what the graph provides

Respond in this JSON format:
{{
  "recommendations": [
    {{
      "type": "variant_upgrade" | "missing_clause" | "optional_addition",
      "clause_type": "clause type id",
      "current_clause_id": "current clause id or null",
      "recommended_clause_id": "recommended clause id",
      "title": "short title",
      "reason": "from graph data",
      "benefit": "from graph data", 
      "priority": "high" | "medium" | "low",
      "recommendation_strength": "from graph data"
    }}
  ],
  "summary": "1-2 sentence overall summary"
}}
"""

CUSTOMIZATION_PROMPT = """
## Clause to Customize
- Clause ID: {clause_id}
- Clause Type: {clause_type}
- Current Variant: {variant}
- Risk Level: {risk_level}

### Current Clause Text:
{clause_text}

### All Available Variants for Reference:
{all_variants}

### Parameters Used in This Clause:
{parameters}

## User's Customization Request:
"{user_instruction}"

## Your Task
Customize the clause text based on the user's request. 

CRITICAL RULES:
1. PRESERVE all {{{{PLACEHOLDER}}}} tokens exactly as they appear — do not remove, rename, or change any
2. Maintain legal enforceability
3. Keep the customization consistent with Indian contract law (or the relevant jurisdiction)
4. Explain what you changed and why

Respond in this JSON format:
{{
  "customized_text": "the full customized clause text with all placeholders preserved",
  "changes_summary": "brief description of what was changed",
  "risk_impact": "lower" | "same" | "higher",
  "risk_explanation": "why the risk changed or stayed the same",
  "preserved_placeholders": ["list of all {{{{PLACEHOLDERS}}}} in the customized text"],
  "legal_notes": "any legal considerations about this customization"
}}
"""

RISK_ANALYSIS_PROMPT = """
## Contract Profile
- Contract Type: {contract_type} ({contract_description})
- Jurisdiction: {jurisdiction}
- Total Active Clauses: {total_clauses}

## Clause Risk Levels (from Knowledge Graph):
{clause_risks}

## Conflicts Detected (from CONFLICTS_WITH relationships):
{conflicts}

## Missing Dependencies (from REQUIRES relationships):
{missing_dependencies}

## Gap Analysis (clause types available but not selected):
{gaps}

## Your Task
Provide a comprehensive risk analysis. Remember: all risk data comes from the graph.
Explain each risk clearly and suggest specific, actionable mitigations.

Respond in this JSON format:
{{
  "overall_risk_score": <weighted average from clause risk_levels, 1-10>,
  "overall_risk_label": "Low" | "Medium" | "High" | "Critical",
  "summary": "2-3 sentence executive summary of the contract's risk profile",
  "clause_risks": [
    {{
      "clause_id": "id",
      "clause_type": "type",
      "risk_level": <from graph>,
      "explanation": "what this risk level means in plain language",
      "mitigation": "specific action to reduce risk"
    }}
  ],
  "conflicts": [
    {{
      "clause_a": "id",
      "clause_b": "id",
      "severity": "from graph",
      "description": "plain language explanation",
      "resolution": "from graph resolution_advice"
    }}
  ],
  "gaps": [
    {{
      "missing_clause_type": "type",
      "reason": "why it's needed",
      "impact": "what happens without it",
      "is_critical": true/false
    }}
  ],
  "action_items": [
    "ordered list of recommended actions, most important first"
  ]
}}
"""

CHATBOT_PROMPT = """
## Contract Context
- Contract Type: {contract_type}
- Jurisdiction: {jurisdiction}
- Contract Status: {contract_status}

## Relevant Clause Data (from Knowledge Graph):
{relevant_clauses}

## Relevant Parameters:
{relevant_parameters}

## Chat History:
{chat_history}

## User's Question:
"{user_message}"

## Your Task
Answer the user's question using ONLY the provided context.
- Quote specific clause text when relevant
- Cite clause IDs (e.g., "According to clause CONF_MOD_001...")
- Be concise and professional
- If you cannot answer from the context, say so clearly

Respond in this JSON format:
{{
  "answer": "your response in markdown format",
  "citations": [
    {{
      "clause_id": "id",
      "clause_type": "type",
      "relevant_snippet": "short quote from the clause"
    }}
  ],
  "follow_up_suggestions": [
    "1-2 suggested follow-up questions the user might want to ask"
  ]
}}
"""
