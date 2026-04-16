#!/usr/bin/env python3
"""
experiment.py - LegalWiz Three-Condition Hallucination Evaluation
=================================================================
Measures hallucination rates, risk score consistency, placeholder
preservation, and latency across four experimental conditions:

  C1  — LLM Only (no graph context)
  C2  — Standard Vector RAG (cosine similarity, sentence-transformers)
  C3  — LegalWiz Full System without grounding validator
  C3v — LegalWiz Full System with grounding validator (production baseline)

Run:
    cd /path/to/legalwiz-clm-system/routes
    python eval/experiment.py

Outputs:
    ../results/eval_raw.json
    ../results/eval_summary.txt
"""

import sys
import os

# ---- macOS OpenMP / BLAS deadlock fix — MUST be set before torch/numpy import ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# -----------------------------------------------------------------------------------

import json
import re
import time
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. PATH BOOTSTRAP — must happen before any local-module imports
# ---------------------------------------------------------------------------
# The script lives in  routes/eval/experiment.py.
# All the project modules (config, graph_rag_engine, llm_config …) are in
# routes/, so we add that directory to sys.path.
SCRIPT_DIR   = Path(__file__).resolve().parent          # routes/eval/
ROUTES_DIR   = SCRIPT_DIR.parent                        # routes/
PROJECT_ROOT = ROUTES_DIR.parent                        # legalwiz-clm-system/
RESULTS_DIR  = PROJECT_ROOT / "results"

sys.path.insert(0, str(ROUTES_DIR))

# Load .env BEFORE importing config (which also calls load_dotenv, but be safe)
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# 1. LOCAL IMPORTS (after path fix)
# ---------------------------------------------------------------------------
from config import get_neo4j_driver, get_connection          # DB helpers
from graph_rag_engine import (
    GraphRAGRetriever, LLMClient, GroundingValidator,
    _get_shared_driver,
)
from llm_config import LLM_CONFIG
import copy

# ---------------------------------------------------------------------------
# EVAL-SPECIFIC LLM CONFIG
# Force a lighter model to conserve API quota:
#   • Groq  → llama-3.1-8b-instant  (instead of 70b)
#   • Gemini → gemini-2.0-flash      (unchanged, already lightweight)
# ---------------------------------------------------------------------------
EVAL_LLM_CONFIG = copy.deepcopy(LLM_CONFIG)
if EVAL_LLM_CONFIG.get("provider") == "groq":
    EVAL_LLM_CONFIG["model"] = "llama-3.1-8b-instant"
    log_msg = "Groq provider detected — using llama-3.1-8b-instant to save quota."
else:
    log_msg = f"LLM provider: {EVAL_LLM_CONFIG.get('provider')} / model: {EVAL_LLM_CONFIG.get('model')}"

# ---------------------------------------------------------------------------
# 2. LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval")

# ---------------------------------------------------------------------------
# 3. C2 VECTOR RAG — use sklearn TF-IDF cosine similarity
#    (avoids the macOS ARM PyTorch mmap deadlock from sentence-transformers)
# ---------------------------------------------------------------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    log.warning(
        "scikit-learn not installed — C2 (Vector RAG) will be skipped.\n"
        "Install with:  pip install scikit-learn"
    )

# compat alias so the rest of the code works unchanged
_ST_AVAILABLE = _SKLEARN_AVAILABLE

# ---------------------------------------------------------------------------
# 4. CONSTANTS
# ---------------------------------------------------------------------------
JURISDICTION = "India"
CUSTOMIZATION_INSTRUCTION = (
    "Make this clause more protective for the drafting party "
    "while maintaining enforceability under Indian Contract Act 1872."
)
QA_QUESTIONS = [
    "What are the confidentiality obligations in this contract?",
    "What happens if either party breaches the payment terms?",
    "What is the governing law and dispute resolution mechanism?",
]
RATE_LIMIT_SLEEP = 3          # seconds between LLM calls (conservative for free-tier)
CLAUSE_TEXT_PREVIEW_LEN = 600 # chars passed to C2 LLM per clause


# ---------------------------------------------------------------------------
# 5. TEST CONTRACT DEFINITIONS (synthetic — never inserted into real DB)
# ---------------------------------------------------------------------------
# We only define the contract_type and how many active clauses to pick.
# Actual clause IDs are fetched live from Neo4j so they are always valid.

TEST_CONTRACT_SPECS = [
    {"contract_id": "test_contract_1", "contract_type": "employment_nda",            "n_clauses": 5},
    {"contract_id": "test_contract_2", "contract_type": "saas_service_agreement",    "n_clauses": 8},
    {"contract_id": "test_contract_3", "contract_type": "consulting_service_agreement","n_clauses": 7},
    {"contract_id": "test_contract_4", "contract_type": "vendor_agreement",           "n_clauses": 10},
    {"contract_id": "test_contract_5", "contract_type": "partnership_agreement",      "n_clauses": 6},
]

# ---------------------------------------------------------------------------
# 6. NEO4J HELPERS
# ---------------------------------------------------------------------------

def neo4j_id(contract_type: str) -> str:
    """Convert underscore contract type to Neo4j dash-style id."""
    return contract_type.replace("_", "-")


def fetch_clause_ids_for_contract_type(
    contract_type: str, jurisdiction: str, n: int
) -> List[str]:
    """
    Query Neo4j for the first N clause IDs of a given contract type,
    preferring the Moderate variant where multiple variants exist.
    Returns empty list if the contract type is not found.
    """
    driver = _get_shared_driver()
    ct_id = neo4j_id(contract_type)
    with driver.session() as session:
        result = session.run(
            """
            MATCH (ct:ContractType {id: $ct_id})-[:CONTAINS_CLAUSE]->(ctype:ClauseType)
                  -[:HAS_VARIANT]->(c:Clause)
            WHERE c.jurisdiction = $jurisdiction
            WITH ctype, c
            ORDER BY ctype.id, c.variant
            WITH ctype, collect(c) AS variants
            WITH ctype,
                 [v IN variants WHERE v.variant = 'Moderate' | v][0] AS preferred,
                 variants[0] AS fallback
            WITH ctype, COALESCE(preferred, fallback) AS chosen
            ORDER BY ctype.id
            RETURN chosen.id AS clause_id
            LIMIT $n
            """,
            {"ct_id": ct_id, "jurisdiction": jurisdiction, "n": n},
        )
        ids = [r["clause_id"] for r in result if r["clause_id"]]
    if not ids:
        log.warning(
            "No clauses found for contract_type=%s jurisdiction=%s "
            "— will use partial/empty set.",
            contract_type, jurisdiction,
        )
    return ids


def fetch_all_clauses_from_neo4j() -> List[Dict]:
    """Fetch ALL clause texts from Neo4j (used by C2 to build the corpus)."""
    driver = _get_shared_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Clause)
            RETURN c.id AS clause_id, c.raw_text AS raw_text,
                   c.risk_level AS risk_level, c.clause_type AS clause_type,
                   c.variant AS variant
            """
        )
        return [dict(r) for r in result]


def get_neo4j_risk_level(clause_id: str) -> Optional[float]:
    """Return the stored risk_level for a single clause from Neo4j."""
    driver = _get_shared_driver()
    with driver.session() as session:
        result = session.run(
            "MATCH (c:Clause {id: $id}) RETURN c.risk_level AS rl",
            {"id": clause_id},
        )
        row = result.single()
        return row["rl"] if row else None


def get_clause_raw_text(clause_id: str) -> Optional[str]:
    """Return raw_text for a clause from Neo4j."""
    driver = _get_shared_driver()
    with driver.session() as session:
        result = session.run(
            "MATCH (c:Clause {id: $id}) RETURN c.raw_text AS rt",
            {"id": clause_id},
        )
        row = result.single()
        return row["rt"] if row else None


def verify_clause_ids_batch(clause_ids: List[str]) -> Dict[str, bool]:
    """Return {clause_id: exists_in_neo4j} for a list of IDs."""
    if not clause_ids:
        return {}
    driver = _get_shared_driver()
    with driver.session() as session:
        result = session.run(
            """
            UNWIND $ids AS check_id
            OPTIONAL MATCH (c:Clause {id: check_id})
            RETURN check_id AS id, c IS NOT NULL AS exists
            """,
            {"ids": clause_ids},
        )
        return {r["id"]: r["exists"] for r in result}


# ---------------------------------------------------------------------------
# 7. METRIC HELPERS
# ---------------------------------------------------------------------------

def extract_clause_ids_from_response(response: Any) -> List[str]:
    """
    Find all clause_id values in an LLM response dict (nested structure).
    We look for keys named exactly 'clause_id', 'recommended_clause_id',
    'current_clause_id', 'clause_a', 'clause_b' and also scan plain text
    output with a regex for typical ID patterns like CONF_MOD_001.
    """
    ids: List[str] = []

    def _recurse(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("clause_id", "recommended_clause_id",
                         "current_clause_id", "clause_a", "clause_b"):
                    if isinstance(v, str) and v:
                        ids.append(v)
                else:
                    _recurse(v)
        elif isinstance(obj, list):
            for item in obj:
                _recurse(item)
        elif isinstance(obj, str):
            # Regex sweep for known ID patterns: WORD_WORD_DDD
            found = re.findall(r'\b[A-Z][A-Z0-9_]{2,}[-_][A-Z0-9_]{2,}(?:[-_][A-Z0-9]+)*\b', obj)
            ids.extend(found)

    _recurse(response)
    # Deduplicate but preserve order
    seen, unique = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    return unique


def extract_risk_levels_from_response(response: Any) -> List[Dict]:
    """
    Extract all {clause_id, risk_level} pairs from LLM response.
    Handles both float and string risk levels.
    """
    pairs: List[Dict] = []

    def _recurse(obj):
        if isinstance(obj, dict):
            if "clause_id" in obj and "risk_level" in obj:
                cid = obj.get("clause_id")
                rl  = obj.get("risk_level")
                if cid and rl is not None:
                    try:
                        pairs.append({"clause_id": cid, "risk_level": float(rl)})
                    except (TypeError, ValueError):
                        pass
            for v in obj.values():
                _recurse(v)
        elif isinstance(obj, list):
            for item in obj:
                _recurse(item)

    _recurse(response)
    return pairs


def extract_citations_from_response(response: Any) -> List[Dict]:
    """Return list of citation dicts from chatbot-style response."""
    if isinstance(response, dict):
        return response.get("citations", [])
    return []


def compute_hallucination_rate(cited_ids: List[str]) -> Tuple[float, List[str], List[str]]:
    """
    Check cited IDs against Neo4j.
    Returns (hallucination_rate, invalid_ids, valid_ids).
    """
    if not cited_ids:
        return 0.0, [], []
    existence = verify_clause_ids_batch(cited_ids)
    invalid = [i for i, ex in existence.items() if not ex]
    valid   = [i for i, ex in existence.items() if ex]
    rate    = len(invalid) / len(cited_ids) if cited_ids else 0.0
    return rate, invalid, valid


def compute_placeholder_preservation(
    original_text: str, customized_text: str
) -> float:
    """Return preservation_rate = preserved / original (1.0 if no placeholders)."""
    orig = set(re.findall(r'\{\{[A-Z_0-9]+\}\}', original_text))
    cust = set(re.findall(r'\{\{[A-Z_0-9]+\}\}', customized_text))
    if not orig:
        return 1.0
    preserved = orig & cust
    return len(preserved) / len(orig)


def safe_sleep(secs: float = RATE_LIMIT_SLEEP):
    """Sleep to respect LLM rate limits."""
    time.sleep(secs)


# ---------------------------------------------------------------------------
# 8. TEST CONTRACT BUILDER
# ---------------------------------------------------------------------------

def build_test_contracts() -> List[Dict]:
    """
    Build synthetic test contract dicts by fetching real clause IDs from Neo4j.
    Nothing is written to Postgres — these are in-memory only.
    """
    contracts = []
    for spec in TEST_CONTRACT_SPECS:
        clause_ids = fetch_clause_ids_for_contract_type(
            spec["contract_type"], JURISDICTION, spec["n_clauses"]
        )
        if not clause_ids:
            log.warning(
                "Contract type '%s' returned 0 clause IDs — "
                "fetching any available clauses as fallback.",
                spec["contract_type"],
            )
            # Broad fallback: any clause in Neo4j
            driver = _get_shared_driver()
            with driver.session() as session:
                res = session.run(
                    "MATCH (c:Clause) RETURN c.id AS cid LIMIT $n",
                    {"n": spec["n_clauses"]},
                )
                clause_ids = [r["cid"] for r in res]

        contracts.append({
            "contract_id":   spec["contract_id"],
            "contract_type": spec["contract_type"],
            "jurisdiction":  JURISDICTION,
            "active_clause_ids": clause_ids,
            "title": f"Test Contract — {spec['contract_type'].replace('_', ' ').title()}",
            "status": "draft",
        })
        log.info(
            "  Built %s  → %d clauses",
            spec["contract_id"], len(clause_ids),
        )
    return contracts


# ---------------------------------------------------------------------------
# 9. LLM CALL WRAPPER (with retry + rate-limit sleep)
# ---------------------------------------------------------------------------

def llm_call(
    llm: LLMClient,
    prompt: str,
    system_prompt: str = "",
    task_label: str = "",
) -> Tuple[Optional[Dict], float]:
    """
    Call the LLM, sleep for rate limits, return (response_dict, latency_ms).
    On failure, returns (None, latency_ms).
    """
    t0 = time.perf_counter()
    try:
        response = llm.generate(prompt, system_prompt)
        latency  = (time.perf_counter() - t0) * 1000
        safe_sleep()
        return response, latency
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        log.error("  LLM call failed [%s]: %s", task_label, exc)
        # Retry once
        safe_sleep(4)
        try:
            t0 = time.perf_counter()
            response = llm.generate(prompt, system_prompt)
            latency  = (time.perf_counter() - t0) * 1000
            safe_sleep()
            return response, latency
        except Exception as exc2:
            latency = (time.perf_counter() - t0) * 1000
            log.error("  LLM retry failed [%s]: %s", task_label, exc2)
            safe_sleep()
            return None, latency


# ---------------------------------------------------------------------------
# 10. C1 — LLM ONLY (no graph data)
# ---------------------------------------------------------------------------

C1_SYS = (
    "You are a legal AI assistant. Answer only in valid JSON. "
    "Do not invent clause IDs. If you don't have enough information, say so."
)


def c1_recommendations(llm: LLMClient, contract: Dict) -> Tuple[Optional[Dict], float]:
    prompt = f"""
You are advising on a {contract['contract_type']} contract governed by {contract['jurisdiction']} law.
Recommend improvements to the contract clauses. You do not have access to specific clause IDs.

Respond in JSON:
{{
  "recommendations": [
    {{
      "type": "variant_upgrade" | "missing_clause" | "optional_addition",
      "clause_type": "clause type name",
      "current_clause_id": null,
      "recommended_clause_id": null,
      "title": "short title",
      "reason": "reason for recommendation",
      "benefit": "expected benefit",
      "priority": "high" | "medium" | "low"
    }}
  ],
  "summary": "1-2 sentence overall summary"
}}
"""
    return llm_call(llm, prompt, C1_SYS, "C1/recommendations")


def c1_risk(llm: LLMClient, contract: Dict) -> Tuple[Optional[Dict], float]:
    prompt = f"""
You are analysing the risk profile of a {contract['contract_type']} contract 
governed by {contract['jurisdiction']} law.

Respond in JSON:
{{
  "overall_risk_score": <number 1-10>,
  "overall_risk_label": "Low" | "Medium" | "High" | "Critical",
  "summary": "executive summary",
  "clause_risks": [
    {{
      "clause_id": null,
      "clause_type": "clause type name",
      "risk_level": <number>,
      "explanation": "plain language explanation",
      "mitigation": "suggestion"
    }}
  ],
  "conflicts": [],
  "gaps": [],
  "action_items": ["ordered list of actions"]
}}
"""
    return llm_call(llm, prompt, C1_SYS, "C1/risk")


def c1_customization(
    llm: LLMClient, clause_id: str, raw_text: str
) -> Tuple[Optional[Dict], float]:
    prompt = f"""
Customize the following legal clause text. Preserve ALL {{{{PLACEHOLDER}}}} tokens exactly.

Clause text:
{raw_text}

Instruction: {CUSTOMIZATION_INSTRUCTION}

Respond in JSON:
{{
  "customized_text": "full customized text with all placeholders preserved",
  "changes_summary": "what was changed",
  "risk_impact": "lower" | "same" | "higher",
  "risk_explanation": "why risk changed",
  "preserved_placeholders": ["list of {{{{PLACEHOLDER}}}} tokens"]
}}
"""
    return llm_call(llm, prompt, C1_SYS, f"C1/customization/{clause_id}")


def c1_qa(
    llm: LLMClient, contract: Dict, question: str
) -> Tuple[Optional[Dict], float]:
    prompt = f"""
Answer the following question about a {contract['contract_type']} contract 
under {contract['jurisdiction']} law. You have no specific clause text available.

Question: {question}

Respond in JSON:
{{
  "answer": "your answer",
  "citations": [],
  "answerable": true | false
}}
"""
    return llm_call(llm, prompt, C1_SYS, f"C1/qa")


# ---------------------------------------------------------------------------
# 11. C2 — VECTOR RAG (sentence-transformers)
# ---------------------------------------------------------------------------

class VectorRAGCorpus:
    """
    Pure-Python TF-IDF cosine similarity retriever over all Neo4j clause texts.
    No PyTorch / GPU dependency — works on macOS ARM without deadlocks.
    """

    def __init__(self, clauses: List[Dict]):
        self.clauses = clauses
        texts = [c.get("raw_text", "") or "" for c in clauses]
        log.info("  [C2] Building TF-IDF index over %d clause texts…", len(clauses))
        self.vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, 2),
            max_features=20_000,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(texts)  # sparse (n_clauses, vocab)
        log.info("  [C2] TF-IDF index ready (vocab=%d).", len(self.vectorizer.vocabulary_))

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Return top-k clauses by TF-IDF cosine similarity."""
        q_vec = self.vectorizer.transform([query])           # sparse (1, vocab)
        sims  = cosine_similarity(q_vec, self.matrix)[0]    # dense (n_clauses,)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [self.clauses[i] for i in top_idx]


C2_SYS = C1_SYS


def _format_c2_context(retrieved: List[Dict]) -> str:
    lines = []
    for c in retrieved:
        text = (c.get("raw_text") or "")[:CLAUSE_TEXT_PREVIEW_LEN]
        lines.append(
            f"[Clause ID: {c['clause_id']} | Type: {c['clause_type']} | "
            f"Variant: {c['variant']} | Risk: {c['risk_level']}]\n{text}"
        )
    return "\n\n".join(lines)


def c2_recommendations(
    llm: LLMClient, corpus: "VectorRAGCorpus", contract: Dict
) -> Tuple[Optional[Dict], float]:
    query = f"clause recommendations for {contract['contract_type']} {contract['jurisdiction']}"
    ctx   = _format_c2_context(corpus.retrieve(query))
    prompt = f"""
You are advising on a {contract['contract_type']} contract under {contract['jurisdiction']} law.

Below are the most relevant clause options retrieved from the knowledge base:

{ctx}

Recommend improvements. Only cite clause IDs from the context above.

Respond in JSON:
{{
  "recommendations": [
    {{
      "type": "variant_upgrade" | "missing_clause" | "optional_addition",
      "clause_type": "clause type name",
      "current_clause_id": null,
      "recommended_clause_id": "clause_id from context or null",
      "title": "short title",
      "reason": "reason",
      "benefit": "benefit",
      "priority": "high" | "medium" | "low"
    }}
  ],
  "summary": "1-2 sentence summary"
}}
"""
    return llm_call(llm, prompt, C2_SYS, "C2/recommendations")


def c2_risk(
    llm: LLMClient, corpus: "VectorRAGCorpus", contract: Dict
) -> Tuple[Optional[Dict], float]:
    query = f"risk analysis {contract['contract_type']} {contract['jurisdiction']}"
    ctx   = _format_c2_context(corpus.retrieve(query))
    prompt = f"""
Analyse the risk profile of a {contract['contract_type']} contract under {contract['jurisdiction']} law.

Relevant clauses:
{ctx}

Only cite clause IDs from the context above.

Respond in JSON:
{{
  "overall_risk_score": <number 1-10>,
  "overall_risk_label": "Low" | "Medium" | "High" | "Critical",
  "summary": "executive summary",
  "clause_risks": [
    {{
      "clause_id": "id from context or null",
      "clause_type": "type",
      "risk_level": <number>,
      "explanation": "explanation",
      "mitigation": "suggestion"
    }}
  ],
  "conflicts": [],
  "gaps": [],
  "action_items": ["actions"]
}}
"""
    return llm_call(llm, prompt, C2_SYS, "C2/risk")


def c2_customization(
    llm: LLMClient, corpus: "VectorRAGCorpus",
    clause_id: str, raw_text: str
) -> Tuple[Optional[Dict], float]:
    ctx   = _format_c2_context(corpus.retrieve(raw_text[:200]))
    prompt = f"""
Customize the following clause. Preserve ALL {{{{PLACEHOLDER}}}} tokens.

Clause text:
{raw_text}

Similar clauses for reference:
{ctx}

Instruction: {CUSTOMIZATION_INSTRUCTION}

Respond in JSON:
{{
  "customized_text": "full customized text",
  "changes_summary": "what was changed",
  "risk_impact": "lower" | "same" | "higher",
  "risk_explanation": "explanation",
  "preserved_placeholders": ["list of {{{{PLACEHOLDER}}}}"]
}}
"""
    return llm_call(llm, prompt, C2_SYS, f"C2/customization/{clause_id}")


def c2_qa(
    llm: LLMClient, corpus: "VectorRAGCorpus",
    contract: Dict, question: str
) -> Tuple[Optional[Dict], float]:
    ctx   = _format_c2_context(corpus.retrieve(question))
    prompt = f"""
Answer the following question about a {contract['contract_type']} contract 
under {contract['jurisdiction']} law using the context below.

Context:
{ctx}

Question: {question}

Only cite clause IDs from the context above.

Respond in JSON:
{{
  "answer": "your answer",
  "citations": [
    {{
      "clause_id": "id from context",
      "clause_type": "type",
      "relevant_snippet": "short quote"
    }}
  ],
  "answerable": true | false
}}
"""
    return llm_call(llm, prompt, C2_SYS, "C2/qa")


# ---------------------------------------------------------------------------
# 12. C3 — LEGALWIZ FULL SYSTEM (with / without validator)
# ---------------------------------------------------------------------------
from llm_config import (
    SYSTEM_PROMPT_RECOMMENDATIONS, RECOMMENDATION_PROMPT,
    SYSTEM_PROMPT_RISK,           RISK_ANALYSIS_PROMPT,
    SYSTEM_PROMPT_CUSTOMIZATION,  CUSTOMIZATION_PROMPT,
    SYSTEM_PROMPT_CHATBOT,        CHATBOT_PROMPT,
)


def _format_alternatives(alternatives):
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


def _format_requires(requires):
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


def _format_optional_gaps(gaps):
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


def _format_clause_risks(risks):
    lines = []
    for r in risks:
        lines.append(
            f"- {r['clause_id']} ({r.get('clause_type_name', r.get('clause_type', 'Unknown'))}): "
            f"variant={r['variant']}, risk_level={r['risk_level']}/10, "
            f"importance={r.get('importance_level', 'Medium')}, "
            f"category={r.get('category', 'General')}"
        )
    return "\n".join(lines) or "No clause risks found."


def _format_conflicts(conflicts):
    if not conflicts:
        return "No conflicts detected between active clauses."
    lines = []
    for c in conflicts:
        lines.append(
            f"- CONFLICT: {c['clause_a_id']} ({c['clause_a_type']}/{c['clause_a_variant']}) "
            f"↔ {c['clause_b_id']} ({c['clause_b_type']}/{c['clause_b_variant']})\n"
            f"  Severity: {c['severity']}\n"
            f"  Reason: {c['reason']}\n"
            f"  Resolution: {c['resolution_advice']}"
        )
    return "\n".join(lines)


def _format_missing_deps(deps):
    if not deps:
        return "No missing dependencies detected."
    lines = []
    for d in deps:
        critical = "CRITICAL" if d["is_critical"] else "recommended"
        lines.append(
            f"- {d['source_name']} REQUIRES {d['missing_name']} ({critical})\n"
            f"  Reason: {d['reason']}"
        )
    return "\n".join(lines)


def _format_gaps(gaps):
    if not gaps:
        return "No clause type gaps."
    lines = []
    for g in gaps:
        lines.append(
            f"- Missing: {g['clause_type_name']} "
            f"(importance: {g['importance_level']}, {g.get('description', '')})"
        )
    return "\n".join(lines)


def _format_clause_context(clauses):
    lines = []
    for c in clauses:
        text = (c.get("raw_text") or "")[:500]
        lines.append(
            f"### {c.get('clause_type_name', c.get('clause_type', 'Unknown'))} "
            f"({c['variant']} variant, risk: {c['risk_level']})\n"
            f"**Clause ID:** {c['clause_id']}\n"
            f"**Text:**\n{text}\n"
        )
    return "\n".join(lines)


def _format_variants(variants):
    lines = []
    for v in variants:
        text_preview = v["raw_text"][:300] + "..." if len(v.get("raw_text", "")) > 300 else v.get("raw_text", "")
        lines.append(
            f"### {v['variant']} Variant (risk: {v['risk_level']})\n"
            f"ID: {v['clause_id']}\n"
            f"Text:\n{text_preview}\n"
        )
    return "\n".join(lines)


def _format_parameters(params):
    if not params:
        return "No parameters in this clause."
    lines = []
    for p in params:
        req = "REQUIRED" if p["is_required"] else "optional"
        lines.append(f"- {p['parameter_name']} (type: {p['data_type']}, {req})")
    return "\n".join(lines)


def c3_recommendations(
    llm: LLMClient, retriever: GraphRAGRetriever,
    val: Optional[GroundingValidator], contract: Dict
) -> Tuple[Optional[Dict], float]:
    ctx = retriever.get_recommendation_context(
        contract["contract_type"], contract["jurisdiction"],
        contract["active_clause_ids"],
    )
    # Build the prompt exactly as production does
    active_clauses_str = ", ".join(contract["active_clause_ids"])
    prompt = RECOMMENDATION_PROMPT.format(
        contract_type=contract["contract_type"],
        jurisdiction=contract["jurisdiction"],
        active_clauses=active_clauses_str,
        alternatives_data=_format_alternatives(ctx["alternatives"]),
        requires_data=_format_requires(ctx["requires"]),
        optional_gaps=_format_optional_gaps(ctx["optional_gaps"]),
    )
    response, latency = llm_call(llm, prompt, SYSTEM_PROMPT_RECOMMENDATIONS, "C3/recommendations")
    if response and val:
        recs = response.get("recommendations", [])
        grounding = val.validate_recommendations(recs, ctx)
        if not grounding["valid"]:
            # Remove hallucinated recommendations (as production does)
            ungrounded_ids = {id(r) for r in grounding.get("ungrounded_recommendations", [])}
            response["recommendations"] = [r for r in recs if id(r) not in ungrounded_ids]
        response["_grounding"] = grounding
    return response, latency


def c3_risk(
    llm: LLMClient, retriever: GraphRAGRetriever,
    val: Optional[GroundingValidator], contract: Dict
) -> Tuple[Optional[Dict], float]:
    ctx = retriever.get_risk_context(
        contract["contract_type"], contract["jurisdiction"],
        contract["active_clause_ids"],
    )
    ct_info = ctx.get("contract_type_info", {})
    prompt = RISK_ANALYSIS_PROMPT.format(
        contract_type=contract["contract_type"],
        contract_description=ct_info.get("description", ""),
        jurisdiction=contract["jurisdiction"],
        total_clauses=len(contract["active_clause_ids"]),
        clause_risks=_format_clause_risks(ctx["clause_risks"]),
        conflicts=_format_conflicts(ctx["conflicts"]),
        missing_dependencies=_format_missing_deps(ctx["missing_dependencies"]),
        gaps=_format_gaps(ctx["gaps"]),
    )
    response, latency = llm_call(llm, prompt, SYSTEM_PROMPT_RISK, "C3/risk")
    if response and val:
        llm_risks = response.get("clause_risks", [])
        risk_validation = val.validate_risk_scores(llm_risks, ctx["clause_risks"])
        # Correct mismatched risk scores to graph truth
        graph_risk_map = {r["clause_id"]: r["risk_level"] for r in ctx["clause_risks"]}
        for lr in llm_risks:
            cid = lr.get("clause_id")
            if cid in graph_risk_map:
                lr["risk_level"] = graph_risk_map[cid]
        response["_validation"] = risk_validation
    # Attach graph_risks so we can compute RSC accurately later
    if response:
        response["_graph_clause_risks"] = ctx.get("clause_risks", [])
    return response, latency


def c3_customization(
    llm: LLMClient, retriever: GraphRAGRetriever,
    val: Optional[GroundingValidator],
    clause_id: str,
) -> Tuple[Optional[Dict], float]:
    ctx = retriever.get_customization_context(clause_id)
    if not ctx:
        return None, 0.0
    clause_data = ctx["clause"]
    raw_text = clause_data.get("raw_text", "")
    prompt = CUSTOMIZATION_PROMPT.format(
        clause_id=clause_id,
        clause_type=clause_data.get("clause_type", ""),
        variant=clause_data.get("variant", ""),
        risk_level=clause_data.get("risk_level", ""),
        clause_text=raw_text,
        all_variants=_format_variants(ctx["all_variants"]),
        parameters=_format_parameters(ctx["parameters"]),
        user_instruction=CUSTOMIZATION_INSTRUCTION,
    )
    response, latency = llm_call(llm, prompt, SYSTEM_PROMPT_CUSTOMIZATION, f"C3/customization/{clause_id}")
    if response and val:
        ctext = response.get("customized_text", "")
        ph_check = val.validate_placeholders(raw_text, ctext)
        response["_placeholder_check"] = ph_check
    elif response:
        ctext = response.get("customized_text", "")
        # Manual placeholder check (same logic)
        orig = set(re.findall(r'\{\{[A-Z_0-9]+\}\}', raw_text))
        cust = set(re.findall(r'\{\{[A-Z_0-9]+\}\}', ctext))
        missing = orig - cust
        ppr = len(orig & cust) / len(orig) if orig else 1.0
        response["_placeholder_check"] = {
            "valid": len(missing) == 0,
            "preservation_rate": ppr,
            "missing_placeholders": sorted(list(missing)),
            "original_placeholders": sorted(list(orig)),
        }
    # Always attach original raw_text for manual PPR computation
    if response:
        response["_raw_text"] = raw_text
    return response, latency


def c3_qa(
    llm: LLMClient, retriever: GraphRAGRetriever,
    val: Optional[GroundingValidator],
    contract: Dict, question: str,
) -> Tuple[Optional[Dict], float]:
    # Build a minimal fake contract_id context using active clause IDs
    # (we don't have a real pg contract_id, so we skip parameter values)
    driver = _get_shared_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Clause)
            WHERE c.id IN $ids
            OPTIONAL MATCH (ct:ClauseType)-[:HAS_VARIANT]->(c)
            RETURN c.id AS clause_id, c.clause_type AS clause_type,
                   c.variant AS variant, c.risk_level AS risk_level,
                   c.raw_text AS raw_text, ct.name AS clause_type_name
            """,
            {"ids": contract["active_clause_ids"]},
        )
        clauses = [dict(r) for r in result]

    prompt = CHATBOT_PROMPT.format(
        contract_type=contract["contract_type"],
        jurisdiction=contract["jurisdiction"],
        contract_status=contract.get("status", "draft"),
        relevant_clauses=_format_clause_context(clauses),
        relevant_parameters="No parameter values set yet.",
        chat_history="No previous conversation.",
        user_message=question,
    )
    response, latency = llm_call(llm, prompt, SYSTEM_PROMPT_CHATBOT, "C3/qa")
    if response and val:
        citations = response.get("citations", [])
        cit_check = val.validate_citations(citations, clauses)
        response["_citation_check"] = cit_check
    return response, latency


# ---------------------------------------------------------------------------
# 13. SINGLE-TASK METRIC COMPUTATION
# ---------------------------------------------------------------------------

def measure_recommendations(response: Optional[Dict], condition: str) -> Dict:
    if response is None:
        return {
            "hallucination_rate": None, "cited_ids": [], "invalid_ids": [],
            "risk_score_consistency": None, "placeholder_preservation_rate": None,
        }
    cited = extract_clause_ids_from_response(response)
    hr, invalid, valid = compute_hallucination_rate(cited)
    return {
        "hallucination_rate": hr,
        "cited_ids": cited,
        "invalid_ids": invalid,
        "risk_score_consistency": None,
        "placeholder_preservation_rate": None,
    }


def measure_risk(response: Optional[Dict], contract: Dict) -> Dict:
    if response is None:
        return {
            "hallucination_rate": None, "cited_ids": [], "invalid_ids": [],
            "risk_score_consistency": None, "placeholder_preservation_rate": None,
        }
    cited = extract_clause_ids_from_response(response)
    hr, invalid, _ = compute_hallucination_rate(cited)

    # RSC: compare LLM-reported risk_level against Neo4j stored value
    risk_pairs = extract_risk_levels_from_response(response)
    # Filter out pairs where clause_id is None
    risk_pairs = [p for p in risk_pairs if p.get("clause_id")]
    total_reported = len(risk_pairs)
    matching = 0
    for pair in risk_pairs:
        neo_rl = get_neo4j_risk_level(pair["clause_id"])
        if neo_rl is not None and abs(float(pair["risk_level"]) - float(neo_rl)) < 0.01:
            matching += 1
    rsc = matching / total_reported if total_reported > 0 else None

    return {
        "hallucination_rate": hr,
        "cited_ids": cited,
        "invalid_ids": invalid,
        "risk_score_consistency": rsc,
        "placeholder_preservation_rate": None,
    }


def measure_customization(
    response: Optional[Dict], clause_id: str, raw_text: str
) -> Dict:
    if response is None or not raw_text:
        return {
            "hallucination_rate": 0.0, "cited_ids": [], "invalid_ids": [],
            "risk_score_consistency": None, "placeholder_preservation_rate": None,
        }
    ctext = response.get("customized_text", "")
    ppr = compute_placeholder_preservation(raw_text, ctext)
    return {
        "hallucination_rate": 0.0,   # customization doesn't cite clause IDs
        "cited_ids": [],
        "invalid_ids": [],
        "risk_score_consistency": None,
        "placeholder_preservation_rate": ppr,
    }


def measure_qa(response: Optional[Dict]) -> Dict:
    if response is None:
        return {
            "hallucination_rate": None, "cited_ids": [], "invalid_ids": [],
            "risk_score_consistency": None, "placeholder_preservation_rate": None,
        }
    # Citations list approach
    citations = response.get("citations", [])
    cited = [c.get("clause_id") for c in citations if c.get("clause_id")]
    # Also sweep full response text
    cited.extend(extract_clause_ids_from_response(response))
    # Deduplicate
    cited = list(dict.fromkeys(cited))
    hr, invalid, _ = compute_hallucination_rate(cited)
    return {
        "hallucination_rate": hr,
        "cited_ids": cited,
        "invalid_ids": invalid,
        "risk_score_consistency": None,
        "placeholder_preservation_rate": None,
    }


# ---------------------------------------------------------------------------
# 14. CONDITION RUNNERS
# ---------------------------------------------------------------------------

def run_condition(
    condition_name: str,
    contracts: List[Dict],
    llm: LLMClient,
    retriever: Optional[GraphRAGRetriever],
    val: Optional[GroundingValidator],
    corpus: Optional["VectorRAGCorpus"],
) -> List[Dict]:
    """
    Run all 5 contracts × 4 tasks for a single condition.
    Returns a flat list of result dicts (one per task per contract).
    """
    results = []

    for ci, contract in enumerate(contracts, start=1):
        cid = contract["contract_id"]
        ctype = contract["contract_type"]
        active_ids = contract["active_clause_ids"]

        # ---- Task A: Recommendations ----
        print(f"  Running {condition_name}, Contract {ci}/5, Task recommendations…")
        try:
            if condition_name == "C1":
                response, latency = c1_recommendations(llm, contract)
            elif condition_name == "C2":
                response, latency = c2_recommendations(llm, corpus, contract)
            else:  # C3 / C3v
                response, latency = c3_recommendations(llm, retriever, val, contract)

            metrics = measure_recommendations(response, condition_name)
            metrics["latency_ms"] = latency
        except Exception as exc:
            log.error("    Error in %s/%s/recommendations: %s", condition_name, cid, exc)
            response, metrics = None, {
                "hallucination_rate": None, "cited_ids": [], "invalid_ids": [],
                "risk_score_consistency": None, "placeholder_preservation_rate": None,
                "latency_ms": 0.0,
            }

        results.append({
            "condition": condition_name,
            "contract_id": cid,
            "contract_type": ctype,
            "task": "recommendations",
            "llm_response": response,
            "metrics": metrics,
        })

        # ---- Task B: Risk Analysis ----
        print(f"  Running {condition_name}, Contract {ci}/5, Task risk…")
        try:
            if condition_name == "C1":
                response, latency = c1_risk(llm, contract)
            elif condition_name == "C2":
                response, latency = c2_risk(llm, corpus, contract)
            else:
                response, latency = c3_risk(llm, retriever, val, contract)

            metrics = measure_risk(response, contract)
            metrics["latency_ms"] = latency
        except Exception as exc:
            log.error("    Error in %s/%s/risk: %s", condition_name, cid, exc)
            response, metrics = None, {
                "hallucination_rate": None, "cited_ids": [], "invalid_ids": [],
                "risk_score_consistency": None, "placeholder_preservation_rate": None,
                "latency_ms": 0.0,
            }

        results.append({
            "condition": condition_name,
            "contract_id": cid,
            "contract_type": ctype,
            "task": "risk",
            "llm_response": response,
            "metrics": metrics,
        })

        # ---- Task C: Customization (one call per active clause) ----
        print(f"  Running {condition_name}, Contract {ci}/5, Task customization ({len(active_ids)} clauses)…")
        for clause_id in active_ids:
            raw_text = get_clause_raw_text(clause_id) or ""
            try:
                if condition_name == "C1":
                    response, latency = c1_customization(llm, clause_id, raw_text)
                elif condition_name == "C2":
                    response, latency = c2_customization(llm, corpus, clause_id, raw_text)
                else:
                    response, latency = c3_customization(llm, retriever, val, clause_id)

                metrics = measure_customization(response, clause_id, raw_text)
                metrics["latency_ms"] = latency
            except Exception as exc:
                log.error("    Error in %s/%s/customization/%s: %s", condition_name, cid, clause_id, exc)
                response, metrics = None, {
                    "hallucination_rate": 0.0, "cited_ids": [], "invalid_ids": [],
                    "risk_score_consistency": None, "placeholder_preservation_rate": None,
                    "latency_ms": 0.0,
                }

            results.append({
                "condition": condition_name,
                "contract_id": cid,
                "contract_type": ctype,
                "task": "customization",
                "clause_id": clause_id,
                "llm_response": response,
                "metrics": metrics,
            })

        # ---- Task D: Q&A Chatbot (3 questions) ----
        print(f"  Running {condition_name}, Contract {ci}/5, Task Q&A…")
        for qi, question in enumerate(QA_QUESTIONS, start=1):
            try:
                if condition_name == "C1":
                    response, latency = c1_qa(llm, contract, question)
                elif condition_name == "C2":
                    response, latency = c2_qa(llm, corpus, contract, question)
                else:
                    response, latency = c3_qa(llm, retriever, val, contract, question)

                metrics = measure_qa(response)
                metrics["latency_ms"] = latency
            except Exception as exc:
                log.error("    Error in %s/%s/qa/q%d: %s", condition_name, cid, qi, exc)
                response, metrics = None, {
                    "hallucination_rate": None, "cited_ids": [], "invalid_ids": [],
                    "risk_score_consistency": None, "placeholder_preservation_rate": None,
                    "latency_ms": 0.0,
                }

            results.append({
                "condition": condition_name,
                "contract_id": cid,
                "contract_type": ctype,
                "task": "qa",
                "question": question,
                "llm_response": response,
                "metrics": metrics,
            })

    return results


# ---------------------------------------------------------------------------
# 15. AGGREGATE METRICS
# ---------------------------------------------------------------------------

def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def aggregate(results: List[Dict]) -> Dict:
    """Compute aggregate metrics for a single condition's results."""

    def rows_by_task(task):
        return [r for r in results if r["task"] == task]

    # HR (all tasks)
    all_hr  = [r["metrics"]["hallucination_rate"] for r in results]
    rec_hr  = [r["metrics"]["hallucination_rate"] for r in rows_by_task("recommendations")]
    chat_hr = [r["metrics"]["hallucination_rate"] for r in rows_by_task("qa")]

    # RSC (risk only)
    rsc_vals = [r["metrics"]["risk_score_consistency"] for r in rows_by_task("risk")]

    # PPR (customization only)
    ppr_vals = [r["metrics"]["placeholder_preservation_rate"] for r in rows_by_task("customization")]

    # Latency (all tasks)
    latencies = [r["metrics"].get("latency_ms", 0.0) for r in results]

    return {
        "hr_all":       _safe_mean(all_hr),
        "hr_recs":      _safe_mean(rec_hr),
        "hr_chat":      _safe_mean(chat_hr),
        "rsc":          _safe_mean(rsc_vals),
        "ppr":          _safe_mean(ppr_vals),
        "avg_latency":  _safe_mean(latencies) if latencies else None,
    }


# ---------------------------------------------------------------------------
# 16. SUMMARY TABLE
# ---------------------------------------------------------------------------

def pct(v: Optional[float], default: str = "N/A") -> str:
    if v is None:
        return default
    return f"{v*100:.1f}%"


def ms(v: Optional[float], default: str = "N/A") -> str:
    if v is None:
        return default
    return f"{v:.0f}ms"


def format_summary_table(aggregates: Dict[str, Dict]) -> str:
    col = [
        ("CONDITION",          20),
        ("HR (all)",            9),
        ("HR (recs)",          10),
        ("HR (chat)",          10),
        ("RSC",                 7),
        ("PPR",                 7),
        ("Avg Latency",        12),
    ]
    header    = " | ".join(f"{h:<{w}}" for h, w in col)
    separator = "-+-".join("-" * w for _, w in col)

    display_names = {
        "C1":  "C1 (LLM only)      ",
        "C2":  "C2 (Vector RAG)    ",
        "C3":  "C3 (no validator)  ",
        "C3v": "C3 (with validator)",
    }

    rows = []
    for cname, agg in aggregates.items():
        cells = [
            display_names.get(cname, f"{cname:<20}"),
            pct(agg.get("hr_all")),
            pct(agg.get("hr_recs")),
            pct(agg.get("hr_chat")),
            pct(agg.get("rsc")),
            pct(agg.get("ppr")),
            ms(agg.get("avg_latency")),
        ]
        rows.append(" | ".join(f"{c:<{w}}" for c, (_, w) in zip(cells, col)))

    lines = [
        "=" * (sum(w for _, w in col) + 3 * (len(col) - 1)),
        "LEGALWIZ HALLUCINATION EVALUATION — RESULTS SUMMARY",
        "=" * (sum(w for _, w in col) + 3 * (len(col) - 1)),
        "",
        header,
        separator,
    ] + rows + [""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 17. MAIN
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("LegalWiz Hallucination Evaluation")
    print("="*60 + "\n")

    # --- Shared objects ---
    llm       = LLMClient(EVAL_LLM_CONFIG)       # uses 8b/flash model
    retriever = GraphRAGRetriever()
    val_on    = GroundingValidator(retriever)    # C3v
    val_off   = None                             # C3

    print(f"  {log_msg}")
    print(f"  Rate-limit sleep: {RATE_LIMIT_SLEEP}s between calls\n")

    if not llm.is_configured():
        log.error("LLM not configured — set LLM_API_KEY in .env. Aborting.")
        sys.exit(1)

    # --- Build test contracts ---
    print("Building synthetic test contracts from Neo4j…")
    contracts = build_test_contracts()
    print(f"  {len(contracts)} contracts ready.\n")

    # --- C2 corpus (optional) ---
    corpus = None
    if _ST_AVAILABLE:
        print("Building C2 vector corpus…")
        all_clauses = fetch_all_clauses_from_neo4j()
        if all_clauses:
            corpus = VectorRAGCorpus(all_clauses)
            print(f"  Corpus: {len(all_clauses)} clauses indexed.\n")
        else:
            log.warning("No clauses returned from Neo4j — C2 will be skipped.")
    else:
        print("sentence-transformers not available — skipping C2.\n")

    # Conditions to run (comment out any to reduce API calls)
    conditions_to_run = [
        ("C1",  None,      None,    None),
        ("C2",  None,      None,    corpus),
        ("C3",  retriever, val_off, None),
        ("C3v", retriever, val_on,  None),
    ]

    all_results: List[Dict] = []
    aggregates: Dict[str, Dict] = {}

    for (condition_name, retr, val, corp) in conditions_to_run:
        if condition_name == "C2" and corpus is None:
            print(f"\n[SKIP] {condition_name} — no vector corpus.\n")
            continue

        print(f"\n{'='*50}")
        print(f"CONDITION: {condition_name}")
        print(f"{'='*50}\n")

        cond_results = run_condition(
            condition_name=condition_name,
            contracts=contracts,
            llm=llm,
            retriever=retr,
            val=val,
            corpus=corp,
        )
        all_results.extend(cond_results)
        aggregates[condition_name] = aggregate(cond_results)
        print(f"\n  ✓ {condition_name} complete — {len(cond_results)} task results.")

    # --- Write raw JSON ---
    raw_path = RESULTS_DIR / "eval_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Raw results → {raw_path}")

    # --- Write summary ---
    summary_text = format_summary_table(aggregates)
    summary_path = RESULTS_DIR / "eval_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"✓ Summary      → {summary_path}")

    # --- Print summary to stdout ---
    print("\n" + summary_text)


if __name__ == "__main__":
    main()
