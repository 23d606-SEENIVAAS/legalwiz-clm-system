# graph_rag_engine.py - Core Graph RAG Engine for LegalWiz CLM
"""
Central engine that powers all AI features through Graph RAG:
1. GraphRAGRetriever - Fetches structured context from Neo4j
2. LLMClient - Provider-agnostic LLM interface (Gemini default)
3. GroundingValidator - Validates LLM output against graph data

Architecture: Graph Retrieval → LLM Generation → Validation → Response
The graph DECIDES, the LLM EXPLAINS, the validator VERIFIES.
"""

import json
import re
import os
from typing import List, Dict, Optional, Any, Tuple
from neo4j import GraphDatabase
import psycopg2
from psycopg2.extras import RealDictCursor
from config import DB_CONFIG, NEO4J_CONFIG
from llm_config import LLM_CONFIG


# ============================================================================
# 1. GRAPH RAG RETRIEVER
#    Traverses Neo4j to fetch structured context for LLM consumption
# ============================================================================

class GraphRAGRetriever:
    """
    Retrieves structured context from the Neo4j knowledge graph.
    Each method returns graph data formatted for LLM prompts.
    """
    
    def __init__(self):
        self.neo4j_config = NEO4J_CONFIG
        self.db_config = DB_CONFIG
    
    def _get_driver(self):
        return GraphDatabase.driver(
            self.neo4j_config["uri"],
            auth=(self.neo4j_config["username"], self.neo4j_config["password"]),
            database=self.neo4j_config.get("database", "neo4j")
        )
    
    def _get_pg(self):
        return psycopg2.connect(**self.db_config)
    
    # ------ Active Clause Helpers ------
    
    def get_active_clause_ids(self, contract_id: str) -> List[str]:
        """Get list of active clause_ids for a contract from Supabase."""
        conn = self._get_pg()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT clause_id, clause_type, variant
                    FROM contract_clauses
                    WHERE contract_id = %s AND is_active = true
                    ORDER BY sequence
                """, (contract_id,))
                return cur.fetchall()
        finally:
            conn.close()
    
    def get_contract_info(self, contract_id: str) -> Optional[Dict]:
        """Get contract metadata from Supabase."""
        conn = self._get_pg()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, title, contract_type, jurisdiction, status, description
                    FROM contracts WHERE id = %s
                """, (contract_id,))
                return cur.fetchone()
        finally:
            conn.close()
    
    # ------ RECOMMENDATION CONTEXT ------
    
    def get_recommendation_context(
        self, contract_type: str, jurisdiction: str, active_clause_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Retrieve all data needed for smart recommendations:
        1. ALTERNATIVE_TO relationships for current clauses
        2. REQUIRES dependencies that might be missing
        3. Optional clause types not yet selected
        """
        driver = self._get_driver()
        neo4j_ct_id = contract_type.replace("_", "-")
        
        try:
            with driver.session() as session:
                # 1. Get alternatives to current active clauses
                alternatives = session.run("""
                    MATCH (active:Clause)-[alt:ALTERNATIVE_TO]->(better:Clause)
                    WHERE active.id IN $active_ids
                      AND better.jurisdiction = $jurisdiction
                      AND NOT better.id IN $active_ids
                    RETURN 
                        active.id AS current_clause_id,
                        active.variant AS current_variant,
                        active.risk_level AS current_risk,
                        active.clause_type AS clause_type,
                        better.id AS recommended_clause_id,
                        better.variant AS recommended_variant,
                        better.risk_level AS recommended_risk,
                        alt.alternative_type AS alternative_type,
                        alt.reason AS reason,
                        alt.benefit AS benefit,
                        alt.recommendation_strength AS strength
                    ORDER BY alt.recommendation_strength DESC
                """, {
                    "active_ids": active_clause_ids,
                    "jurisdiction": jurisdiction
                })
                alternatives_data = [dict(r) for r in alternatives]
                
                # 2. Get REQUIRES dependencies and check for gaps
                requires = session.run("""
                    MATCH (ct:ContractType {id: $contract_type})
                          -[:CONTAINS_CLAUSE]->(clauseType:ClauseType)
                          -[:HAS_VARIANT]->(c:Clause)
                    WHERE c.id IN $active_ids
                    WITH DISTINCT clauseType
                    MATCH (clauseType)-[req:REQUIRES]->(required:ClauseType)
                    OPTIONAL MATCH (required)-[:HAS_VARIANT]->(reqClause:Clause)
                    WHERE reqClause.jurisdiction = $jurisdiction
                    RETURN 
                        clauseType.id AS source_clause_type,
                        clauseType.name AS source_name,
                        required.id AS required_clause_type,
                        required.name AS required_name,
                        req.dependency_type AS dependency_type,
                        req.is_critical AS is_critical,
                        req.reason AS reason,
                        collect(reqClause.id) AS available_clause_ids
                """, {
                    "contract_type": neo4j_ct_id,
                    "active_ids": active_clause_ids,
                    "jurisdiction": jurisdiction
                })
                requires_data = []
                for r in requires:
                    rec = dict(r)
                    # Check if any of the required clause type's variants are in active clauses
                    has_required = any(cid in active_clause_ids for cid in rec["available_clause_ids"])
                    rec["is_missing"] = not has_required
                    requires_data.append(rec)
                
                # 3. Get optional clause types available but not selected
                optional_gaps = session.run("""
                    MATCH (ct:ContractType {id: $contract_type})
                          -[rel:CONTAINS_CLAUSE]->(clauseType:ClauseType)
                          -[:HAS_VARIANT]->(c:Clause)
                    WHERE rel.mandatory = false
                      AND c.jurisdiction = $jurisdiction
                      AND NOT c.id IN $active_ids
                    WITH DISTINCT clauseType, rel
                    RETURN 
                        clauseType.id AS clause_type_id,
                        clauseType.name AS clause_type_name,
                        clauseType.category AS category,
                        clauseType.importance_level AS importance_level,
                        rel.description AS description
                    ORDER BY clauseType.importance_level DESC
                """, {
                    "contract_type": neo4j_ct_id,
                    "active_ids": active_clause_ids,
                    "jurisdiction": jurisdiction
                })
                optional_data = [dict(r) for r in optional_gaps]
                
                return {
                    "alternatives": alternatives_data,
                    "requires": [r for r in requires_data if r["is_missing"]],
                    "optional_gaps": optional_data
                }
        finally:
            driver.close()
    
    # ------ CUSTOMIZATION CONTEXT ------
    
    def get_customization_context(self, clause_id: str) -> Dict[str, Any]:
        """
        Retrieve all data needed for clause customization:
        1. Full clause text + metadata
        2. All variant alternatives for this clause type
        3. Parameters used in this clause
        """
        driver = self._get_driver()
        try:
            with driver.session() as session:
                # 1. Get the target clause + its clause type info
                clause_result = session.run("""
                    MATCH (ct:ClauseType)-[:HAS_VARIANT]->(c:Clause {id: $clause_id})
                    RETURN
                        c.id AS clause_id,
                        c.raw_text AS raw_text,
                        c.variant AS variant,
                        c.risk_level AS risk_level,
                        c.jurisdiction AS jurisdiction,
                        c.clause_type AS clause_type,
                        ct.id AS clause_type_id,
                        ct.name AS clause_type_name,
                        ct.category AS category,
                        ct.importance_level AS importance_level
                """, {"clause_id": clause_id})
                
                clause_data = clause_result.single()
                if not clause_data:
                    return None
                clause_dict = dict(clause_data)
                
                # 2. Get all variants of this clause type
                variants_result = session.run("""
                    MATCH (ct:ClauseType {id: $clause_type_id})-[:HAS_VARIANT]->(v:Clause)
                    WHERE v.jurisdiction = $jurisdiction
                    RETURN
                        v.id AS clause_id,
                        v.variant AS variant,
                        v.risk_level AS risk_level,
                        v.raw_text AS raw_text
                    ORDER BY v.risk_level
                """, {
                    "clause_type_id": clause_dict["clause_type_id"],
                    "jurisdiction": clause_dict["jurisdiction"]
                })
                variants = [dict(v) for v in variants_result]
                
                # 3. Get parameters for this clause
                params_result = session.run("""
                    MATCH (c:Clause {id: $clause_id})-[:CONTAINS_PARAM]->(p:Parameter)
                    RETURN
                        p.id AS parameter_id,
                        p.name AS parameter_name,
                        p.data_type AS data_type,
                        p.is_required AS is_required
                    ORDER BY p.name
                """, {"clause_id": clause_id})
                parameters = [dict(p) for p in params_result]
                
                return {
                    "clause": clause_dict,
                    "all_variants": variants,
                    "parameters": parameters
                }
        finally:
            driver.close()
    
    # ------ RISK ANALYSIS CONTEXT ------
    
    def get_risk_context(
        self, contract_type: str, jurisdiction: str, active_clause_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Retrieve all data needed for risk analysis:
        1. Risk levels for all active clauses
        2. CONFLICTS_WITH relationships between active clauses
        3. REQUIRES dependencies and missing gaps
        4. Clause types available but not included
        """
        driver = self._get_driver()
        neo4j_ct_id = contract_type.replace("_", "-")
        
        try:
            with driver.session() as session:
                # 1. Get risk levels + metadata for active clauses
                clause_risks = session.run("""
                    MATCH (c:Clause)
                    WHERE c.id IN $active_ids
                    OPTIONAL MATCH (ct:ClauseType)-[:HAS_VARIANT]->(c)
                    RETURN
                        c.id AS clause_id,
                        c.clause_type AS clause_type,
                        c.variant AS variant,
                        c.risk_level AS risk_level,
                        ct.importance_level AS importance_level,
                        ct.category AS category,
                        ct.name AS clause_type_name
                    ORDER BY c.risk_level DESC
                """, {"active_ids": active_clause_ids})
                risk_data = [dict(r) for r in clause_risks]
                
                # 2. Get CONFLICTS_WITH between active clauses
                conflicts = session.run("""
                    MATCH (a:Clause)-[conf:CONFLICTS_WITH]->(b:Clause)
                    WHERE a.id IN $active_ids AND b.id IN $active_ids
                    RETURN
                        a.id AS clause_a_id,
                        a.clause_type AS clause_a_type,
                        a.variant AS clause_a_variant,
                        b.id AS clause_b_id,
                        b.clause_type AS clause_b_type,
                        b.variant AS clause_b_variant,
                        conf.severity AS severity,
                        conf.reason AS reason,
                        conf.conflict_type AS conflict_type,
                        conf.resolution_advice AS resolution_advice
                    ORDER BY conf.severity DESC
                """, {"active_ids": active_clause_ids})
                conflict_data = [dict(r) for r in conflicts]
                
                # 3. Get REQUIRES dependencies and detect missing ones
                missing_deps = session.run("""
                    MATCH (ct:ContractType {id: $contract_type})
                          -[:CONTAINS_CLAUSE]->(clauseType:ClauseType)
                          -[:HAS_VARIANT]->(c:Clause)
                    WHERE c.id IN $active_ids
                    WITH DISTINCT clauseType
                    MATCH (clauseType)-[req:REQUIRES]->(required:ClauseType)
                    OPTIONAL MATCH (required)-[:HAS_VARIANT]->(reqClause:Clause)
                    WHERE reqClause.jurisdiction = $jurisdiction
                    WITH clauseType, required, req, collect(reqClause.id) AS available_ids
                    WHERE NONE(aid IN available_ids WHERE aid IN $active_ids)
                    RETURN
                        clauseType.id AS source_type,
                        clauseType.name AS source_name,
                        required.id AS missing_type,
                        required.name AS missing_name,
                        req.dependency_type AS dependency_type,
                        req.is_critical AS is_critical,
                        req.reason AS reason
                """, {
                    "contract_type": neo4j_ct_id,
                    "active_ids": active_clause_ids,
                    "jurisdiction": jurisdiction
                })
                missing_dep_data = [dict(r) for r in missing_deps]
                
                # 4. Gap analysis: clause types in contract template but not selected
                gaps = session.run("""
                    MATCH (ct:ContractType {id: $contract_type})
                          -[rel:CONTAINS_CLAUSE]->(clauseType:ClauseType)
                    WHERE rel.mandatory = true
                    WITH clauseType, rel
                    OPTIONAL MATCH (clauseType)-[:HAS_VARIANT]->(c:Clause)
                    WHERE c.jurisdiction = $jurisdiction
                    WITH clauseType, rel, collect(c.id) AS variant_ids
                    WHERE NONE(vid IN variant_ids WHERE vid IN $active_ids)
                    RETURN
                        clauseType.id AS clause_type_id,
                        clauseType.name AS clause_type_name,
                        clauseType.importance_level AS importance_level,
                        rel.description AS description
                """, {
                    "contract_type": neo4j_ct_id,
                    "active_ids": active_clause_ids,
                    "jurisdiction": jurisdiction
                })
                gap_data = [dict(r) for r in gaps]
                
                # 5. Get contract type metadata
                ct_info = session.run("""
                    MATCH (ct:ContractType {id: $contract_type})
                    RETURN ct.name AS name, ct.description AS description,
                           ct.complexity AS complexity, ct.use_case AS use_case
                """, {"contract_type": neo4j_ct_id})
                ct_record = ct_info.single()
                ct_data = dict(ct_record) if ct_record else {}
                
                return {
                    "contract_type_info": ct_data,
                    "clause_risks": risk_data,
                    "conflicts": conflict_data,
                    "missing_dependencies": missing_dep_data,
                    "gaps": gap_data
                }
        finally:
            driver.close()
    
    # ------ CHATBOT / QA CONTEXT ------
    
    def get_qa_context(
        self, contract_id: str, active_clause_ids: List[str], question: str
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for answering a question about the contract.
        Fetches clause texts + parameters for the relevant clauses.
        """
        driver = self._get_driver()
        
        try:
            with driver.session() as session:
                # Get all active clauses with full text
                clauses = session.run("""
                    MATCH (c:Clause)
                    WHERE c.id IN $active_ids
                    OPTIONAL MATCH (ct:ClauseType)-[:HAS_VARIANT]->(c)
                    RETURN
                        c.id AS clause_id,
                        c.clause_type AS clause_type,
                        c.variant AS variant,
                        c.risk_level AS risk_level,
                        c.raw_text AS raw_text,
                        ct.name AS clause_type_name
                    ORDER BY c.clause_type
                """, {"active_ids": active_clause_ids})
                clause_data = [dict(r) for r in clauses]
                
                # Get parameter values from Supabase
                conn = self._get_pg()
                try:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        cur.execute("""
                            SELECT cp.parameter_id, 
                                   cp.value_text, cp.value_integer, cp.value_decimal,
                                   cp.value_date, cp.value_currency
                            FROM contract_parameters cp
                            WHERE cp.contract_id = %s
                        """, (contract_id,))
                        param_rows = cur.fetchall()
                finally:
                    conn.close()
                
                # Build param values map
                param_values = {}
                for row in param_rows:
                    pid = row["parameter_id"]
                    if row["value_text"]:
                        param_values[pid] = row["value_text"]
                    elif row["value_integer"] is not None:
                        param_values[pid] = str(row["value_integer"])
                    elif row["value_decimal"] is not None:
                        param_values[pid] = str(row["value_decimal"])
                    elif row["value_date"]:
                        param_values[pid] = row["value_date"].isoformat()
                    elif row["value_currency"]:
                        param_values[pid] = str(row["value_currency"])
                
                return {
                    "clauses": clause_data,
                    "parameter_values": param_values
                }
        finally:
            driver.close()
    
    # ------ VALIDATION HELPERS ------
    
    def verify_clause_exists(self, clause_id: str) -> bool:
        """Verify a clause_id exists in Neo4j."""
        driver = self._get_driver()
        try:
            with driver.session() as session:
                result = session.run(
                    "MATCH (c:Clause {id: $id}) RETURN c.id",
                    {"id": clause_id}
                )
                return result.single() is not None
        finally:
            driver.close()
    
    def verify_clause_ids_batch(self, clause_ids: List[str]) -> Dict[str, bool]:
        """Verify multiple clause_ids exist in Neo4j. Returns {id: exists}."""
        driver = self._get_driver()
        try:
            with driver.session() as session:
                result = session.run("""
                    UNWIND $ids AS check_id
                    OPTIONAL MATCH (c:Clause {id: check_id})
                    RETURN check_id AS id, c IS NOT NULL AS exists
                """, {"ids": clause_ids})
                return {r["id"]: r["exists"] for r in result}
        finally:
            driver.close()


# ============================================================================
# 2. LLM CLIENT
#    Provider-agnostic LLM interface. Currently supports Google Gemini.
# ============================================================================

class LLMClient:
    """
    Provider-agnostic LLM client.
    Wraps LLM API calls with structured output enforcement.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or LLM_CONFIG
        self.provider = self.config.get("provider", "gemini")
        self._client = None
        self._model = None
    
    def _init_gemini(self):
        """Initialize Google Gemini client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config["api_key"])
            
            self._model = genai.GenerativeModel(
                model_name=self.config.get("model", "gemini-2.0-flash"),
                generation_config={
                    "temperature": self.config.get("temperature", 0.1),
                    "max_output_tokens": self.config.get("max_tokens", 4096),
                }
            )
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )
    
    def _init_openai(self):
        """Initialize OpenAI or Groq client (both use OpenAI SDK)."""
        try:
            from openai import OpenAI
            if self.provider == "groq":
                # Groq uses OpenAI-compatible API
                self._client = OpenAI(
                    api_key=self.config["api_key"],
                    base_url="https://api.groq.com/openai/v1"
                )
            else:
                # Standard OpenAI
                self._client = OpenAI(api_key=self.config["api_key"])
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: str = "",
        retry_count: int = 2
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        Always returns parsed JSON (structured output).
        
        Args:
            prompt: The user/task prompt
            system_prompt: System instructions
            retry_count: Number of retries on parse failure
            
        Returns:
            Parsed JSON dict from the LLM response
        """
        for attempt in range(retry_count + 1):
            try:
                if self.provider == "gemini":
                    return self._generate_gemini(prompt, system_prompt)
                elif self.provider in ("openai", "groq"):
                    return self._generate_openai(prompt, system_prompt)
                else:
                    raise ValueError(f"Unknown LLM provider: {self.provider}")
            except json.JSONDecodeError as e:
                if attempt < retry_count:
                    continue
                raise ValueError(
                    f"LLM returned invalid JSON after {retry_count + 1} attempts: {e}"
                )
    
    def _generate_gemini(self, prompt: str, system_prompt: str = "") -> Dict:
        """Generate using Google Gemini."""
        if not self._model:
            self._init_gemini()
        
        json_instruction = "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no explanations, just the JSON object."
        full_prompt = f"{system_prompt}\n\n{prompt}{json_instruction}" if system_prompt else f"{prompt}{json_instruction}"
        
        try:
            response = self._model.generate_content(full_prompt)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "ResourceExhausted" in err_msg or "quota" in err_msg.lower():
                raise RuntimeError(
                    "Gemini API quota exceeded. Your free tier limit has been reached. "
                    "Options: (1) Wait ~60 seconds and retry, (2) Upgrade your API plan at https://ai.google.dev, "
                    "or (3) Use a different API key."
                )
            raise
        
        # Parse JSON response
        text = response.text.strip()
        
        # Handle markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        return json.loads(text)
    
    def _generate_openai(self, prompt: str, system_prompt: str = "") -> Dict:
        """Generate using OpenAI or Groq (OpenAI-compatible API)."""
        if not self._client:
            self._init_openai()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Default models per provider
        default_model = "llama-3.1-70b-versatile" if self.provider == "groq" else "gpt-4o"
        
        response = self._client.chat.completions.create(
            model=self.config.get("model", default_model),
            messages=messages,
            temperature=self.config.get("temperature", 0.1),
            max_tokens=self.config.get("max_tokens", 4096),
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def is_configured(self) -> bool:
        """Check if the LLM is properly configured."""
        return bool(self.config.get("api_key"))


# ============================================================================
# 3. GROUNDING VALIDATOR
#    Validates LLM output against graph data to prevent hallucination
# ============================================================================

class GroundingValidator:
    """
    Validates LLM outputs against the knowledge graph.
    Prevents hallucination by cross-checking every claim.
    """
    
    def __init__(self, retriever: GraphRAGRetriever):
        self.retriever = retriever
    
    def validate_clause_ids(self, clause_ids: List[str]) -> Dict[str, Any]:
        """
        Validate that all clause_ids in LLM output actually exist in Neo4j.
        Returns validation result with details on invalid IDs.
        """
        if not clause_ids:
            return {"valid": True, "invalid_ids": [], "valid_ids": []}
        
        existence_map = self.retriever.verify_clause_ids_batch(clause_ids)
        invalid = [cid for cid, exists in existence_map.items() if not exists]
        valid = [cid for cid, exists in existence_map.items() if exists]
        
        return {
            "valid": len(invalid) == 0,
            "invalid_ids": invalid,
            "valid_ids": valid,
            "total_checked": len(clause_ids),
            "validation_rate": len(valid) / len(clause_ids) if clause_ids else 1.0
        }
    
    def validate_placeholders(
        self, original_text: str, customized_text: str
    ) -> Dict[str, Any]:
        """
        Validate that ALL {{PLACEHOLDER}} tokens from original text
        are preserved in the customized text.
        """
        # Extract placeholders
        original_placeholders = set(re.findall(r'\{\{[A-Z_0-9]+\}\}', original_text))
        customized_placeholders = set(re.findall(r'\{\{[A-Z_0-9]+\}\}', customized_text))
        
        missing = original_placeholders - customized_placeholders
        added = customized_placeholders - original_placeholders
        
        return {
            "valid": len(missing) == 0,
            "original_placeholders": sorted(list(original_placeholders)),
            "preserved_placeholders": sorted(list(original_placeholders & customized_placeholders)),
            "missing_placeholders": sorted(list(missing)),
            "new_placeholders": sorted(list(added)),
            "preservation_rate": (
                len(original_placeholders & customized_placeholders) / len(original_placeholders)
                if original_placeholders else 1.0
            )
        }
    
    def validate_recommendations(
        self, recommendations: List[Dict], graph_context: Dict
    ) -> Dict[str, Any]:
        """
        Validate that all recommendations reference real graph data.
        Checks:
        1. All recommended clause_ids exist
        2. Recommendations come from actual alternatives/requires data
        """
        rec_clause_ids = [
            r.get("recommended_clause_id") 
            for r in recommendations 
            if r.get("recommended_clause_id")
        ]
        
        # Validate clause IDs
        id_validation = self.validate_clause_ids(rec_clause_ids) if rec_clause_ids else {
            "valid": True, "invalid_ids": [], "valid_ids": []
        }
        
        # Check that recommendations are grounded in graph context
        graph_alt_ids = {a["recommended_clause_id"] for a in graph_context.get("alternatives", [])}
        graph_req_types = {r["required_clause_type"] for r in graph_context.get("requires", [])}
        
        grounded_recs = []
        ungrounded_recs = []
        
        for rec in recommendations:
            rec_id = rec.get("recommended_clause_id", "")
            rec_type = rec.get("clause_type", "")
            
            if rec_id in graph_alt_ids or rec_type in graph_req_types:
                grounded_recs.append(rec)
            else:
                ungrounded_recs.append(rec)
        
        return {
            "valid": id_validation["valid"] and len(ungrounded_recs) == 0,
            "id_validation": id_validation,
            "grounded_count": len(grounded_recs),
            "ungrounded_count": len(ungrounded_recs),
            "ungrounded_recommendations": ungrounded_recs,
            "grounding_rate": (
                len(grounded_recs) / len(recommendations) 
                if recommendations else 1.0
            )
        }
    
    def validate_risk_scores(
        self, llm_risks: List[Dict], graph_risks: List[Dict]
    ) -> Dict[str, Any]:
        """
        Validate that risk scores from LLM match graph data.
        Risk scores must come from the graph, not be invented.
        """
        graph_risk_map = {r["clause_id"]: r["risk_level"] for r in graph_risks}
        
        mismatches = []
        for llm_risk in llm_risks:
            cid = llm_risk.get("clause_id")
            llm_level = llm_risk.get("risk_level")
            graph_level = graph_risk_map.get(cid)
            
            if graph_level is not None and llm_level != graph_level:
                mismatches.append({
                    "clause_id": cid,
                    "llm_risk": llm_level,
                    "graph_risk": graph_level
                })
        
        return {
            "valid": len(mismatches) == 0,
            "mismatches": mismatches,
            "checked_count": len(llm_risks),
            "accuracy_rate": (
                (len(llm_risks) - len(mismatches)) / len(llm_risks)
                if llm_risks else 1.0
            )
        }
    
    def validate_citations(
        self, citations: List[Dict], clause_data: List[Dict]
    ) -> Dict[str, Any]:
        """
        Validate that chatbot citations reference real clause data.
        Checks that cited clause_ids exist in the provided clause context.
        """
        available_ids = {c["clause_id"] for c in clause_data}
        
        valid_citations = []
        invalid_citations = []
        
        for citation in citations:
            if citation.get("clause_id") in available_ids:
                valid_citations.append(citation)
            else:
                invalid_citations.append(citation)
        
        return {
            "valid": len(invalid_citations) == 0,
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
            "citation_rate": (
                len(valid_citations) / len(citations) 
                if citations else 1.0
            )
        }


# ============================================================================
# 4. CONVENIENCE: Singleton instances for use across routes
# ============================================================================

# These are initialized once and reused across all route modules
retriever = GraphRAGRetriever()
llm_client = LLMClient()
validator = GroundingValidator(retriever)
