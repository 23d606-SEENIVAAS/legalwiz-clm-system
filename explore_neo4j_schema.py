"""
Neo4j Schema Explorer for LegalWiz CLM
Run this to dump the full graph structure (node labels, relationships, properties, sample data)
Usage: python explore_neo4j_schema.py
"""
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


def explore_schema():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    with driver.session(database=NEO4J_DATABASE) as session:
        print("=" * 80)
        print("NEO4J GRAPH SCHEMA EXPLORATION")
        print("=" * 80)
        
        # 1. Node labels and counts
        print("\n📦 NODE LABELS & COUNTS")
        print("-" * 40)
        result = session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")
        labels = [r["label"] for r in result]
        for label in labels:
            count_result = session.run(f"MATCH (n:`{label}`) RETURN count(n) AS c")
            count = count_result.single()["c"]
            print(f"  :{label}  →  {count} nodes")
        
        # 2. Relationship types and counts
        print("\n🔗 RELATIONSHIP TYPES & COUNTS")
        print("-" * 40)
        result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType")
        rel_types = [r["relationshipType"] for r in result]
        for rel_type in rel_types:
            count_result = session.run(f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) AS c")
            count = count_result.single()["c"]
            print(f"  [:{rel_type}]  →  {count} relationships")
        
        # 3. Relationship patterns (which labels connect to which)
        print("\n🔀 RELATIONSHIP PATTERNS")
        print("-" * 40)
        result = session.run("""
            CALL db.schema.visualization() YIELD nodes, relationships
            RETURN nodes, relationships
        """)
        # Fallback: manual pattern detection
        result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN DISTINCT labels(a) AS from_labels, type(r) AS rel_type, labels(b) AS to_labels,
                   count(*) AS count
            ORDER BY count DESC
        """)
        for r in result:
            from_l = ":".join(r["from_labels"])
            to_l = ":".join(r["to_labels"])
            print(f"  (:{from_l})-[:{r['rel_type']}]->(:{to_l})  ×{r['count']}")
        
        # 4. Node properties per label
        print("\n📋 NODE PROPERTIES (per label)")
        print("-" * 40)
        for label in labels:
            result = session.run(f"""
                MATCH (n:`{label}`)
                WITH n LIMIT 1
                RETURN keys(n) AS props, n AS sample
            """)
            record = result.single()
            if record:
                props = sorted(record["props"])
                sample = dict(record["sample"])
                print(f"\n  :{label}")
                print(f"    Properties: {props}")
                # Print sample with truncated values
                print(f"    Sample node:")
                for key in sorted(sample.keys()):
                    val = sample[key]
                    val_str = str(val)
                    if len(val_str) > 120:
                        val_str = val_str[:120] + "..."
                    print(f"      {key}: {val_str}")
        
        # 5. Relationship properties
        print("\n\n📋 RELATIONSHIP PROPERTIES")
        print("-" * 40)
        for rel_type in rel_types:
            result = session.run(f"""
                MATCH ()-[r:`{rel_type}`]->()
                WITH r LIMIT 1
                RETURN keys(r) AS props, properties(r) AS sample
            """)
            record = result.single()
            if record and record["props"]:
                props = sorted(record["props"])
                sample = dict(record["sample"])
                print(f"\n  [:{rel_type}]")
                print(f"    Properties: {props}")
                print(f"    Sample: {sample}")

        # 6. Sample ContractType nodes (to see what contract types exist)
        print("\n\n📄 ALL CONTRACT TYPES")
        print("-" * 40)
        result = session.run("MATCH (ct:ContractType) RETURN ct.id AS id, ct.name AS name ORDER BY ct.id")
        for r in result:
            print(f"  {r['id']}  →  {r['name']}")
        
        # 7. Sample ClauseType nodes
        print("\n📄 ALL CLAUSE TYPES")
        print("-" * 40)
        result = session.run("MATCH (ct:ClauseType) RETURN ct.id AS id, ct.name AS name ORDER BY ct.id")
        for r in result:
            print(f"  {r['id']}  →  {r['name']}")
        
        # 8. Clause variants for one contract type (sample)
        print("\n📄 SAMPLE: Clauses for first ContractType")
        print("-" * 40)
        result = session.run("""
            MATCH (ct:ContractType)
            WITH ct LIMIT 1
            MATCH (ct)-[rel:CONTAINS_CLAUSE]->(clauseType:ClauseType)-[:HAS_VARIANT]->(c:Clause)
            RETURN ct.id AS contract_type,
                   clauseType.id AS clause_type,
                   c.id AS clause_id,
                   c.variant AS variant,
                   c.jurisdiction AS jurisdiction,
                   c.risk_level AS risk_level,
                   SUBSTRING(c.raw_text, 0, 100) AS text_preview,
                   rel.sequence AS sequence,
                   rel.mandatory AS mandatory
            ORDER BY rel.sequence, c.variant
            LIMIT 30
        """)
        for r in result:
            print(f"  [{r['sequence']}] {r['clause_type']} / {r['variant']} / {r['jurisdiction']}")
            print(f"      ID: {r['clause_id']}  Risk: {r['risk_level']}  Mandatory: {r['mandatory']}")
            print(f"      Text: {r['text_preview']}...")
            print()

        # 9. Parameter nodes sample
        print("\n📄 SAMPLE: Parameter nodes (first 20)")
        print("-" * 40)
        result = session.run("""
            MATCH (p:Parameter)
            RETURN p.id AS id, p.name AS name, p.data_type AS data_type, p.is_required AS is_required
            ORDER BY p.id
            LIMIT 20
        """)
        for r in result:
            req = "REQUIRED" if r["is_required"] else "optional"
            print(f"  {r['id']}  →  {r['name']}  ({r['data_type']}, {req})")

        # 10. Clause→Parameter relationships sample
        print("\n📄 SAMPLE: Clause→Parameter relationships (first 15)")
        print("-" * 40)
        result = session.run("""
            MATCH (c:Clause)-[:CONTAINS_PARAM]->(p:Parameter)
            RETURN c.id AS clause_id, p.id AS param_id, p.name AS param_name
            ORDER BY c.id
            LIMIT 15
        """)
        for r in result:
            print(f"  {r['clause_id']}  →  {r['param_id']} ({r['param_name']})")

        # 11. Overall stats
        print("\n\n📊 OVERALL STATISTICS")
        print("-" * 40)
        result = session.run("""
            MATCH (n) RETURN count(n) AS total_nodes
        """)
        print(f"  Total nodes: {result.single()['total_nodes']}")
        
        result = session.run("""
            MATCH ()-[r]->() RETURN count(r) AS total_rels
        """)
        print(f"  Total relationships: {result.single()['total_rels']}")

    driver.close()
    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    if not NEO4J_URI:
        print("❌ NEO4J_URI not set. Please create a .env file with your Neo4j credentials.")
        print("   See .env.example for the template.")
    else:
        explore_schema()
