import os
import psycopg2

DDL = """
-- Add is_active column to contract_clauses (idempotent)
ALTER TABLE contract_clauses 
ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT true;

-- Add index for faster filtering
CREATE INDEX IF NOT EXISTS idx_contract_clauses_active 
ON contract_clauses(contract_id, is_active) 
WHERE is_active = true;

-- Add index for clause_type filtering (for variant switching)
CREATE INDEX IF NOT EXISTS idx_contract_clauses_type_variant
ON contract_clauses(contract_id, clause_type, variant);

-- Drop old unique constraint (allows multiple variants per clause_type)
ALTER TABLE contract_clauses
DROP CONSTRAINT IF EXISTS unique_clause_per_contract;

-- Add new unique constraint (clause_id still unique per contract)
ALTER TABLE contract_clauses
ADD CONSTRAINT IF NOT EXISTS unique_clause_id_per_contract 
UNIQUE (contract_id, clause_id);
"""

def _get_db_config():
    config = {
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": os.getenv("DB_NAME", "postgres"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD"),
        "sslmode": os.getenv("DB_SSLMODE", "require"),
    }
    missing = [k for k, v in config.items() if v is None]
    if missing:
        raise EnvironmentError(
            f"Missing required env vars: {', '.join(f'DB_{k.upper()}' for k in missing)}"
        )
    return config

def run_migration():
    conn = None
    try:
        conn = psycopg2.connect(**_get_db_config())
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(DDL)
            print("✅ Migration completed!")
            print("  ✓ Added is_active column")
            print("  ✓ Created indexes")
            print("  ✓ Updated constraints")
    
    except Exception as e:
        print("❌ Error:", e)
    
    finally:
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    run_migration()
