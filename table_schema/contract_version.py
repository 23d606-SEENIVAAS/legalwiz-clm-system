import os
import psycopg2

DDL = """
-- 1) CONTRACT_VERSIONS TABLE
CREATE TABLE IF NOT EXISTS contract_versions (
    id              SERIAL PRIMARY KEY,

    -- Link to contracts table
    contract_id     UUID NOT NULL
                    REFERENCES contracts(id)
                    ON DELETE CASCADE,

    -- Version tracking
    version_number  INTEGER NOT NULL,
    
    -- Full snapshot
    content         JSONB NOT NULL,

    -- Metadata
    change_summary  TEXT,
    changed_by      UUID,

    -- Timestamps
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Ensure one version number per contract
    CONSTRAINT unique_version_per_contract 
        UNIQUE (contract_id, version_number)
);

-- 2) Indexes
CREATE INDEX IF NOT EXISTS idx_contract_versions_contract
    ON contract_versions(contract_id);

CREATE INDEX IF NOT EXISTS idx_contract_versions_number
    ON contract_versions(contract_id, version_number DESC);

-- 3) Trigger for auto-increment version_number
CREATE OR REPLACE FUNCTION increment_version_number()
RETURNS TRIGGER AS $$
BEGIN
    NEW.version_number := COALESCE(
        (SELECT MAX(version_number) + 1 
         FROM contract_versions 
         WHERE contract_id = NEW.contract_id), 1
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_increment_version ON contract_versions;
CREATE TRIGGER trigger_increment_version
    BEFORE INSERT ON contract_versions
    FOR EACH ROW EXECUTE FUNCTION increment_version_number();
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

def create_schema():
    conn = None
    try:
        conn = psycopg2.connect(**_get_db_config())
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(DDL)
        print("✅ contract_versions table created with auto-increment versions!")
        print("   🔗 FK → contracts(id)")
        print("   📦 JSONB content snapshot")
        print("   ⚙️  Auto version_number trigger")
    except Exception as e:
        print("❌ Error:", e)
    finally:
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    create_schema()
