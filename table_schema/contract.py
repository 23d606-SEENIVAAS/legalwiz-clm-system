import os
import psycopg2

DDL = """
-- 1) ENUM TYPES (safe creation)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'contract_type_enum') THEN
        CREATE TYPE contract_type_enum AS ENUM (
            'employment_nda',
            'saas_service_agreement',
            'consulting_service_agreement',
            'software_license_agreement',
            'data_processing_agreement',
            'vendor_agreement',
            'partnership_agreement'
        );
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'contract_status_enum') THEN
        CREATE TYPE contract_status_enum AS ENUM (
            'draft', 'in_review', 'approved', 'signed', 'active', 'terminated'
        );
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'jurisdiction_enum') THEN
        CREATE TYPE jurisdiction_enum AS ENUM ('India', 'USA', 'UK');
    END IF;
END$$;

-- 2) MAIN CONTRACTS TABLE
CREATE TABLE IF NOT EXISTS contracts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title           TEXT NOT NULL CHECK (LENGTH(title) <= 500),
    contract_type   contract_type_enum NOT NULL,
    jurisdiction    jurisdiction_enum NOT NULL DEFAULT 'India'::jurisdiction_enum,
    status          contract_status_enum NOT NULL DEFAULT 'draft'::contract_status_enum,
    created_by      UUID NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    description     TEXT CHECK (description IS NULL OR LENGTH(description) <= 2000),
    tags            TEXT[]
);

-- 3) INDEXES
CREATE INDEX IF NOT EXISTS idx_contracts_created_by ON contracts(created_by);
CREATE INDEX IF NOT EXISTS idx_contracts_status ON contracts(status);
CREATE INDEX IF NOT EXISTS idx_contracts_type ON contracts(contract_type);
CREATE INDEX IF NOT EXISTS idx_contracts_jurisdiction ON contracts(jurisdiction);
CREATE INDEX IF NOT EXISTS idx_contracts_created_at ON contracts(created_at DESC);
"""

def _get_db_config():
    """Build connection config from environment variables."""
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
            f"Missing required environment variables: {', '.join(f'DB_{k.upper()}' for k in missing)}\n"
            "Copy .env.example to .env and fill in your credentials."
        )
    return config

def create_schema():
    conn = None
    try:
        conn = psycopg2.connect(**_get_db_config())
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(DDL)
        print("✅ Contracts table created successfully!")
        print("   🎯 7 contract types (matches API) + Enums + Indexes")
    except Exception as e:
        print("❌ Error:", e)
    finally:
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    create_schema()
