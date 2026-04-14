import os
import psycopg2

DDL = """
-- Full reset: DROP and recreate with corrected 7 contract types
-- WARNING: This drops and recreates contracts table. Use only in dev/initial setup.
DROP TYPE IF EXISTS contract_type_enum CASCADE;
DROP TYPE IF EXISTS contract_status_enum CASCADE;
DROP TYPE IF EXISTS jurisdiction_enum CASCADE;

CREATE TYPE contract_type_enum AS ENUM (
    'employment_nda',
    'saas_service_agreement',
    'consulting_service_agreement',
    'software_license_agreement',
    'data_processing_agreement',
    'vendor_agreement',
    'partnership_agreement'
);

CREATE TYPE contract_status_enum AS ENUM (
    'draft', 'in_review', 'approved', 'signed', 'active', 'terminated'
);

CREATE TYPE jurisdiction_enum AS ENUM ('India', 'USA', 'UK');

-- CONTRACTS TABLE (no auth.users FK for standalone deployment)
DROP TABLE IF EXISTS contracts CASCADE;

CREATE TABLE contracts (
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

CREATE INDEX idx_contracts_created_by ON contracts(created_by);
CREATE INDEX idx_contracts_status ON contracts(status);
CREATE INDEX idx_contracts_type ON contracts(contract_type);
CREATE INDEX idx_contracts_created_at ON contracts(created_at DESC);

SELECT '✅ Contracts table recreated with 7 correct contract types!' as status;
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
        print("✅ contracts table reset with 7 correct types!")
        print("   🎯 Matches API: employment_nda, saas_service_agreement, etc.")
        print("   🔓 No auth.users dependency")
        print("   ⚠️  All dependent tables were CASCADE dropped — re-run other migrations")
    except Exception as e:
        print("❌ Error:", e)
    finally:
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    print("⚠️  WARNING: This will DROP and recreate the contracts table and all dependents!")
    confirm = input("Type 'yes' to confirm: ")
    if confirm.strip().lower() == "yes":
        create_schema()
    else:
        print("Aborted.")
