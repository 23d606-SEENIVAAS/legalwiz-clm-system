import os
import psycopg2

DDL = """
-- 1) CONTRACT_PARAMETERS TABLE
CREATE TABLE IF NOT EXISTS contract_parameters (
    id              SERIAL PRIMARY KEY,

    -- Keys
    contract_id     UUID NOT NULL
                    REFERENCES contracts(id)
                    ON DELETE CASCADE,

    parameter_id    TEXT NOT NULL,

    -- Polymorphic value storage
    value_text      TEXT,
    value_integer   INTEGER,
    value_decimal   NUMERIC(20,4),
    value_date      DATE,
    value_currency  JSONB,      -- e.g. {"amount": 500000, "currency": "INR"}

    -- Metadata
    provided_by     UUID,       -- auth.users(id) if you want FK later
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- One value per parameter per contract
    CONSTRAINT unique_contract_parameter UNIQUE (contract_id, parameter_id)
);

-- 2) Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_contract_params_contract
    ON contract_parameters(contract_id);

CREATE INDEX IF NOT EXISTS idx_contract_params_param
    ON contract_parameters(parameter_id);
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
        print("✅ contract_parameters table created with FK → contracts!")
        print("   🎯 UNIQUE(contract_id, parameter_id) enforced")
    except Exception as e:
        print("❌ Error:", e)
    finally:
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    create_schema()
