import os
import psycopg2

DDL = """
-- 1) ENUMS FOR PARAMETER METADATA
DO $$
BEGIN
    -- Data type of the parameter value
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'param_data_type_enum') THEN
        CREATE TYPE param_data_type_enum AS ENUM (
            'string',
            'integer',
            'decimal',
            'date',
            'currency',
            'boolean'
        );
    END IF;

    -- Category group from Sheet 2 (Core, Scope, etc.)
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'param_category_enum') THEN
        CREATE TYPE param_category_enum AS ENUM (
            'core',
            'scope',
            'confidentiality',
            'non_compete',
            'ip',
            'payment',
            'term',
            'termination',
            'liability',
            'sla',
            'dispute_resolution',
            'governing_law',
            'data_protection',
            'other'
        );
    END IF;

    -- Required vs optional
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'param_required_enum') THEN
        CREATE TYPE param_required_enum AS ENUM (
            'required',
            'optional'
        );
    END IF;
END$$;

-- 2) PARAMETER_DEFINITIONS TABLE
CREATE TABLE IF NOT EXISTS parameter_definitions (
    -- P_001, P_002 ... from Sheet 2
    parameter_id         TEXT PRIMARY KEY,      -- e.g. 'P_001'
    
    -- Placeholder name used in clause templates
    parameter_name       TEXT NOT NULL,         -- e.g. '{{PARTY_A_NAME}}' or '[CITY]'
    
    -- Typed metadata
    data_type            param_data_type_enum NOT NULL,
    category             param_category_enum   NOT NULL,
    required_optional    param_required_enum   NOT NULL DEFAULT 'required',
    
    -- Docs
    description          TEXT,
    example_value        TEXT,
    validation_rule      TEXT,                  -- e.g. 'positive_integer', 'email', 'date_yyyy_mm_dd'
    input_format         TEXT,                  -- e.g. 'text', 'textarea', 'select', 'date', 'number'
    
    -- Link to clauses (Neo4j ids) where this parameter is used
    used_in_clauses      TEXT[] NOT NULL DEFAULT '{}',
    
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3) Indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_param_defs_used_in_clauses
    ON parameter_definitions
    USING GIN (used_in_clauses);

CREATE INDEX IF NOT EXISTS idx_param_defs_category
    ON parameter_definitions(category);

CREATE INDEX IF NOT EXISTS idx_param_defs_data_type
    ON parameter_definitions(data_type);
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
        print("✅ parameter_definitions table created with enums + GIN index on used_in_clauses!")
    except Exception as e:
        print("❌ Error:", e)
    finally:
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    create_schema()
