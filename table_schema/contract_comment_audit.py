import os
import psycopg2

DDL_COMMENTS = """
-- ENUM for comment types
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'comment_type_enum') THEN
        CREATE TYPE comment_type_enum AS ENUM (
            'general', 'suggestion', 'concern', 'approval', 'rejection'
        );
    END IF;
END$$;

-- CONTRACT_COMMENTS table
CREATE TABLE IF NOT EXISTS contract_comments (
    id              SERIAL PRIMARY KEY,
    contract_id     UUID NOT NULL 
                    REFERENCES contracts(id) ON DELETE CASCADE,
    clause_id       INTEGER REFERENCES contract_clauses(id) ON DELETE SET NULL,
    user_id         UUID NOT NULL,
    comment_text    TEXT NOT NULL,
    comment_type    comment_type_enum NOT NULL DEFAULT 'general',
    parent_comment_id INTEGER REFERENCES contract_comments(id) ON DELETE CASCADE,
    is_resolved     BOOLEAN NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_comments_contract ON contract_comments(contract_id);
CREATE INDEX IF NOT EXISTS idx_comments_clause ON contract_comments(clause_id);
CREATE INDEX IF NOT EXISTS idx_comments_thread ON contract_comments(parent_comment_id);
"""

DDL_AUDIT = """
CREATE TABLE IF NOT EXISTS audit_logs (
    id              SERIAL PRIMARY KEY,
    contract_id     UUID REFERENCES contracts(id) ON DELETE SET NULL,
    user_id         UUID,
    action          TEXT NOT NULL,
    entity_type     TEXT NOT NULL,
    entity_id       UUID,
    old_value       JSONB,
    new_value       JSONB,
    ip_address      INET,
    user_agent      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_contract ON audit_logs(contract_id);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at DESC);
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
            cur.execute(DDL_COMMENTS)
            cur.execute(DDL_AUDIT)
        print("✅ contract_comments & audit_logs created!")
    except Exception as e:
        print("❌ Error:", e)
    finally:
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    create_schema()
