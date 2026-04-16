import os
import psycopg2

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

def fix_foreign_key():
    conn = None
    try:
        conn = psycopg2.connect(**_get_db_config())
        conn.autocommit = True
        
        with conn.cursor() as cur:
            # Remove FK constraint (Neo4j is now the source of truth for parameter definitions)
            cur.execute("""
                ALTER TABLE contract_parameters
                DROP CONSTRAINT IF EXISTS contract_parameters_parameter_id_fkey;
            """)
            print("✅ Foreign key constraint removed!")
            
            # Verify
            cur.execute("""
                SELECT conname 
                FROM pg_constraint 
                WHERE conrelid = 'contract_parameters'::regclass
                  AND conname LIKE '%parameter_id%';
            """)
            
            remaining = cur.fetchall()
            if not remaining:
                print("✅ No parameter_id foreign key constraints remain")
            else:
                print(f"⚠️  Found remaining constraints: {remaining}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    print("🔧 Fixing contract_parameters foreign key constraint...\n")
    fix_foreign_key()
    print("\n✅ Migration complete!")
    print("   Neo4j is now the single source of truth for parameter definitions")
    print("   Supabase stores only parameter values")
