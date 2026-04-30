"""
Test schema extraction from SQLite for databases with missing schema.sql files.
"""
import sqlite3
from pathlib import Path

def extract_schema_from_sqlite(db_path: Path) -> str:
    """Extract schema (CREATE TABLE statements) directly from SQLite database."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get all table creation statements from sqlite_master
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name")
        tables = cursor.fetchall()
        conn.close()
        
        schema_sql = "\n".join([table[0] for table in tables if table[0]])
        return schema_sql if schema_sql else ""
    except Exception as e:
        print(f"Error: {e}")
        return ""

# Test with a previously skipped database
test_dbs = ["college_2", "chinook_1", "twitter_1", "company_1"]

for db_id in test_dbs:
    db_path = Path(f"spider/database/{db_id}/{db_id}.sqlite")
    schema_path = Path(f"spider/database/{db_id}/schema.sql")
    
    print(f"\n{'='*60}")
    print(f"Testing: {db_id}")
    print(f"  SQLite exists: {db_path.exists()}")
    print(f"  schema.sql exists: {schema_path.exists()}")
    
    if db_path.exists():
        schema = extract_schema_from_sqlite(db_path)
        if schema:
            print(f"  ✓ Schema extracted from SQLite ({len(schema)} chars)")
            print(f"  First 150 chars:")
            print(f"    {schema[:150]}...")
        else:
            print(f"  ✗ Failed to extract schema from SQLite")
    else:
        print(f"  ✗ SQLite database not found")
