"""
Test SQL generation for one previously-skipped database (college_2).
"""
import json
from pathlib import Path
import sqlite3
from LLM_model import LLM_model
from SchemaRetriever import SchemaRetriever

# Extract schema from SQLite
def extract_schema_from_sqlite(db_path: Path) -> str:
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name")
        tables = cursor.fetchall()
        conn.close()
        schema_sql = "\n".join([table[0] for table in tables if table[0]])
        return schema_sql if schema_sql else ""
    except Exception as e:
        return ""

# Load a sample from college_2
dataset_path = Path("spider/train_spider.json")
with open(dataset_path) as f:
    data = json.load(f)

# Find a college_2 example
sample = None
for item in data:
    if item.get("db_id") == "college_2":
        sample = item
        break

if not sample:
    print("❌ No college_2 sample found in dataset")
else:
    db_id = "college_2"
    question = sample["question"]
    gold_sql = sample["query"]
    
    print(f"Database: {db_id}")
    print(f"Question: {question}")
    print(f"Gold SQL: {gold_sql}\n")
    
    # Extract schema from SQLite
    db_path = Path(f"spider/database/{db_id}/{db_id}.sqlite")
    schema_sql = extract_schema_from_sqlite(db_path)
    
    if not schema_sql:
        print("❌ Failed to extract schema")
    else:
        print(f"✓ Schema extracted ({len(schema_sql)} chars)\n")
        
        # Create temp schema file
        temp_schema = Path(f"temp_schema_{db_id}.sql")
        temp_schema.write_text(schema_sql)
        
        # Initialize retriever and LLM
        print("Initializing SchemaRetriever...")
        retriever = SchemaRetriever(str(temp_schema))
        retriever.collection_name = db_id
        retriever.collection = retriever.client.get_or_create_collection(
            name=db_id,
            embedding_function=retriever.embed_model
        )
        retriever.store_schema()
        
        print("Initializing LLM model...")
        llm = LLM_model(collection_name=db_id, ollama_model="qwen2.5:7b")
        
        print("Generating SQL...\n")
        result = llm.generate_sql(question)
        
        predicted_sql = result.get("sql_query", "").strip()
        
        print(f"Predicted SQL:\n{predicted_sql}\n")
        print(f"Match with gold: {predicted_sql.lower() == gold_sql.lower()}")
        
        # Cleanup
        temp_schema.unlink()
        print("\n✓ Test completed successfully - schema extraction + generation works!")
