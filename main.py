import os
from SchemaRetriever import SchemaRetriever
from LLM_model import LLM_model
from SQLValidatorAgent import SQLValidatorAgent
import sqlite3
import sqlparse
from dotenv import load_dotenv
import re

load_dotenv()

class DatabaseExecutorAgent:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def execute_query(self, sql_query: str):
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql_query)
            if sql_query.strip().lower().startswith("select"):
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            else:
                self.conn.commit()
                return f"{cursor.rowcount} rows affected."
        except Exception as e:
            return f"Execution error: {str(e)}"

    def close(self):
        self.conn.close()

def clean_for_sqlite(stmt: str) -> str:
    # Remove ENGINE=... and AUTO_INCREMENT=... at the end
    stmt = re.sub(r"ENGINE=\w+", "", stmt, flags=re.IGNORECASE)
    stmt = re.sub(r"AUTO_INCREMENT=\d+", "", stmt, flags=re.IGNORECASE)
    # Remove DEFAULT ... ON UPDATE ...
    stmt = re.sub(r"DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP", "", stmt, flags=re.IGNORECASE)
    # Remove ON UPDATE ...
    stmt = re.sub(r"ON UPDATE [^,\n]+", "", stmt, flags=re.IGNORECASE)
    # Replace int with INTEGER
    stmt = re.sub(r"\bint\b", "INTEGER", stmt, flags=re.IGNORECASE)
    # Replace decimal(...) with REAL
    stmt = re.sub(r"decimal\([^\)]*\)", "REAL", stmt, flags=re.IGNORECASE)
    # Remove COLLATE ...
    stmt = re.sub(r"COLLATE [^ ]+", "", stmt, flags=re.IGNORECASE)
    # Remove CHARACTER SET ...
    stmt = re.sub(r"CHARACTER SET [^ ]+", "", stmt, flags=re.IGNORECASE)
    # Remove COMMENT ...
    stmt = re.sub(r"COMMENT '[^']*'", "", stmt, flags=re.IGNORECASE)
    # Remove KEY/UNIQUE KEY definitions (SQLite uses CREATE INDEX)
    stmt = re.sub(r",?\s*KEY `[^`]+` \([^\)]+\)", "", stmt)
    stmt = re.sub(r",?\s*UNIQUE KEY `[^`]+` \([^\)]+\)", "", stmt)
    # Remove trailing commas before closing parenthesis
    stmt = re.sub(r",\s*\)", ")", stmt)
    # Remove multiple spaces
    stmt = re.sub(r"\s+", " ", stmt)
    return stmt.strip()

def initialize_sqlite_db(sql_file_path: str, db_path: str):
    retriever = SchemaRetriever(sql_file_path)
    with open(sql_file_path, "r") as f:
        sql_content = retriever._preprocess_sql(f.read())
    statements = [str(stmt).strip() for stmt in sqlparse.parse(sql_content) if str(stmt).strip()]
    filtered_statements = []
    for stmt in statements:
        if stmt.lower().startswith("create table"):
            stmt = clean_for_sqlite(stmt)
            filtered_statements.append(stmt)
        elif stmt.lower().startswith("insert into"):
            filtered_statements.append(stmt)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    print("\n--- Executing SQL Statements ---")
    for stmt in filtered_statements:
        print(f"\n[SQL] {stmt[:100]}{'...' if len(stmt) > 100 else ''}")
        try:
            cursor.execute(stmt)
        except Exception as e:
            print(f"[ERROR] {e}")
    conn.commit()
    print("\n--- Tables Present in Database ---")
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        if tables:
            for t in tables:
                print(f"Table: {t[0]}")
        else:
            print("No tables found.")
    except Exception as e:
        print(f"[ERROR] Could not list tables: {e}")
    conn.close()

if __name__ == "__main__":
    sql_file = "ecommerce_schema.sql"
    db_path = "ecommerce.db"
    # 1. Initialize SQLite DB from .sql file
    initialize_sqlite_db(sql_file, db_path)
    # 2. Store schema in ChromaDB
    retriever = SchemaRetriever(sql_file)
    retriever.store_schema()
    # 3. LLM model for SQL generation
    llm = LLM_model(collection_name="ecommerce_schema")
    # 4. SQL Validator
    validator = SQLValidatorAgent()
    # 5. Database Executor
    db_executor = DatabaseExecutorAgent(db_path)

    questions = [
        "List all customers",
        "Show orders for customer with id 5",
        "Find expensive products over $100"
    ]

    for question in questions:
        try:
            print(f"\n🔍 Question: {question}")
            sql = llm.generate_sql(question)
            print(f"🧠 Generated SQL: {sql}")
            validated_sql = validator.validate_sql(question, llm.get_schema(), sql)
            print(f"✅ Final SQL to execute: {validated_sql}")
            result = db_executor.execute_query(validated_sql)
            print(f"📊 Query Result: {result}")
        except Exception as e:
            print(f"🚨 Error: {str(e)}")

    db_executor.close()
