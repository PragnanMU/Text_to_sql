import os
import sqlparse
import re
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

class SchemaRetriever:
    def __init__(self, sql_file_path: str):
        self.sql_file_path = sql_file_path

        # Extract collection name from the file name
        file_name = os.path.basename(sql_file_path)
        self.collection_name = os.path.splitext(file_name)[0]

        # Create schema directory with absolute path
        self.persist_directory = os.path.abspath("schema")
        os.makedirs(self.persist_directory, exist_ok=True)

        # Default embedding avoids the heavy sentence-transformers/torch stack.
        self.embed_model = DefaultEmbeddingFunction()

        # Initialize PersistentClient
        self.client = PersistentClient(
            path=self.persist_directory,
            settings=Settings(allow_reset=True)
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embed_model
        )

        # Verify storage system
        self._verify_storage_system()

    def _verify_storage_system(self):
        """Verify the storage location is writable and being used"""
        test_file = os.path.join(self.persist_directory, "storage_test.tmp")
        try:
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            print(f"✅ Storage directory is writable: {self.persist_directory}")
        except Exception as e:
            print(f"❌ Storage verification failed: {str(e)}")
            raise

    def _preprocess_sql(self, sql_content: str) -> str:
        """
        Preprocess SQL content to make it compatible with SQLite.

        Args:
            sql_content (str): Raw SQL content.

        Returns:
            str: Processed SQL content.
        """
        # Remove MySQL-specific dump headers
        sql_content = re.sub(r'/\*!.*?\*/', '', sql_content, flags=re.DOTALL)
        sql_content = re.sub(r'--.*?(\n|$)', '\n', sql_content, flags=re.MULTILINE)

        # Remove backticks
        sql_content = re.sub(r'`([^`]+)`', r'\1', sql_content)

        # Replace AUTO_INCREMENT with AUTOINCREMENT
        sql_content = re.sub(r'\bAUTO_INCREMENT\b', 'AUTOINCREMENT', sql_content, flags=re.IGNORECASE)

        # Ensure AUTOINCREMENT is only used with INTEGER PRIMARY KEY
        sql_content = re.sub(
            r'(\b\w+\b\s+(?:INT|INTEGER)\s+PRIMARY\s+KEY)\s+AUTOINCREMENT',
            r'\1 AUTOINCREMENT',
            sql_content,
            flags=re.IGNORECASE
        )

        # Remove AUTOINCREMENT if not used with INTEGER PRIMARY KEY
        sql_content = re.sub(
            r'(\b\w+\b\s+(?!INT|INTEGER)[^,]+)\s+AUTOINCREMENT',
            r'\1',
            sql_content,
            flags=re.IGNORECASE
        )

        # Remove ON UPDATE/ON DELETE clauses
        sql_content = re.sub(
            r'\bON\s+(UPDATE|DELETE)\s+\w+(\s+\w+)*',
            '',
            sql_content,
            flags=re.IGNORECASE
        )

        # Convert MySQL-specific data types
        sql_content = re.sub(r'\bVARCHA[R]?\b(\(\d+\))?', 'TEXT', sql_content, flags=re.IGNORECASE)
        sql_content = re.sub(r'\bTIMESTAMP\b', 'TEXT', sql_content, flags=re.IGNORECASE)
        sql_content = re.sub(r'\bDATETIME\b', 'TEXT', sql_content, flags=re.IGNORECASE)
        sql_content = re.sub(r'\bTINYINT\b', 'INTEGER', sql_content, flags=re.IGNORECASE)
        sql_content = re.sub(r'\bBIGINT\b', 'INTEGER', sql_content, flags=re.IGNORECASE)
        sql_content = re.sub(r'\bDECIMAL\(\d+,\d+\)\b', 'REAL', sql_content, flags=re.IGNORECASE)

        # Remove MySQL-specific clauses
        sql_content = re.sub(r'\bENGINE\s*=\s*\w+\b', '', sql_content, flags=re.IGNORECASE)
        sql_content = re.sub(r'\bDEFAULT\s+CHARSET\s*=\s*\w+\b', '', sql_content, flags=re.IGNORECASE)
        sql_content = re.sub(r'\bCOLLATE\s*=\s*\w+\b', '', sql_content, flags=re.IGNORECASE)
        sql_content = re.sub(r'\bCHARACTER\s+SET\s+\w+\b', '', sql_content, flags=re.IGNORECASE)

        # Remove invalid table options (e.g., =8192)
        sql_content = re.sub(r'\)\s*=[\d\w\s]+$', ')', sql_content)

        # Convert UNIQUE KEY to UNIQUE constraint
        sql_content = re.sub(
            r',\s*UNIQUE\s+KEY\s+\w+\s*\((\w+)\)',
            r', UNIQUE (\1)',
            sql_content,
            flags=re.IGNORECASE
        )

        # Remove KEY clauses (indexes will be created separately)
        sql_content = re.sub(
            r',\s*KEY\s+\w+\s*\((\w+)\)\s*(?=(,|\)))',
            '',
            sql_content,
            flags=re.IGNORECASE
        )

        # Remove trailing commas
        sql_content = re.sub(r',(\s*\))', r'\1', sql_content)

        # Normalize whitespace
        sql_content = ' '.join(sql_content.split())

        return sql_content

    def extract_schema_statements(self):
        """Extract CREATE TABLE statements from SQL file"""
        try:
            with open(self.sql_file_path, "r") as f:
                sql_content = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"SQL file not found at: {self.sql_file_path}")

        # Preprocess SQL content
        sql_content = self._preprocess_sql(sql_content)

        parsed = sqlparse.parse(sql_content)
        schema_statements = []

        for stmt in parsed:
            stmt_str = str(stmt).strip()
            if stmt_str and stmt.get_type().lower() == 'create' and 'table' in stmt_str.lower():
                schema_statements.append(stmt_str)

        print(f"\n🔍 Found {len(schema_statements)} CREATE TABLE statements:")
        for i, stmt in enumerate(schema_statements):
            print(f"Statement {i + 1}: {stmt[:50]}...")

        return schema_statements

    def store_schema(self):
        """Store schema statements in ChromaDB"""
        # Check if collection already has data
        if self.collection.count() > 0:
            print(f"\n⚠️ Collection '{self.collection_name}' already exists with {self.collection.count()} items. Skipping storage.")
            return
            
        schema_statements = self.extract_schema_statements()

        # Batch insert
        self.collection.add(
            documents=schema_statements,
            ids=[f"schema_doc_{i}" for i in range(len(schema_statements))]
        )

        print(f"\n✅ Stored {len(schema_statements)} schema statements in:")
        print(f"- Collection: '{self.collection_name}'")
        print(f"- Location: {self.persist_directory}")

        # Verify persistence
        self._verify_data_persistence(len(schema_statements))

    def _verify_data_persistence(self, expected_count: int):
        """Verify the data was actually persisted"""
        actual_count = self.collection.count()
        db_file = os.path.join(self.persist_directory, "chroma.sqlite3")

        print("\n🔍 Storage Integrity Check:")
        print(f"- Expected items: {expected_count}")
        print(f"- Actual items: {actual_count}")

        if os.path.exists(db_file):
            print(f"- Database file exists ({os.path.getsize(db_file)} bytes)")
        else:
            print("- Database file not found!")

        if actual_count != expected_count:
            raise ValueError(f"Storage verification failed: expected {expected_count} items, got {actual_count}")