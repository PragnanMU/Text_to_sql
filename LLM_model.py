import os
import google.generativeai as genai
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from dotenv import load_dotenv
import json

load_dotenv()

class LLM_model:
    def __init__(self, collection_name: str, persist_directory: str = "schema"):
        """
        Initialize the LLM_model with a ChromaDB collection and Gemini API.

        Args:
            collection_name (str): Name of the ChromaDB collection.
            persist_directory (str): Directory to store ChromaDB data (default: 'schema').
        """
        self.collection_name = collection_name
        self.persist_directory = os.path.abspath(persist_directory)
        os.makedirs(self.persist_directory, exist_ok=True)

        # Load Gemini API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        genai.configure(api_key=api_key)

        # Initialize ChromaDB client
        self.client = PersistentClient(
            path=self.persist_directory,
            settings=Settings(allow_reset=True)
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=GoogleGenerativeAiEmbeddingFunction(
                    api_key=api_key,
                    model_name="models/embedding-001"
                )
            )
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=GoogleGenerativeAiEmbeddingFunction(
                    api_key=api_key,
                    model_name="models/embedding-001"
                )
            )

        # Initialize Gemini model
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.generation_config = {
            "temperature": 0.3,
            "max_output_tokens": 500
        }

    def get_schema(self):
        """
        Retrieve schema statements from ChromaDB collection.

        Returns:
            list: List of schema statements (CREATE TABLE statements).
        """
        try:
            results = self.collection.get()
            schema_statements = results.get("documents", [])
            if not schema_statements:
                print(f"⚠️ No schema statements found in collection: '{self.collection_name}'")
            return schema_statements
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve schema: {str(e)}")

    def _clean_sql_response(self, response_text: str) -> str:
        """
        Clean and extract SQL query from Gemini response.

        Args:
            response_text (str): Raw response from Gemini

        Returns:
            str: Cleaned SQL query
        """
        # Remove markdown code blocks if present
        if "```sql" in response_text:
            response_text = response_text.split("```sql")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        # Remove any remaining markdown or unwanted characters
        response_text = response_text.replace("`", "").strip()

        # Remove SQL comments (-- or /* */)
        lines = [line for line in response_text.split('\n')
                if not line.strip().startswith('--')]
        response_text = '\n'.join(lines)
        response_text = response_text.split('/*')[0].strip()

        return response_text

    def generate_sql(self, nlp_question: str):
        """
        Generate a SQL query from an NLP question.

        Args:
            nlp_question (str): Natural language question

        Returns:
            dict: {"sql_query": ..., "summary": ...}
        """
        schema_statements = self.get_schema()
        if not schema_statements:
            raise ValueError("No schema available to generate SQL")

        schema_text = "\n\n".join(schema_statements)

        prompt = f"""You are a SQL expert. Generate a SQL query for this question using ONLY the schema below.

SCHEMA:
{schema_text}

QUESTION: {nlp_question}

genaerate result in json format
result={{
    \"sql_query\": \"sql query\",
    \"summary\": \"summary of the query\"
}}
RULES:
1. Return ONLY the JSON result
2. Do NOT include explanations
3. Use correct table/column names from schema
4. Query must be syntactically valid
5. If question is ambiguous, make reasonable assumptions

JSON RESULT:"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            response_text = response.text.strip()
            # Try to parse as JSON
            try:
                # Remove markdown code block if present
                if response_text.startswith("```json"):
                    response_text = response_text.split("```json")[1].split("```", 1)[0].strip()
                elif response_text.startswith("```"):
                    response_text = response_text.split("```", 1)[1].split("```", 1)[0].strip()
                result = json.loads(response_text)
                if not ("sql_query" in result and "summary" in result):
                    raise ValueError("Missing keys in model response")
                return result
            except Exception:
                # Fallback: treat as plain SQL
                sql_query = self._clean_sql_response(response_text)
                return {"sql_query": sql_query, "summary": "(No summary returned)"}
        except Exception as e:
            raise RuntimeError(f"SQL generation failed: {str(e)}")

    def store_schema(self, schema_statements: list):
        """
        Store schema statements in ChromaDB.

        Args:
            schema_statements (list): List of CREATE TABLE statements
        """
        if not schema_statements:
            raise ValueError("No schema statements provided")

        # Clear existing data
        self.collection.delete(ids=[str(i) for i in range(self.collection.count())])

        # Add new schema
        self.collection.add(
            documents=schema_statements,
            ids=[str(i) for i in range(len(schema_statements))]
        )

        print(f"✅ Stored {len(schema_statements)} schema statements in collection '{self.collection_name}'")
