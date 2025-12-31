import os
import google.generativeai as genai
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from dotenv import load_dotenv
import json
import re

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
        embedding_function = GoogleGenerativeAiEmbeddingFunction(
            api_key=api_key,
            model_name="models/embedding-001"
        )
        
        # Check if collection exists first to avoid "already exists" error
        try:
            # Try to get existing collection first
            self.collection = self.client.get_collection(name=self.collection_name)
        except Exception:
            # Collection doesn't exist, create it with the specified embedding function
            try:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=embedding_function
                )
            except Exception as e:
                # If creation fails (e.g., collection was created between check and create),
                # try get_or_create_collection as fallback
                if "already exists" in str(e).lower():
                    self.collection = self.client.get_collection(name=self.collection_name)
                else:
                    raise RuntimeError(f"Failed to create collection '{self.collection_name}': {str(e)}")

        # Initialize Gemini model
        self.model = genai.GenerativeModel("gemini-2.5-flash-lite")
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

    def _extract_retry_delay(self, error_message: str) -> float:
        """
        Extract retry delay from Gemini API error message.
        
        Args:
            error_message (str): Error message from API
            
        Returns:
            float: Retry delay in seconds, or default 5.0 if not found
        """
        # Look for "Please retry in X.XXXXs" pattern (handles decimal seconds)
        match = re.search(r'Please retry in ([\d.]+)s', error_message)
        if match:
            return float(match.group(1))
        
        # Look for retry_delay { seconds: X } pattern (protobuf format)
        match = re.search(r'retry_delay\s*\{[^}]*seconds[:\s]+(\d+)', error_message, re.DOTALL)
        if match:
            return float(match.group(1))
        
        # Look for "retry_delay { seconds: X }" in a simpler format
        match = re.search(r'seconds[:\s]+(\d+)\s*\}', error_message)
        if match:
            return float(match.group(1))
        
        # Default retry delay
        return 5.0
    
    def _is_quota_error(self, error: Exception) -> bool:
        """
        Check if the error is a quota/rate limit error.
        
        Args:
            error (Exception): The exception to check
            
        Returns:
            bool: True if it's a quota/rate limit error
        """
        error_str = str(error).lower()
        return (
            "429" in error_str or
            "quota" in error_str or
            "rate limit" in error_str or
            "rate-limit" in error_str or
            "exceeded" in error_str
        )

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

Generate result in JSON format
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
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```", 1)[0].strip()
                elif response_text.startswith("```"):
                    response_text = response_text.split("```", 1)[1].split("```", 1)[0].strip()
                
                # Try to find JSON object in the response
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    response_text = response_text[json_start:json_end]
                
                result = json.loads(response_text)
                
                # Validate required keys
                if not isinstance(result, dict):
                    raise ValueError("Response is not a JSON object")
                
                if "sql_query" not in result:
                    raise ValueError("Missing 'sql_query' key in response")
                
                # Ensure sql_query is a string
                sql_query = str(result.get("sql_query", "")).strip()
                if not sql_query:
                    raise ValueError("SQL query is empty")
                
                # Get summary or use default
                summary = result.get("summary", "SQL query generated successfully")
                
                return {"sql_query": sql_query, "summary": summary}
                
            except json.JSONDecodeError as je:
                # If JSON parsing fails, try to extract SQL from the response
                print(f"⚠️ JSON parsing failed, attempting to extract SQL directly: {str(je)}")
                sql_query = self._clean_sql_response(response_text)
                if sql_query:
                    return {"sql_query": sql_query, "summary": "SQL extracted from response (JSON parsing failed)"}
                else:
                    raise ValueError(f"Could not extract SQL from response: {response_text[:200]}")
            except Exception as parse_error:
                # Other parsing errors
                print(f"⚠️ Error parsing response: {str(parse_error)}")
                sql_query = self._clean_sql_response(response_text)
                if sql_query:
                    return {"sql_query": sql_query, "summary": "SQL extracted from response"}
                raise
                
        except Exception as e:
            error_str = str(e)
            
            # Check if it's a quota/rate limit error - fail immediately without retries
            if self._is_quota_error(e):
                raise RuntimeError(
                    f"SQL generation failed: Quota/Rate limit exceeded. "
                    f"Please check your Gemini API plan and billing details.\n\n"
                    f"Error: {error_str[:500]}\n\n"
                    f"For more information: https://ai.google.dev/gemini-api/docs/rate-limits"
                )
            else:
                # Non-quota error
                raise RuntimeError(f"SQL generation failed: {error_str}")

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
