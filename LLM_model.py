import json
import os
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_ollama import ChatOllama


class LLM_model:
    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "schema",
        ollama_model: str = "qwen2.5:7b",
    ):
        self.collection_name = collection_name
        self.persist_directory = os.path.abspath(persist_directory)
        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = PersistentClient(
            path=self.persist_directory,
            settings=Settings(allow_reset=True),
        )

        # Use default Chroma embedding to avoid pulling torch.
        self.embedding_function = DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )

        self.model = ChatOllama(
            model=ollama_model,
            temperature=0.3,
            num_predict=500,
        )

    def get_schema(self):
        try:
            results = self.collection.get()
            schema_statements = results.get("documents", [])
            if not schema_statements:
                print(f"No schema statements found in collection: '{self.collection_name}'")
            return schema_statements
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve schema: {str(e)}")

    def _clean_sql_response(self, response_text: str) -> str:
        if "```sql" in response_text:
            response_text = response_text.split("```sql")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        response_text = response_text.replace("`", "").strip()

        lines = [
            line for line in response_text.split("\n") if not line.strip().startswith("--")
        ]
        response_text = "\n".join(lines)
        response_text = response_text.split("/*")[0].strip()

        return response_text

    def _extract_json_payload(self, text: str) -> dict:
        cleaned_text = text.strip()

        if "```json" in cleaned_text:
            cleaned_text = cleaned_text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text.split("```", 1)[1].split("```", 1)[0].strip()

        json_start = cleaned_text.find("{")
        json_end = cleaned_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            cleaned_text = cleaned_text[json_start:json_end]

        return json.loads(cleaned_text)

    def generate_sql(self, nlp_question: str):
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
    "sql_query": "sql query",
    "summary": "summary of the query"
}}
RULES:
1. Return ONLY the JSON result
2. Do NOT include explanations
3. Use correct table/column names from schema
4. Query must be syntactically valid
5. If question is ambiguous, make reasonable assumptions

JSON RESULT:"""

        try:
            response = self.model.invoke(prompt)
            response_text = str(response.content).strip()

            try:
                result = self._extract_json_payload(response_text)

                if not isinstance(result, dict):
                    raise ValueError("Response is not a JSON object")
                if "sql_query" not in result:
                    raise ValueError("Missing 'sql_query' key in response")

                sql_query = str(result.get("sql_query", "")).strip()
                if not sql_query:
                    raise ValueError("SQL query is empty")

                summary = str(result.get("summary", "SQL query generated successfully"))
                return {"sql_query": sql_query, "summary": summary}
            except Exception as parse_error:
                print(f"JSON parsing failed, extracting SQL directly: {str(parse_error)}")
                sql_query = self._clean_sql_response(response_text)
                if sql_query:
                    return {
                        "sql_query": sql_query,
                        "summary": "SQL extracted from response",
                    }
                raise ValueError(f"Could not extract SQL from response: {response_text[:200]}")

        except Exception as e:
            raise RuntimeError(f"SQL generation failed: {str(e)}")
