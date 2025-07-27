import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

class SQLValidatorAgent:
    def __init__(self):
        """
        Initialize the SQL Validator Agent using Gemini 1.5 Flash.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.generation_config = {
            "temperature": 0.2,
            "max_output_tokens": 300
        }

    def _check_sql(self, question: str, schema: str, sql_query: str) -> str:
        """
        Ask Gemini to validate the SQL query against the question and schema.

        Returns:
            str: One of ["VALID", "INVALID", "FIXED:<corrected_sql>"]
        """
        prompt = f"""
You are a SQL validation agent. Check if the given SQL query is correct for the question, based on the schema.

SCHEMA:
{schema}

QUESTION:
{question}

SQL QUERY:
{sql_query}

RULES:
1. First check if the SQL correctly answers the question based on the schema.
2. If correct, return: VALID
3. If incorrect, and you can fix it, return: FIXED:<corrected_sql>
4. If incorrect and you cannot fix it confidently, return: INVALID
5. Do NOT include explanations.
"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            answer = response.text.strip()

            # Clean markdown/code block if any
            if "```" in answer:
                answer = answer.split("```")[1].strip()

            return answer
        except Exception as e:
            raise RuntimeError(f"Validation failed: {str(e)}")

    def validate_sql(self, question: str, schema_statements: list, sql_query: str) -> str:
        """
        Validate the generated SQL query. Retry up to 3 times if needed.

        Args:
            question (str): Natural language question
            schema_statements (list): List of CREATE TABLE statements
            sql_query (str): SQL query to validate

        Returns:
            str: Final validation result or corrected SQL query
        """
        schema = "\n".join(schema_statements)

        for attempt in range(3):
            result = self._check_sql(question, schema, sql_query)

            if result == "VALID":
                print(f"✅ Attempt {attempt + 1}: SQL is valid.")
                return sql_query
            elif result.startswith("FIXED:"):
                sql_query = result.replace("FIXED:", "").strip()
                print(f"⚠️ Attempt {attempt + 1}: SQL was fixed. New query:\n{sql_query}")
            else:
                print(f"❌ Attempt {attempt + 1}: SQL is invalid. Retrying...")

        print("❌ SQL could not be validated after 3 attempts.")
        return sql_query  # Return best-effort SQL

