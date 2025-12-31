import google.generativeai as genai
import os
from dotenv import load_dotenv
import re

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

        self.model = genai.GenerativeModel("gemini-2.5-flash-lite")
        self.generation_config = {
            "temperature": 0.2,
            "max_output_tokens": 300
        }
    
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

    def _check_sql(self, question: str, schema: str, sql_query: str) -> str:
        """
        Ask Gemini to validate the SQL query against the question and schema.

        Returns:
            str: One of ["VALID", "INVALID", "FIXED:<corrected_sql>"]
        """
        prompt = f"""
Please review this SQL query for correctness.

Database Schema:
{schema}

User Question:
{question}

Generated SQL Query:
{sql_query}

Please respond with exactly one of these options:
- VALID (if the query is correct)
- FIXED: [corrected SQL query] (if you can fix it)
- INVALID (if it cannot be fixed)

Do not include any other text or explanations.
"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            # Check if response is valid
            if not response.text:
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        if candidate.finish_reason == 2:
                            return "INVALID"  # Content was filtered
                        elif candidate.finish_reason == 3:
                            return "INVALID"  # Safety concerns
                        else:
                            return "INVALID"  # Other finish reasons
                return "INVALID"
            
            answer = response.text.strip()

            # Clean markdown/code block if any
            if "```" in answer:
                answer = answer.split("```")[1].strip()

            return answer
        except Exception as e:
            error_str = str(e)
            
            # Check if it's a quota/rate limit error - skip validation immediately
            if self._is_quota_error(e):
                print(f"⚠️ Quota/Rate limit exceeded during validation. Skipping validation.")
                return "INVALID"
            else:
                # Non-quota error
                print(f"⚠️ Gemini API error during validation: {error_str}")
                return "INVALID"

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
            try:
                result = self._check_sql(question, schema, sql_query)

                if result == "VALID":
                    print(f"✅ Attempt {attempt + 1}: SQL is valid.")
                    return sql_query
                elif result.startswith("FIXED:"):
                    sql_query = result.replace("FIXED:", "").strip()
                    print(f"⚠️ Attempt {attempt + 1}: SQL was fixed. New query:\n{sql_query}")
                else:
                    print(f"❌ Attempt {attempt + 1}: SQL is invalid. Retrying...")
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1}: Validation error: {str(e)}")
                if attempt == 2:  # Last attempt
                    print("❌ All validation attempts failed. Returning original SQL.")
                    return sql_query

        print("❌ SQL could not be validated after 3 attempts.")
        return sql_query  # Return best-effort SQL

