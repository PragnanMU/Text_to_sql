from langchain_ollama import ChatOllama
import re


class SQLValidatorAgent:
    def __init__(self, ollama_model: str = "qwen2.5:7b"):
        self.model = ChatOllama(
            model=ollama_model,
            temperature=0.2,
            num_predict=300,
        )

    def _extract_schema_elements(self, schema: str):
        """Extract table names and column names from CREATE TABLE statements."""
        tables = {}
        # Match CREATE TABLE statements
        for match in re.finditer(r'CREATE TABLE\s+(\w+)\s*\((.*?)\);', schema, re.DOTALL | re.IGNORECASE):
            table_name = match.group(1).lower()
            columns_str = match.group(2)
            # Extract column names (first word before type)
            columns = []
            for col_match in re.finditer(r'(\w+)\s+(?:int|text|varchar|datetime|decimal|float|blob|real)', columns_str, re.IGNORECASE):
                columns.append(col_match.group(1).lower())
            tables[table_name] = columns
        return tables

    def _validate_schema_usage(self, sql_query: str, schema_elements: dict) -> (bool, str):
        """Check if SQL query uses only tables/columns from schema."""
        sql_lower = sql_query.lower()
        
        # Extract table references (after FROM, JOIN)
        table_refs = set()
        for match in re.finditer(r'(?:FROM|JOIN)\s+(\w+)', sql_lower):
            table_refs.add(match.group(1))
        
        # Extract column references (simplified)
        col_refs = set()
        for match in re.finditer(r'(\w+)\.(\w+)', sql_lower):
            col_refs.add((match.group(1), match.group(2)))
        
        valid_tables = set(schema_elements.keys())
        
        # Check if all referenced tables exist
        for table_ref in table_refs:
            if table_ref not in valid_tables:
                return False, f"Table '{table_ref}' not found in schema"
        
        # Check if all qualified column references exist
        for table_ref, col_ref in col_refs:
            if table_ref not in valid_tables:
                return False, f"Table '{table_ref}' not found in schema"
            if col_ref not in schema_elements[table_ref]:
                return False, f"Column '{col_ref}' not found in table '{table_ref}'"
        
        return True, "All tables and columns are in schema"

    def _check_sql(self, question: str, schema: str, sql_query: str) -> str:
        # First validate schema usage
        schema_elements = self._extract_schema_elements(schema)
        is_valid, schema_msg = self._validate_schema_usage(sql_query, schema_elements)
        if not is_valid:
            return f"INVALID: {schema_msg}"
        
        prompt = f"""You are a strict SQL validator. Review this SQL query for correctness.

CRITICAL: Only use tables and columns that exist in the schema below.

Database Schema:
{schema}

User Question:
{question}

Generated SQL Query:
{sql_query}

Please respond with exactly one of these options:
- VALID (if the query is correct and uses only schema tables/columns)
- FIXED: [corrected SQL query] (if you can fix it while using ONLY schema elements)
- INVALID (if it references non-existent tables/columns or cannot be fixed)

Do not include any other text or explanations.
"""
        try:
            response = self.model.invoke(prompt)
            answer = str(response.content).strip()

            if not answer:
                return "INVALID"

            if "```" in answer:
                answer = answer.split("```")[1].strip()

            return answer
        except Exception as e:
            print(f"Ollama validation error: {str(e)}")
            return "INVALID"

    def validate_sql(self, question: str, schema_statements: list, sql_query: str) -> str:
        schema = "\n".join(schema_statements)

        for attempt in range(3):
            try:
                result = self._check_sql(question, schema, sql_query)

                if result == "VALID":
                    print(f"Attempt {attempt + 1}: SQL is valid (schema-compliant).")
                    return sql_query
                if result.startswith("FIXED:"):
                    sql_query = result.replace("FIXED:", "").strip()
                    print(f"Attempt {attempt + 1}: SQL was fixed (schema-compliant).")
                elif result.startswith("INVALID:"):
                    # Schema violation detected
                    print(f"Attempt {attempt + 1}: {result}")
                    return sql_query  # Return original or mark as invalid
                else:
                    print(f"Attempt {attempt + 1}: SQL is invalid. Retrying...")
            except Exception as e:
                print(f"Attempt {attempt + 1}: Validation error: {str(e)}")
                if attempt == 2:
                    print("All validation attempts failed. Returning original SQL.")
                    return sql_query

        print("SQL could not be validated after 3 attempts.")
        return sql_query
