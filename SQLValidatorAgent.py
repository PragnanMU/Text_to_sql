from langchain_ollama import ChatOllama


class SQLValidatorAgent:
    def __init__(self, ollama_model: str = "qwen2.5:7b"):
        self.model = ChatOllama(
            model=ollama_model,
            temperature=0.2,
            num_predict=300,
        )

    def _check_sql(self, question: str, schema: str, sql_query: str) -> str:
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
                    print(f"Attempt {attempt + 1}: SQL is valid.")
                    return sql_query
                if result.startswith("FIXED:"):
                    sql_query = result.replace("FIXED:", "").strip()
                    print(f"Attempt {attempt + 1}: SQL was fixed.")
                else:
                    print(f"Attempt {attempt + 1}: SQL is invalid. Retrying...")
            except Exception as e:
                print(f"Attempt {attempt + 1}: Validation error: {str(e)}")
                if attempt == 2:
                    print("All validation attempts failed. Returning original SQL.")
                    return sql_query

        print("SQL could not be validated after 3 attempts.")
        return sql_query
