"""
Few-shot examples from Spider dataset for improved prompting.
These are real examples that help the model understand the task better.
Focus on common pitfalls: alias usage, aggregates, joins, and schema clarity.
"""

FEWSHOT_EXAMPLES = [
    {
        "question": "How many students are there?",
        "schema": "CREATE TABLE student (student_id int, student_name text, major_id int);",
        "sql": "SELECT COUNT(*) FROM student"
    },
    {
        "question": "What are the names of all students?",
        "schema": "CREATE TABLE student (student_id int, student_name text, major_id int);",
        "sql": "SELECT student_name FROM student"
    },
    {
        "question": "Find students whose names start with 'J'.",
        "schema": "CREATE TABLE student (student_id int, student_name text, major_id int);",
        "sql": "SELECT * FROM student WHERE student_name LIKE 'J%'"
    },
    {
        "question": "What are the distinct majors?",
        "schema": "CREATE TABLE student (student_id int, student_name text, major_id int); CREATE TABLE major (major_id int, major_name text);",
        "sql": "SELECT DISTINCT major_name FROM major"
    },
    {
        "question": "Find students in the Computer Science major.",
        "schema": "CREATE TABLE student (student_id int, student_name text, major_id int); CREATE TABLE major (major_id int, major_name text);",
        "sql": "SELECT student_name FROM student JOIN major ON student.major_id = major.major_id WHERE major.major_name = 'Computer Science'"
    },
    {
        "question": "How many students are in each major?",
        "schema": "CREATE TABLE student (student_id int, student_name text, major_id int); CREATE TABLE major (major_id int, major_name text);",
        "sql": "SELECT major.major_name, COUNT(student.student_id) FROM student JOIN major ON student.major_id = major.major_id GROUP BY major.major_name"
    },
    {
        "question": "What is the average age of students?",
        "schema": "CREATE TABLE student (student_id int, student_name text, age int);",
        "sql": "SELECT AVG(age) FROM student"
    },
    {
        "question": "Find the oldest student.",
        "schema": "CREATE TABLE student (student_id int, student_name text, age int);",
        "sql": "SELECT student_name FROM student WHERE age = (SELECT MAX(age) FROM student)"
    },
    {
        "question": "What is the maximum and minimum budget?",
        "schema": "CREATE TABLE company (company_id int, budget_in_billions decimal);",
        "sql": "SELECT MAX(budget_in_billions), MIN(budget_in_billions) FROM company"
    },
    {
        "question": "Count employees in each department, ordered by count.",
        "schema": "CREATE TABLE department (dept_id int, dept_name text); CREATE TABLE employee (emp_id int, dept_id int);",
        "sql": "SELECT department.dept_name, COUNT(employee.emp_id) AS emp_count FROM department LEFT JOIN employee ON department.dept_id = employee.dept_id GROUP BY department.dept_name ORDER BY emp_count DESC"
    },
    {
        "question": "Find users with average score above 80.",
        "schema": "CREATE TABLE user (user_id int, user_name text); CREATE TABLE score (user_id int, score int);",
        "sql": "SELECT user.user_name FROM user JOIN score ON user.user_id = score.user_id GROUP BY user.user_id HAVING AVG(score.score) > 80"
    },
    {
        "question": "Get all products with price greater than 100.",
        "schema": "CREATE TABLE product (product_id int, product_name text, price decimal);",
        "sql": "SELECT product_name FROM product WHERE price > 100"
    }
]

def format_fewshot_examples(include_count: int = 8) -> str:
    """Format few-shot examples for prompting."""
    examples_to_use = FEWSHOT_EXAMPLES[:include_count]
    formatted = """EXAMPLES OF CORRECT SQL GENERATION:
Follow these patterns when generating SQL. Pay attention to:
- Use table aliases (t1, t2) only when you need them in JOINs
- Always qualify column names with table name in JOINs: table.column
- Use COUNT(*), AVG(), MAX(), MIN() correctly (no double wrapping)
- Use DISTINCT when needed, GROUP BY for aggregations
- Proper WHERE clause syntax with AND/OR

"""
    
    for i, example in enumerate(examples_to_use, 1):
        formatted += f"Example {i}:\n"
        formatted += f"  Schema: {example['schema']}\n"
        formatted += f"  Question: {example['question']}\n"
        formatted += f"  SQL: {example['sql']}\n\n"
    
    return formatted

