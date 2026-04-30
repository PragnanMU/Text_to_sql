"""
Few-shot examples from Spider dataset for improved prompting.
These are real examples that help the model understand the task better.
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
    }
]

def format_fewshot_examples(include_count: int = 5) -> str:
    """Format few-shot examples for prompting."""
    examples_to_use = FEWSHOT_EXAMPLES[:include_count]
    formatted = "EXAMPLES (do this for the question below):\n\n"
    
    for i, example in enumerate(examples_to_use, 1):
        formatted += f"Example {i}:\n"
        formatted += f"  Schema: {example['schema']}\n"
        formatted += f"  Question: {example['question']}\n"
        formatted += f"  SQL: {example['sql']}\n\n"
    
    return formatted
