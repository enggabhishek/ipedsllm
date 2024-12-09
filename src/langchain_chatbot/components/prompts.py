# from examples import get_example_selector
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a PostgreSQL expert. Given an input question, create a syntactically correct PostgreSQL query to run and return ONLY the generated Query and nothing else. Remember NOT include backticks ```sql ``` before and after the created query. Unless otherwise specified, do not return more than \
        {top_k} rows.\n\nHere is the relevant table info: {table_info}\
        Finally, Use only tables names, Column names and Encoded values mentioned in:\n\n {context} to create correct SQL Query and pay close attention on which column is in which table.\
        - If the input variable is requesting a name or count of institutes, location of institutes, always include the 'hd2022' table as a reference. Use the join function with the table except ic2022campuses table specified in the context variable, joining on the unitid column. Whenever the input requests the total count or names of institutes, ensure to include the 'unitid', 'stabbr', 'instnm', and 'city' columns from the hd2022 table.\
        - Do not join 'hd2022' table and 'ic2022campuses' table while creating SQL query\
        - If the input mentions enrollment information, use the 'effy2022' table for creating SQL queries and avoid using the 'gr2022' table.\
        - Always use '=' or 'IN'  operators for the given 'Encoded values' in the 'WHERE' clause condition of generated SQL query.\
        Follow these Instructions for creating syntactically correct SQL query:\
        - If context contains more than one table then create a query by performing JOIN operation only using the column unitid for the tables.\
        - Be sure not to query for columns that do not exist in the tables and use alias only where required.\
        - Whenever asked for Institute Names, return the institute names using column 'instnm' associated with the 'unitid' in the generated query.\
        - Always use the 'Encoded values' specified in the context in the 'WHERE' clause condition of your SQL query.\
        - Likewise, when asked about the average (AVG function) or ratio, ensure the appropriate aggregation function is used.\
        - Pay close attention to the filtering criteria mentioned in the question and incorporate them using the WHERE clause in your SQL query.\
        - If the question involves multiple conditions, use logical operators such as AND, OR to combine them effectively.\
        - If the question involves grouping of data (e.g., finding totals or averages for different categories), use the GROUP BY clause along with appropriate aggregate functions.\
        - Consider using aliases for tables and columns to improve readability of the query, especially in case of complex joins or subqueries.\
        - If necessary, use subqueries or common table expressions (CTEs) to break down the problem into smaller, more manageable parts."),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}"),
    ]
)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

column_desc_retriver_prompt = """
Retrieve the description of a specific column as a string, based on the column name mentioned in the question, using the provided context.

Context:
{context}

- `Column_Description`: A dictionary where keys are column descriptions and values are column names.

Tasks:
Identify one specific description corresponding to the one column name mentioned in the question. Provide a single column description to assist the downstream Text-to-SQL Agent in formulating SQL queries involving JOINs, filtering, and subqueries.
Question:
{question}

Output format:

'Column_Description key'
    """
    
column_name_retriver_prompt = """
Retrieve the specific column names that are relevant to the question variables within the provided context. Use the following details:

Context:
{context}

Details:

- `Column_Description`: This dictionary contains column descriptions as keys and their respective column names as values.

Tasks to accomplish:
1. Identify column names.

Scan the context for column names pertinent to the question. Include the 'unitid' column of the relevant table to aid the downstream Text-to-SQL Agent in constructing SQL queries involving JOINs, filters, and subqueries.

Question: {question}

Output format:

Column names related to the question: [column_name1, column_name2, ...]
    """

encoded_values_retriver_prompt = """Retrieve the specific one encoded value based on the provided context. Answer the question using only the following context:

{context}

Details:
- `Encoded_Values`: A dictionary containing code descriptions as keys and corresponding encoded values as integers, float or string representing the discrete column values.

Tasks to perform:
1. Identify the encoded value.

Search the context for one encoded value that answers the question. Include a specific one encoded value to help the downstream Text-to-SQL Agent in forming SQL queries involving JOINs, filtering, and subqueries. Note that `unitid` is the primary key for JOIN operations.

Question: {question}
Output format:
Encoded_Values:
"""