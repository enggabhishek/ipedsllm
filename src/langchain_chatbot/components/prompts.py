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
        Finally, Use only tables names and Column names mentioned in:\n\n {context} to create correct SQL Query and pay close attention on which column is in which table. if context contains more than one tables then create a query by performing JOIN operation only using the column unitid for the tables.\
        Follow these Instructions for creating syntactically correct SQL query:\
        - Be sure not to query for columns that do not exist in the tables and use alias only where required.\
        - Whenever asked for Institute Names, return the institute names using column 'instnm' associated with the 'unitid' in the generated query.\
        - Likewise, when asked about the average (AVG function) or ratio, ensure the appropriate aggregation function is used.\
        - Pay close attention to the filtering criteria mentioned in the question and incorporate them using the WHERE clause in your SQL query.\
        - If the question involves multiple conditions, use logical operators such as AND, OR to combine them effectively.\
        - When dealing with date or timestamp columns, use appropriate date functions (e.g., DATE_PART, EXTRACT) for extracting specific parts of the date or performing date arithmetic.\
        - If the question involves grouping of data (e.g., finding totals or averages for different categories), use the GROUP BY clause along with appropriate aggregate functions.\
        - Consider using aliases for tables and columns to improve readability of the query, especially in case of complex joins or subqueries.\
        - If necessary, use subqueries or common table expressions (CTEs) to break down the problem into smaller, more manageable parts."),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}"),
    ]
)

vector_search_template = """Answer the question based only on the following context:
    {context}
    Search for the table descriptions in the context and accordingly search for column names and associated column description. Include only relevant tables and columns which can be used by the downstream Text-to-SQL Agent to create SQL Queries for generating answer.
    Search for any information performing the following tasks:
    1. Table Names
    2. Table Descriptions
    3. Column Names
    4. Column Descriptions
    5. Encoded Values
    Finally, only return table names, column names and Encoded Values only (if availabe).

    Question: {question}
    """

retriever_prompt = ChatPromptTemplate.from_template(vector_search_template)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)
