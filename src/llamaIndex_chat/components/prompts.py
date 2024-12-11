from llama_index.core import PromptTemplate

template = (
    """Given an input question, first create a syntactically correct PostgreSQL query to run. For Example,

If the input question is List the names of colleges in Massachusetts", the query would be "SELECT INSTNM FROM public.\"HD2022\" WHERE STABBR = 'MA';"

If the input question is Total number of institutions in each state", the query would be "SELECT STABBR, COUNT(UNITID) AS TotalInstitutions FROM public.\"HD2022\" GROUP BY STABBR ORDER BY TotalInstitutions DESC;"

If the input Question is Institutes which require Secondary School GPA for getting admission in Undergrad program", the query would be  "SELECT IC.campusid, IC.pcaddr, IC.pccity FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.admcon1 = 1;"

Do not include "" at the start and end of the query. Then look at the results of the query and rather than a few results, return all the results. You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a few relevant columns given the question. DO NOT MAKE ANY DML QUERIES (INSERT, UPDATE, DELETE).

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Pay attention to which column is in which table. Also, qualify column names with the table name when needed. You are required to use the following format, each taking one line:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use tables listed below.
{schema}

Question: {query_str}
SQLQuery: """
)
text2sql_prompt = PromptTemplate(template)

response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n"
    "SQL: {sql_query}\n"
    "SQL Response: {context_str}\n"
    "Response: "
)
response_synthesis_prompt = PromptTemplate(
    response_synthesis_prompt_str,
)
