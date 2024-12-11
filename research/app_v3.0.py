from llama_index.core import SQLDatabase, VectorStoreIndex
import sys
import logging
import nest_asyncio
import os
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
    CustomQueryComponent,
)
from llama_index.core.llms import ChatResponse
from llama_index.core import PromptTemplate
from llama_index.core.query_pipeline import FnComponent
from typing import List
from llama_index.core.retrievers import SQLRetriever
import streamlit as st
import openai
from dotenv import load_dotenv
from IPython.display import Markdown, display
from sqlalchemy import (
    create_engine,
    MetaData,
    select,
)
from llama_index.llms.openai import OpenAI
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)

# from research.prefix import (
#     ADM2022_prefix,
#     HD2022_prefix,
# )

load_dotenv()


# Access environment variables
openai_key = os.getenv("API_KEY")
db_url = os.getenv("DB_URL")


nest_asyncio.apply()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# Creating a SQLAlchemy engine
engine = create_engine(db_url)

# Creating a session
Session = sessionmaker(bind=engine)
session = Session()

# Connecting and creating a cursor
connection = engine.connect()
cursor = connection.connection.cursor()

# Creating metadata object
metadata_obj = MetaData()


sql_database = SQLDatabase(engine)

ADM2022_prefix = """This table contains information about the undergraduate selection process for entering first-time, degree/certificate-seeking students. This includes information about admission considerations, applicants, applicants that were admitted, and admitted students who enrolled. SAT and ACT test scores are included for institutions that require or consider test scores for admission. These data are applicable for institutions that do not have an open admissions policy for entering first-time students."""
C2022_A_prefix = """This table contains the number of awards by type of program, level of award (certificate or degree), first or second major, and by race/ethnicity and gender. Data covers all awards granted between July 1, 2021, and June 30, 2022. Each record is uniquely defined by the variables IPEDS ID (UNITID), classification of instructional program (CIPCODE), first or second major (MAJORNUM), and award level (AWLEVEL). Each record will contain the total awards, awards for men and women, and the total awards and awards for men and women for all nine race/ethnicity categories."""
C2022_B_prefix = """This table contains the number of students who completed any degree or certificate by race/ethnicity and gender."""
C2022_C_prefix = """This table contains the number of students receiving a degree or certificate by the level of award and by race/ethnicity, gender, and age categories. Data covers awards granted between July 1, 2021, and June 30, 2022. This file contains multiple records per institution. Each record is uniquely defined by the variables IPEDS ID (UNITID) and award level (AWLEVELC). Each record will contain the total number of students receiving awards, number of men and women receiving awards, number of students receiving awards for all nine race/ethnicity categories; number of students receiving awards are also available for the following age groups: under 18, 18-24, 25-39, 40 and above."""
C2022_DEP_prefix = """This table contains the number of programs offered by type of program, level of award (certificate or degree), and distance education status. Type of program is categorized according to the 2020 Classification of Instructional Programs (CIP), a detailed coding system for postsecondary instructional programs. Beginning in 2019-20, the less than 1-year certificate award level is divided into the following two award levels: certificates of less than 12 weeks and certificates of at least 12 weeks but less than 1 year. More detailed information on distance education was also added in 2019-20. This file contains multiple records per institution."""
EFFY_2022_prefix = """This table contains the unduplicated headcount of students enrolled over the 12-month period (July 1, 2021 - June 30, 2022) for both undergraduate and graduate levels. Beginning with the 2019-20 data collected in 2020-21, undergraduate level headcounts are available by attendance status (full- and part-time) for both degree/certificate-seeking and nondegree/certificate-seeking students. Degree/certificate-seeking headcounts are further disaggregated by first-time, transfer-ins, and continuing student categories. These enrollment data are particularly valuable for institutions that use non-traditional calendar systems and offer short-term programs. Because this enrollment measure encompasses an entire year, it provides a more complete picture of the number of students these schools serve."""
EFFY2022_DIST_prefix = """This table contains the unduplicated headcount of students enrolled over the 12-month period (July 1, 2021 - June 30, 2022) by distance education status and level of student."""
EFIA2022_prefix = """This table contains data on instructional activity measured in total credit and/or contact hours delivered by institutions."""
FLAGS2022_prefix = """This table contains response status information for every institution in the 2022-23 IPEDS universe. This file will determine institutions that have responded; institutions that did not respond and have imputed data; and survey applicability. It will also identify institutions whose data represents multiple campuses (parent/child reporting)."""
GR200_22_prefix = """This table contains the graduation rate status as of August 31, 2022, for the cohort of full-time, first-time degree/certificate-seeking undergraduates."""
GR2022_prefix = """This table contains the graduation rate status as of August 31, 2022, for the cohort of full-time, first-time degree/certificate-seeking undergraduates in both four-year and two-year institutions."""
GR2022_PEL_SSL_prefix = """This table contains the graduation rate status as of August 31, 2022, for three subcohorts of full-time, first-time degree/certificate-seeking undergraduates. The three subcohorts are students who received a Pell grant; students who received a Direct Subsidized loan and did "NOT" receive a Pell grant; and students who did not receive either a Pell grant or Direct Subsidized loan. In four-year institutions each of the subcohorts will include the number of bachelor degree-seeking students who were enrolled in 2016, the number of bachelor degree-seeking students who completed any degree/certificate within 150 percent of normal time, the number who completed a bachelor's degree within 150 percent of normal time. Data for students seeking a degree/certificate other than a bachelor's degree are also included for four-year institutions. Data for two-year and less-than 2-year institutions include the number of full-time, first-time students who were enrolled in 2019, the number of students who completed any degree/certificate 150 percent of normal time."""
HD2022_prefix = """This table contains directory information for every institution in the 2022-23 IPEDS universe. Includes name, address, city, state, zip code and various URL links to the institution's home page, admissions, financial aid offices, and the net price calculator. Identifies institutions as currently active, and institutions that participate in Title IV federal financial aid programs for which IPEDS is mandatory."""
IC2022_prefix = """This table contains data on program and award level offerings, control and affiliation of the institution. It also contains information on special learning opportunities, student services, disability services, tuition plans, and athletic conference associations. Services and programs for service members and veterans are also included."""
IC2022_AY_prefix = """This table contains data on student charges for a full academic year. Institutions that offer primarily academic programs measured in credit hours have a predominant calendar system of semester, quarter, trimester, or 4-1-4 or other academic calendar system report student charges for the full academic year. Student charges on this data file include: average tuition and required fees for both full-time undergraduate and graduate students; per credit hour charges for both part-time undergraduate and graduate students."""
IC2022_CAMPUSES_prefix = """This table contains data for branch campus locations from the 2022-23 IPEDS universe that were listed in College Navigator. This is not a comprehensive list of all Title IV branch campus locations in the U.S. and other jurisdictions. Main campus institutions that report to IPEDS have the option to list their branch campus locations in College Navigator. In the 2022-23 data collection, 191 main campus institutions reported a total of 504 branch locations. The number of provisionally released branch"""
IC2022_PY_prefix = """This table contains data on student charges by program. Institutions that measure programs in clock hours or have primarily occupational programs measured in credit hours report student charges data by the full length PROGRAM. This file contains the number of programs offered and the price of attendance for entering students that are made available to the public on College Navigator; Price of attendance includes amounts for published tuition and required fees, books and supplies, room and board and other expenses for the largest program. Estimates for books and supplies, room and board and other expenses are those from the Cost of Attendance report used by the financial aid office in determining financial need."""
OM2022_prefix = """This table contains award and enrollment data from degree-granting institutions on four cohorts and eight subcohorts of undergraduates who entered an institution in 2014-15 at three points in time: four-year (August 31, 2018) six-year (August 31, 2020) and eight-year (August 31, 2022)."""
SFA2122_prefix = """This table contains data on the number of full-time, first-time degree/certificate-seeking undergraduate students and all undergraduate students who were awarded different types of student financial aid, including grants and loans, from different sources at each institution."""
SFAV2122_prefix = """This table contains the number of students, who received either or both Post-9/11 GI Benefits or the Department of Defense Tuition Assistance through the institution. Eligible dependents receiving such benefits are included. Students who received both benefits, are counted in both programs. Total dollar amounts received by students are included."""


# set Logging to DEBUG for more detailed outputs
table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [  # list of SQLTableSchema objects
    SQLTableSchema(table_name="ADM2022", context_str=ADM2022_prefix),
    SQLTableSchema(table_name="HD2022", context_str=HD2022_prefix),
    SQLTableSchema(table_name="C2022DEP", context_str=C2022_DEP_prefix),
    SQLTableSchema(table_name="C2022_A", context_str=C2022_A_prefix),
    SQLTableSchema(table_name="C2022_B", context_str=C2022_B_prefix),
    SQLTableSchema(table_name="C2022_C", context_str=C2022_C_prefix),
    SQLTableSchema(table_name="EFFY2022", context_str=EFFY_2022_prefix),
    SQLTableSchema(table_name="EFFY2022_DIST",
                   context_str=EFFY2022_DIST_prefix),
    SQLTableSchema(table_name="EFIA2022", context_str=EFIA2022_prefix),
    SQLTableSchema(table_name="FLAGS2022", context_str=FLAGS2022_prefix),
    SQLTableSchema(table_name="GR200_22", context_str=GR200_22_prefix),
    SQLTableSchema(table_name="GR2022", context_str=GR2022_prefix),
    SQLTableSchema(table_name="GR2022_L2", context_str=GR2022_PEL_SSL_prefix),
    SQLTableSchema(table_name="GR2022_PELL_SSL", context_str=GR2022_prefix),

    SQLTableSchema(table_name="IC2022", context_str=IC2022_prefix),
    SQLTableSchema(table_name="IC2022_AY", context_str=IC2022_AY_prefix),
    SQLTableSchema(table_name="IC2022_CAMPUSES",
                   context_str=IC2022_CAMPUSES_prefix),
    SQLTableSchema(table_name="IC2022_PY", context_str=IC2022_PY_prefix),
    SQLTableSchema(table_name="OM2022", context_str=OM2022_prefix),
    SQLTableSchema(table_name="SFA2122", context_str=SFA2122_prefix),
    SQLTableSchema(table_name="SFAV2122", context_str=SFAV2122_prefix),
]  # add a SQLTableSchema for each table

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)

sql_retriever = SQLRetriever(sql_database)


def get_table_context_str(table_schema_objs: List[SQLTableSchema]):
    """Get table context string."""
    context_strs = []
    for table_schema_obj in table_schema_objs:
        table_info = sql_database.get_single_table_info(
            table_schema_obj.table_name
        )
        if table_schema_obj.context_str:
            table_opt_context = " The table description is: "
            table_opt_context += table_schema_obj.context_str
            table_info += table_opt_context

        context_strs.append(table_info)
    return "\n\n".join(context_strs)


table_parser_component = FnComponent(fn=get_table_context_str)


def parse_response_to_sql(response: ChatResponse) -> str:
    """Parse response to SQL."""
    response = response.message.content
    sql_query_start = response.find("SQLQuery:")
    if sql_query_start != -1:
        response = response[sql_query_start:]
        # TODO: move to removeprefix after Python 3.9+
        if response.startswith("SQLQuery:"):
            response = response[len("SQLQuery:"):]
    sql_result_start = response.find("SQLResult:")
    if sql_result_start != -1:
        response = response[:sql_result_start]
    return response.strip().strip("```").strip()


sql_parser_component = FnComponent(fn=parse_response_to_sql)

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
llm = OpenAI(model="gpt-3.5-turbo-1106")


qp = QP(
    modules={
        "input": InputComponent(),
        "table_retriever": obj_retriever,
        "table_output_parser": table_parser_component,
        "text2sql_prompt": text2sql_prompt,
        "text2sql_llm": llm,
        "sql_output_parser": sql_parser_component,
        "sql_retriever": sql_retriever,
        "response_synthesis_prompt": response_synthesis_prompt,
        "response_synthesis_llm": llm,
    },
    verbose=True,
)

qp.add_chain(["input", "table_retriever", "table_output_parser"])
qp.add_link("input", "text2sql_prompt", dest_key="query_str")
qp.add_link("table_output_parser", "text2sql_prompt", dest_key="schema")
qp.add_chain(
    ["text2sql_prompt", "text2sql_llm", "sql_output_parser", "sql_retriever"]
)
qp.add_link(
    "sql_output_parser", "response_synthesis_prompt", dest_key="sql_query"
)
qp.add_link(
    "sql_retriever", "response_synthesis_prompt", dest_key="context_str"
)
qp.add_link("input", "response_synthesis_prompt", dest_key="query_str")
qp.add_link("response_synthesis_prompt", "response_synthesis_llm")


def main():
    st.set_page_config(page_title="University Explorer Bot",
                       page_icon=':llama:',
                       layout='centered',
                       initial_sidebar_state='collapsed')

    st.header("University Explorer ChatBot")

    input_text = st.text_input("Search the Information you want")

    submit = st.button("Generate")

    if submit:
        response = qp.run(query=input_text)
        st.write(response.message.content)


if __name__ == "__main__":
    main()
