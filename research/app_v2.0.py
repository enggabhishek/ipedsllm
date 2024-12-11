# Importing required libraries
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import List
from operator import itemgetter
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from dotenv import load_dotenv
load_dotenv()

# Access environment variables
openai_api_key = os.getenv("API_KEY")
db_url = os.getenv("DB_URL")

# Setup PostgreSQL database using SQLDatabaseToolkit
db = SQLDatabase.from_uri(
    db_url
)


llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)


class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in SQL database.")


table_names = "\n".join(db.get_usable_table_names())
system = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:

{table_names}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""
table_chain = create_extraction_chain_pydantic(
    Table, llm, system_message=system)

system = f"""Return the names of the SQL tables that are relevant to the user question. \
The tables are:
Admissions
Degrees Awarded
Programs Offered
Enrollment
Instructional Activity
Institutaional Characteristics
Student Financial Aid
Graduation Rates"""


category_chain = create_extraction_chain_pydantic(
    Table, llm, system_message=system)


def get_tables(categories: List[Table]) -> List[str]:
    tables = []
    for category in categories:
        if category.name == "Admissions":
            tables.extend(
                [
                    "ADM2022",
                    "IC2022_CAMPUSES",
                ]
            )
        elif category.name == "Degrees Awarded":
            tables.extend(
                [
                    "C2022_A",
                    "C2022_B",
                    "C2022_C",
                ]
            )
        elif category.name == "Enrollment":
            tables.extend(
                [
                    "EFFY2022",
                    "EFFY2022_DIST",
                ]
            )
        elif category.name == "Programs Offered":
            tables.extend([
                "C2022DEP",
                "IC2022",
            ])
        elif category.name == "Instructional Activity":
            tables.extend(
                [

                    "EFIA2022",
                    "IC2022",
                ]
            )
        elif category.name == "Institutional Characteristics":
            tables.extend(
                [
                    "FLAGS2022",
                    "HD2022",
                    "IC2022",
                    "IC2022_CAMPUSES",
                    "IC2022_PY",
                ]
            )
        elif category.name == "Student Financial Aid":
            tables.extend(
                [
                    "SFA2122",
                    "SFAV2122",
                ]
            )
        elif category.name == "Graduation Rates":
            tables.extend(
                [
                    "GR2022",
                    "GR200_22",
                    "GR2022_PELL_SSL",
                ]
            )
    return tables


table_chain = category_chain | get_tables  # noqa


# Implementing dynamic few-shot prompt
examples = [
    {
        "input": "List the names of colleges in Massachusetts:",
        "query": "SELECT INSTNM FROM public.\"HD2022\" WHERE STABBR = 'MA';"
    },
    {
        "input": "List the colleges in Massachusetts that offer graduate programs:",
        "query": "SELECT INSTNM FROM public.\"HD2022\" WHERE STABBR = 'MA' AND GROFFER = 1;"
    },
    {
        "input": "List the colleges in Massachusetts with their longitude and latitude coordinates:",
        "query": "SELECT INSTNM, LONGITUD, LATITUDE FROM public.\"HD2022\" WHERE STABBR = 'MA';"
    },
    {
        "input": "List the colleges in Massachusetts that are located in urban areas (LOCALE = 11):",
        "query": "SELECT INSTNM FROM public.\"HD2022\" WHERE STABBR = 'MA' AND LOCALE = 11;"
    },
    {
        "input": "List the colleges in Massachusetts along with their control type (e.g., Public, Private Nonprofit, Private For-Profit):",
        "query": "SELECT INSTNM, CASE CONTROL WHEN 1 THEN 'Public' WHEN 2 THEN 'Private Nonprofit' WHEN 3 THEN 'Private For-Profit' END AS CONTROL_TYPE FROM public.\"HD2022\" WHERE STABBR = 'MA';"
    },
    {
        "input": "List the colleges in Massachusetts along with their median ACT score:",
        "query": "SELECT INSTNM, ACT FROM public.\"HD2022\" WHERE STABBR = 'MA';"
    },
    {
        "input": "List the colleges in Massachusetts along with their degree levels offered:",
        "query": "SELECT INSTNM, CASE WHEN HLOFFER = 1 THEN 'Certificate' WHEN UGOFFER = 1 THEN 'Associate' WHEN GROFFER = 1 THEN 'Bachelor' WHEN HDEGOFR1 = 1 THEN 'Graduate' ELSE 'Unknown' END AS DEGREE_LEVELS_OFFERED FROM public.\"HD2022\" WHERE STABBR = 'MA';"
    },
    {
        "input": "Count the number of colleges in Massachusetts:",
        "query": "SELECT COUNT(*) FROM public.\"HD2022\" WHERE STABBR = 'MA';"
    },
    {
        "input": "List the colleges in Massachusetts along with their city and address:",
        "query": "SELECT INSTNM, CITY, ADDR FROM public.\"HD2022\" WHERE STABBR = 'MA';"
    },
    {
        "input": "List the colleges in Massachusetts along with their websites:",
        "query": "SELECT INSTNM, WEBADDR FROM public.\"HD2022\" WHERE STABBR = 'MA';"
    },
    {
        "input": "List the colleges in Massachusetts offering undergraduate programs (ICLEVEL = 1):",
        "query": "SELECT INSTNM FROM public.\"HD2022\" WHERE STABBR = 'MA' AND ICLEVEL = 1;"
    },
    {
        "input": "List the colleges in Massachusetts in rural areas (LOCALE = 41 for rural areas according to the IPEDS locale codes):",
        "query": "SELECT INSTNM FROM public.\"HD2022\" WHERE STABBR = 'MA' AND LOCALE = 41;"
    },
    {
        "input": "List the names and addresses of institutions in Massachusetts that offer undergraduate programs:",
        "query": "SELECT INSTNM, ADDR, CITY FROM public.\"HD2022\" WHERE STABBR = 'MA' AND UGOFFER = 1;"
    },
    {
        "input": "Total number of institutions in each state:",
        "query": "SELECT STABBR, COUNT(UNITID) AS TotalInstitutions FROM public.\"HD2022\" GROUP BY STABBR ORDER BY TotalInstitutions DESC;"
    },
    {
        "input": "List the institutions with the highest undergraduate offering in each Carnegie Classification 2021 category:",
        "query": "SELECT C21BASIC, MAX(UGOFFER) AS MaxUndergraduateOffer FROM public.\"HD2022\" GROUP BY C21BASIC;"
    },
    {
        "input": "Names and website addresses of institutions that have a medical program:",
        "query": "SELECT INSTNM, WEBADDR FROM public.\"HD2022\" WHERE MEDICAL = 1;"
    },
    {
        "input": "List the institutions that are Historically Black Colleges or Universities (HBCU) and offer graduate programs:",
        "query": "SELECT INSTNM FROM public.\"HD2022\" WHERE HBCU = 1 AND GROFFER = 1;"
    },
    {
        "input": "Names and addresses of institutions with a hospital:",
        "query": "SELECT INSTNM, ADDR, CITY FROM public.\"HD2022\" WHERE HOSPITAL = 1;"
    },
    {
        "input": "Institutions that have a medical program and are located in urban areas:",
        "query": "SELECT INSTNM, MEDICAL FROM public.\"HD2022\" WHERE LOCALE = 11 AND MEDICAL = 1;"
    },
    {
        "input": "Names and website addresses of institutions that are Land Grant Institutions:",
        "query": "SELECT INSTNM, WEBADDR FROM public.\"HD2022\" WHERE LANDGRNT = 1;"
    },
    {
        "input": "Institutions that offer undergraduate programs in a rural setting and have a medical program:",
        "query": "SELECT INSTNM, ADDR, STABBR, UGOFFER, MEDICAL FROM public.\"HD2022\" WHERE LOCALE = 33 AND UGOFFER = 1 AND MEDICAL = 1;"
    },
    {
        "input": "Names and locations of institutions that have a medical program and offer graduate programs:",
        "query": "SELECT INSTNM, ADDR, STABBR, UGOFFER, MEDICAL FROM public.\"HD2022\" WHERE MEDICAL = 1 AND GROFFER = 1;"
    },
    {
        "input": "List the names of institutions that offer undergraduate programs and are tribal colleges:",
        "query": "SELECT INSTNM FROM public.\"HD2022\" WHERE UGOFFER = 1 AND TRIBAL = 1;"
    },
    {
        "input": "Institutions that offer both undergraduate and graduate programs:",
        "query": "SELECT INSTNM, UGOFFER, GROFFER FROM public.\"HD2022\" WHERE UGOFFER = 1 AND GROFFER = 1;"
    },
    {
        "input": "Institutions in urban areas that offer medical programs and have a hospital:",
        "query": "SELECT INSTNM, LOCALE, MEDICAL, HOSPITAL FROM public.\"HD2022\" WHERE LOCALE = 11 AND MEDICAL = 1 AND HOSPITAL = 1;"
    },
    {
        "input": "Institutes which require Secondary School GPA for getting admission in Undergrad program.",
        "query": "SELECT IC.campusid, IC.pcaddr, IC.pccity FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.admcon1 = 1;",
    },
    {
        "input": "Institutes which do not require Secondary School GPA for getting admission in Undergrad program but considered if submitted.",
        "query": "SELECT IC.campusid, IC.pcaddr, IC.pccity FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.admcon1 = 5;",
    },
    {
        "input": "Institutes which do not require Secondary School GPA for getting admission in Undergrad program but even if submitted.",
        "query": "SELECT IC.campusid, IC.pcaddr, IC.pccity FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.admcon1 = 3;",
    },
    {
        "input": "Institutes which require Secondary school rank for getting admission in Undergrad program.",
        "query": "SELECT IC.campusid, IC.pcaddr, IC.pccity FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.admcon2 = 1;",
    },
    {
        "input": "Institutes which do not require Secondary school rank for getting admission in Undergrad program but considered if submitted.",
        "query": "SELECT IC.campusid, IC.pcaddr, IC.pccity FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.admcon2 = 5;",
    },
    {
        "input": "Institutes which do not require Secondary school rank for getting admission in undergrad but even if submitted.",
        "query": "SELECT IC.campusid, IC.pcaddr, IC.pccity FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.admcon2 = 3;",
    },
    {
        "input": "Institutes which require Secondary school record for getting admission in Undergrad program.",
        "query": "SELECT IC.campusid, IC.pcaddr, IC.pccity FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.admcon3 = 1;",
    },
    {
        "input": "Institutes which do not require Secondary school record for getting admission in Undergrad program but considered if submitted.",
        "query": "SELECT IC.campusid, IC.pcaddr, IC.pccity FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.admcon3 = 5;",
    },
    {
        "input": "Institutes which do not require Secondary school record for getting admission in Undergrad program but even if submitted.",
        "query": "SELECT IC.campusid, IC.pcaddr, IC.pccity FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.admcon3 = 3;",
    },
    {
        "input": "Institutes which require Completion of college-preparatory program for getting admission in undergrad.",
        "query": "SELECT IC.campusid, IC.pcaddr, IC.pccity FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.admcon4 = 1;",
    },
    {
        "input": "Institutes which do not require Completion of college-preparatory program for getting admission in undergrad but considered if submitted.",
        "query": "SELECT IC.campusid, IC.pcaddr, IC.pccity FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.admcon4 = 5;",
    },
    {
        "input": "Institutes which do not require Completion of college-preparatory program for getting admission in undergrad but even if submitted.",
        "query": "SELECT IC.campusid, IC.pcaddr, IC.pccity FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.admcon4 = 3;",
    },
    {
        "input": "Institutes which require Recommendations for getting admission in undergrad.",
        "query": "SELECT IC.campusid, IC.pcaddr, IC.pccity FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.admcon5 = 1;",
    },
    {
        "input": "Institutes which do not require Recommendations for getting admission in undergrad but considered if submitted.",
        "query": "SELECT IC.campusid, IC.pcaddr, IC.pccity FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.admcon5 = 5;",
    },
    {
        "input": "Institutes which do not require Recommendations for getting admission in undergrad but even if submitted.",
        "query": "SELECT IC.campusid, IC.pcaddr, IC.pccity FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.admcon5 = 3;",
    },
    {
        "input": "Institutes which do not require Personal statement or essay for getting admission in undergrad but even if submitted.",
        "query": "SELECT IC.campusid, IC.pcaddr, IC.pccity FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.admcon11 = 3;"
    },
    {
        "input": "Institutes which require Legacy status for getting admission in undergrad.",
        "query": "SELECT IC.campusid, IC.pcaddr, IC.pccity FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.admcon12 = 1;"
    },
    {
        "input": "Institutes which do not require Legacy status for getting admission in undergrad but considered if submitted.",
        "query": "SELECT IC.campusid, IC.pcaddr, IC.pccity FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.admcon12 = 5;"
    },
    {
        "input": "Institutes which do not require Legacy status for getting admission in undergrad but even if submitted.",
        "query": "SELECT IC.campusid, IC.pcaddr, IC.pccity FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.admcon12 = 3;"
    },
    {
        "input": "Top 5 institutes in each state shows Highest number of applicants applied for the fall term undergrad admission.",
        "query": "SELECT DISTINCT IC.pccity, IC.index, SUM(ADM.applcn) AS applicant_count FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.applcn IS NOT NULL GROUP BY IC.pccity, IC.index ORDER BY SUM(ADM.applcn) DESC LIMIT 5;"
    },
    {
        "input": "Top 5 institutes in each state shows Highest number of men applicants applied for the fall term undergrad admission.",
        "query": "SELECT DISTINCT IC.pccity, IC.index, SUM(ADM.applcnm) AS applicant_count FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.applcnm IS NOT NULL GROUP BY IC.pccity, IC.index ORDER BY SUM(ADM.applcnm) DESC LIMIT 5;"
    },
    {
        "input": "Top 5 institutes in each state shows Highest number of women applicants applied for the fall term undergrad admission.",
        "query": "SELECT DISTINCT IC.pccity, IC.index, SUM(ADM.applcnw) AS applicant_count FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.applcnw IS NOT NULL GROUP BY IC.pccity, IC.index ORDER BY SUM(ADM.applcnw) DESC LIMIT 5;"
    },
    {
        "input": "Top 5 institutes in each state shows Highest number of applicants applied for the fall term undergrad admission.",
        "query": "SELECT DISTINCT IC.pccity, IC.index, SUM(ADM.applcn) AS applicant_count FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.applcn IS NOT NULL GROUP BY IC.pccity, IC.index ORDER BY SUM(ADM.applcn) DESC LIMIT 5;"
    },
    {
        "input": "Top 5 institutes in each state shows Highest number of men applicants applied for the fall term undergrad admission.",
        "query": "SELECT DISTINCT IC.pccity, IC.index, SUM(ADM.applcnm) AS applicant_count FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.applcnm IS NOT NULL GROUP BY IC.pccity, IC.index ORDER BY SUM(ADM.applcnm) DESC LIMIT 5;"
    },
    {
        "input": "Top 5 institutes in each state shows Highest number of women applicants applied for the fall term undergrad admission.",
        "query": "SELECT DISTINCT IC.pccity, IC.index, SUM(ADM.applcnw) AS applicant_count FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.applcnw IS NOT NULL GROUP BY IC.pccity, IC.index ORDER BY SUM(ADM.applcnw) DESC LIMIT 5;"
    },
    {
        "input": "Top 5 institutes in each state shows Highest number of applicants enrolled for the fall term undergrad admission.",
        "query": "SELECT DISTINCT IC.pccity, IC.index, SUM(ADM.enrlt) AS applicant_count FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.enrlt IS NOT NULL GROUP BY IC.pccity, IC.index ORDER BY SUM(ADM.enrlt) DESC LIMIT 5;"
    },
    {
        "input": "Top 5 institutes in each state shows Highest number of men enrolled for the fall term undergrad admission.",
        "query": "SELECT DISTINCT IC.pccity, IC.index, SUM(ADM.enrlm) AS applicant_count FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.enrlm IS NOT NULL GROUP BY IC.pccity, IC.index ORDER BY SUM(ADM.enrlm) DESC LIMIT 5;"
    },
    {
        "input": "Top 5 institutes in each state shows Highest number of women enrolled for the fall term undergrad admission.",
        "query": "SELECT DISTINCT IC.pccity, IC.index, SUM(ADM.enrlw) AS applicant_count FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.enrlw IS NOT NULL GROUP BY IC.pccity, IC.index ORDER BY SUM(ADM.enrlw) DESC LIMIT 5;"
    },
    {
        "input": "Top 5 institutes in each state shows Highest number of students enrolled for Full-Time in the fall term undergrad admission.",
        "query": "SELECT DISTINCT IC.pccity, IC.index, SUM(ADM.enrlft) AS applicant_count FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.enrlft IS NOT NULL GROUP BY IC.pccity, IC.index ORDER BY SUM(ADM.enrlft) DESC LIMIT 5;"
    },
    {
        "input": "Top 5 institutes in each state shows Highest number of men enrolled for Full-Time in the fall term undergrad admission.",
        "query": "SELECT DISTINCT IC.pccity, IC.index, SUM(ADM.enrlftm) AS applicant_count FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.enrlftm IS NOT NULL GROUP BY IC.pccity, IC.index ORDER BY SUM(ADM.enrlftm) DESC LIMIT 5;"
    },
    {
        "input": "Top 5 institutes in each state shows Highest number of women enrolled for Full-Time in the fall term undergrad admission.",
        "query": "SELECT DISTINCT IC.pccity, IC.index, SUM(ADM.enrlftw) AS applicant_count FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.enrlftw IS NOT NULL GROUP BY IC.pccity, IC.index ORDER BY SUM(ADM.enrlftw) DESC LIMIT 5;"
    },
    {
        "input": "Top 5 institutes in each state shows Highest number of students enrolled for Part-Time in the fall term undergrad admission.",
        "query": "SELECT DISTINCT IC.pccity, IC.index, SUM(ADM.enrlpt) AS applicant_count FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.enrlpt IS NOT NULL GROUP BY IC.pccity, IC.index ORDER BY SUM(ADM.enrlpt) DESC LIMIT 5;"
    },
    {
        "input": "Top 5 institutes in each state shows Highest number of men enrolled for Part-Time in the fall term undergrad admission.",
        "query": "SELECT DISTINCT IC.pccity, IC.index, SUM(ADM.enrlptm) AS applicant_count FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.enrlptm IS NOT NULL GROUP BY IC.pccity, IC.index ORDER BY SUM(ADM.enrlptm) DESC LIMIT 5;"
    },
    {
        "input": "Top 5 institutes in each state shows Highest number of women enrolled for Full-Time in the fall term undergrad admission.",
        "query": "SELECT DISTINCT IC.pccity, IC.index, SUM(ADM.enrlftw) AS applicant_count FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.enrlftw IS NOT NULL GROUP BY IC.pccity, IC.index ORDER BY SUM(ADM.enrlftw) DESC LIMIT 5;"
    },
    {
        "input": "Top 5 institutes in each state shows Highest Number of first-time degree/certificate-seeking students submitting SAT scores.",
        "query": "SELECT DISTINCT IC.pccity, IC.index, SUM(ADM.satnum) AS applicant_count FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.satnum IS NOT NULL GROUP BY IC.pccity, IC.index ORDER BY SUM(ADM.satnum) DESC LIMIT 5;"
    },
    {
        "input": "Top 5 institutes in each state shows Highest Percent of first-time degree/certificate-seeking students submitting SAT scores.",
        "query": "SELECT DISTINCT IC.pccity, IC.index, SUM(ADM.satpct) AS applicant_count FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.satpct IS NOT NULL GROUP BY IC.pccity, IC.index ORDER BY SUM(ADM.satpct) DESC LIMIT 5;"
    },
    {
        "input": "Top 5 institutes in each state shows Highest Number of first-time degree/certificate-seeking students submitting ACT scores.",
        "query": "SELECT DISTINCT IC.pccity, IC.index, SUM(ADM.actnum) AS applicant_count FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.actnum IS NOT NULL GROUP BY IC.pccity, IC.index ORDER BY SUM(ADM.actnum) DESC LIMIT 5;"
    },
    {
        "input": "Top 5 institutes in each state shows Highest Percent of first-time degree/certificate-seeking students submitting ACT scores.",
        "query": "SELECT DISTINCT IC.pccity, IC.index, SUM(ADM.actpct) AS applicant_count FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.actpct IS NOT NULL GROUP BY IC.pccity, IC.index ORDER BY SUM(ADM.actpct) DESC LIMIT 5;"
    },
    {
        "input": "Top 5 institutes in each state shows highest SAT Evidence-Based Reading and Writing 25th percentile score.",
        "query": "SELECT DISTINCT IC.pccity, IC.index, SUM(ADM.satvr25) AS applicant_count FROM public.\"ADM2022\" AS ADM INNER JOIN public.\"IC2022_CAMPUSES\" AS IC ON ADM.unitid = IC.index WHERE ADM.satvr25 IS NOT NULL GROUP BY IC.pccity, IC.index ORDER BY SUM(ADM.satvr25) DESC LIMIT 5;"
    },

]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=5,
    input_keys=["input"],
)

example_prompt = PromptTemplate.from_template(
    "User input: {input}\nSQL query: {query}")
prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    prefix="You are a PostgreSQL expert. Given an input question, create a syntactically correct PostgreSQL query to run. Unless otherwise specificed, do not return more than {top_k} rows.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries.\n\nDO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.",
    suffix="User input: {input}\nSQL query: ",
    input_variables=["input", "top_k", "table_info"],
)


query_chain = create_sql_query_chain(llm, db, prompt=prompt)
# Convert "question" key to the "input" key expected by current table_chain.
table_chain = {"input": itemgetter("question")} | table_chain
# Set table_names_to_use using table_chain.
full_chain = RunnablePassthrough.assign(
    table_names_to_use=table_chain) | query_chain


# Streamlit app
# Final Response
def main():
    st.set_page_config(page_title="University Explorer Bot",
                       page_icon=':llama:',
                       layout='centered',
                       initial_sidebar_state='collapsed')

    st.header("University Explorer ChatBot")

    input_text = st.text_input("Search the Information you want")

    submit = st.button("Generate")

    if submit:
        response = full_chain.invoke({"question": input_text})
        query_text = response
        output_text = db.run(response)
        st.write(output_text)
        st.write(query_text)


if __name__ == "__main__":
    main()

# To run the code execute:
    # streamlit run app_v2.0.py
