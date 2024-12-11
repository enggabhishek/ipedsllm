# type: ignore
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
import streamlit as st

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


@st.cache_resource
def get_example_selector():
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        Chroma,
        k=5,
        input_keys=["input"],
    )
    return example_selector
