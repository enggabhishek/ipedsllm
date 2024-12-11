# type:ignore
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def metadata_func(record: dict, metadata: dict) -> dict:
    def column_retriever(ls):
        cname = []
        dtype = []
        cdesc = []
        for i in range(len(ls)):
            cname.append(record.get("Columns")[i].get("Column_Name"))
            dtype.append(record.get("Columns")[i].get("Data_Type"))
            cdesc.append(record.get("Columns")[i].get("Column_Description"))
        return cname, dtype, cdesc
    cname, dtype, cdesc = column_retriever(record.get("Columns"))

    metadata["Table_Name"] = record.get("Table_Name")
    metadata["Table_Description"] = record.get("Table_Description")
    metadata["Column_Names"] = str(cname)
    metadata["Data_Type"] = str(dtype)
    metadata["Column_Description"] = str(cdesc)
    # metadata["share"] = record.get("share")
    return metadata


embedding_function = OpenAIEmbeddings(
    openai_api_key=openai_api_key, model="text-embedding-ada-002")



loader = JSONLoader(
    file_path=r"Data\\data_utils\\tableinfo.json",
    jq_schema=".[].Table_Info[]",
    content_key="Table_Description",
    metadata_func=metadata_func,
)
data = loader.load()
vectorstore = Chroma.from_documents(
    data, embedding_function)
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
retriever = vectorstore.as_retriever()


template = """Answer the question based only on the following context:
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
retriever_prompt = ChatPromptTemplate.from_template(template)
