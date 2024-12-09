import streamlit as st
from prompts import final_prompt, answer_prompt
from table_details import table_chain as select_table
from vector_store import handle_user_query, retriever_prompt
from langchain_core.runnables import RunnablePassthrough,RunnableMap, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain.memory import ChatMessageHistory
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT=os.getenv("LANGCHAIN_PROJECT")

db_url = os.getenv("DB_URL")

#====================MongoDB Client Vector DB Connection=====================================
client = MongoClient(os.getenv('MONGODB_URI'), tls=True,
    tlsAllowInvalidCertificates=True)
mongo_db = client.get_database(os.getenv('DB_NAME'))
c_name = os.getenv('COLLECTION_NAME')
collection= mongo_db.Data
#====================================================================================================
@st.cache_resource
def get_chain():
    db = SQLDatabase.from_uri(db_url)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    generate_query = create_sql_query_chain(llm, db, final_prompt)
    execute_query = QuerySQLDataBaseTool(db=db)
    rephrase_answer = answer_prompt | llm | StrOutputParser()
    
    chain = (
        RunnablePassthrough.assign(context=itemgetter("context"), table_names_to_use=select_table) |
        RunnablePassthrough.assign(query=generate_query).assign(
            result=itemgetter("query") | execute_query
        )
        | rephrase_answer
    )

    return chain



def create_history(messages):
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history


def invoke_chain(question, messages):
    try:
        context = handle_user_query(question,collection)
        print(context)
        chain = get_chain()
        history = create_history(messages)
        response = chain.invoke(
            {"question": question,"context": context, "top_k": 3, "messages": history.messages})
        history.add_user_message(question)
        history.add_ai_message(response)
        
        # Check if response is empty or None
        if not response or response.strip() == "":
            return "Sorry, I couldn't find any specific information related to your query. Please try asking something else or provide more details!"
        
        elif "error" in response:
            return "Sorry, I couldn't find any specific information related to your query. Please try asking something else or provide more details!"  
        return response 
        
    except Exception as e:
        print(f"Error invoking chain: {e}")
        return "Sorry, an error occurred while processing your request."
