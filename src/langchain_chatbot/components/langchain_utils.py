# type: ignore
import streamlit as st
from prompts import final_prompt, answer_prompt, retriever_prompt
from table_details import table_chain as select_table
from vector_store import retriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain.memory import ChatMessageHistory
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_community.utilities.sql_database import SQLDatabase
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
db_url = os.getenv("DB_URL")

class LangChainChatbot:
    def __init__(self):
        self.chain = self.get_chain()

    @st.cache_resource
    def get_chain(_self):
        db = SQLDatabase.from_uri(db_url)
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        context_chain = (
            {"context": itemgetter("question") | retriever,
             "question": itemgetter("question")}
            | retriever_prompt
            | llm
            | StrOutputParser()
        )
        generate_query = create_sql_query_chain(llm, db, final_prompt)
        execute_query = QuerySQLDataBaseTool(db=db)
        rephrase_answer = answer_prompt | llm | StrOutputParser()
        chain = (
            RunnablePassthrough.assign(context=context_chain, table_names_to_use=select_table) |
            RunnablePassthrough.assign(query=generate_query).assign(
                result=itemgetter("query") | execute_query
            )
            | rephrase_answer
        )

        return chain

    def create_history(self, messages):
        history = ChatMessageHistory()
        for message in messages:
            if message["role"] == "user":
                history.add_user_message(message["content"])
            else:
                history.add_ai_message(message["content"])
        return history

    def invoke_chain(self, question, messages):
        try:
            history = self.create_history(messages)
            response = self.chain.invoke(
                {"question": question, "top_k": 3, "messages": history.messages})
            history.add_user_message(question)
            history.add_ai_message(response)
            
            if not response or response.strip() == "":
                return "Sorry, I couldn't find any specific information related to your query. Please try asking something else or provide more details!"
            
            elif "error" in response:
                return "Sorry, I couldn't find any specific information related to your query. Please try asking something else or provide more details!"  
            return response 
            
        except Exception as e:
            print(f"Error invoking chain: {e}")
            return "Sorry, an error occurred while processing your request."

#=============Creating instance==================
chatbot = LangChainChatbot()
