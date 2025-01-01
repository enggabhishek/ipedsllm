from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

class VectorStoreChatbot:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_function = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key, model="text-embedding-ada-002")
        self.loader = JSONLoader(
            file_path=r"Data\\data_utils\\tableinfo.json",
            jq_schema=".[].Table_Info[]",
            content_key="Table_Description",
            metadata_func=self.metadata_func,
        )
        self.data = self.loader.load()
        self.vectorstore = Chroma.from_documents(self.data, self.embedding_function)
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
        self.retriever = self.vectorstore.as_retriever()

    def metadata_func(self, record: dict, metadata: dict) -> dict:
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
        return metadata

    def process_data(self):
        with ThreadPoolExecutor() as executor:
            future = executor.submit(self.loader.load)
            self.data = future.result()
            self.vectorstore = Chroma.from_documents(self.data, self.embedding_function)
            self.retriever = self.vectorstore.as_retriever()
            return self.retriever


vectorStoreInstance = VectorStoreChatbot()
retriever = vectorStoreInstance.process_data()
