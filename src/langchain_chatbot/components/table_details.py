from typing import List
import pandas as pd
import streamlit as st
from operator import itemgetter
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of table in SQL database.")


class TableDetails:
    def __init__(self, model: str, temperature: float, csv_path: str):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.csv_path = csv_path

    @st.cache_data
    def get_table_details(_self) -> str:
        # Read the CSV file into a DataFrame
        table_description = pd.read_csv(_self.csv_path)

        # Iterate over the DataFrame rows to create Document objects
        table_details = ""
        for index, row in table_description.iterrows():
            table_details += f"Table Name: {row['Table']}\nTable Description: {row['Description']}\n\n"

        return table_details

    def get_tables(self, tables: List[Table]) -> List[str]:
        return [table.name for table in tables]

    def create_table_chain(self) -> List[str]:
        self.table_details = self.get_table_details()
        table_details_prompt = f"""Refer the Above Context and Return the names of SQL Tables mentioned in the above context\n\n 
        The tables are:

        {self.table_details}
        """

        table_chain = {"input": itemgetter("question")} | create_extraction_chain_pydantic(
            Table, self.llm, system_message=table_details_prompt) | self.get_tables

        return table_chain


# Usage
table_details_instance = TableDetails(model="gpt-3.5-turbo-1106", temperature=0, csv_path="data/table_descriptions.csv")
table_chain = table_details_instance.create_table_chain()
