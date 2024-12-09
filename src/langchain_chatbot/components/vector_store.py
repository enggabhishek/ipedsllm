from langchain.prompts import ChatPromptTemplate
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough,RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from prompts import column_name_retriver_prompt, encoded_values_retriver_prompt, column_desc_retriver_prompt
import re
import ast
import openai
from dotenv import load_dotenv
load_dotenv()
#==================================================Text Emebedding Model Method======================================
def get_embedding(text):
    
    EMBEDDING_MODEL = "text-embedding-ada-002"
    """Generate an embedding for the given text using OpenAI's API."""

    # Check for valid input
    if not text or not isinstance(text, str):
        return None

    try:
        # Call OpenAI API to get the embedding
        embedding = openai.embeddings.create(input=text, model=EMBEDDING_MODEL).data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return None
    
    
#====================================================Vector Search Method============================================================    
def vector_search(user_query, collection):
  
    # Generate embedding for the user query
    query_embedding = get_embedding(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    # Define the vector search pipeline
    pipeline = [
                {
                    "$vectorSearch":{
                                        "index": "vector_index",
                                        "path": "embedding",
                                        "queryVector": query_embedding,
                                        "numCandidates": 4,
                                        "limit": 3
                                    }
                },
                {
                    "$project": {
                                    "_id": 0,  # Exclude the _id field
                                    "text": 1,
                                    "Table_Description": 1, # Include the Table_Description field
                                    "Table_Name": 1,
                                    "Encoded_Values": 1,  # Include the Encoded_Values field
                                    "Column_Description": 1, # Include the Column_Description field
                                    "score": {
                                                "$meta": "vectorSearchScore"  # Include the search score
                                            }
                                }
                }
        
            ]
    # Execute the search
    results = collection.aggregate(pipeline)
    return list(results)

#=====================================Get table Infos===========================================

def get_table_info(question: str, template: str, context: dict):
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI()

    table_chain = (
        RunnableMap({"context": RunnablePassthrough(), "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )
    return table_chain.invoke({"context": context, "question": question})

#=====================================Pattern Selection for Column Names=====================================
patterns = [
    "(\[.*?\])",  # Pattern 1
    "'(\w+)'",    # Pattern 2
]
def retrieve_list_objects(pattern, text):
    return re.findall(pattern, text)
#====================================Fetching Coloumn Description=======================================================================
def get_substring_before_colon(input_string):
    result = input_string.split(':', 1)[0]
    return result.strip()
#=============================================Fetching Encoded Values==============================================================
def fetch_value(input_string):
  
    last_colon_index = input_string.rfind(':')
    
    if last_colon_index != -1:
        value_string = input_string[last_colon_index+1:].lstrip()
        
        try:
            return int(value_string)
        except ValueError:
            try:
                return float(value_string)
            except ValueError:
                return value_string
    else:
        return None
#============================================================================================================

def handle_user_query(question, collection):

  get_knowledge = vector_search(question, collection)
  output = "["
  count_col = 0
  for result in get_knowledge:
    context = {}
    column_details = ''
    if result.get('Encoded_Values', '-1')!= '-1':
        
      context["Table_Name"] = result.get('Table_Name')
      context["Column_Description"]= result.get('Column_Description')
      column_details+=get_table_info(question, column_name_retriver_prompt, context)
      cdesc = ''
      if "Column names related to the question" in column_details:
        ls = retrieve_list_objects(patterns[1], column_details)
        encoded_values=result.get('Encoded_Values')
        encoded_values=ast.literal_eval(encoded_values)
        code_value =''
        
        for i in ls:
          if encoded_values.get(i, 'N/A')!= 'N/A':
            if isinstance(encoded_values.get(i), str):
              code_value= ast.literal_eval(encoded_values.get(i))
            elif isinstance(encoded_values.get(i), dict):
              code_value = encoded_values.get(i)
            code_value = get_table_info(question, encoded_values_retriver_prompt, code_value)
            cdesc=get_table_info(i, column_desc_retriver_prompt ,context["Column_Description"])
            if count_col == 0:
              count_col+=1
              output+= "{"+ f"'Table_Name': '{result.get('Table_Name')}' ,'Column_Name': '{i}', 'Column_Description': {get_substring_before_colon(cdesc)} ,'Encoded_Values': {fetch_value(code_value)}" + "}"
            else:
              output+= ", {"+ f"'Table_Name': '{result.get('Table_Name')}' , 'Column_Name': '{i}', 'Column_Description': {get_substring_before_colon(cdesc)}, 'Encoded_Values': {fetch_value(code_value)}" + "}"
          else:
            cdesc=get_table_info(i, column_desc_retriver_prompt ,context["Column_Description"])
            if count_col == 0:
              count_col+=1
              output+= "{" + f"'Table_Name': '{result.get('Table_Name')}' ,'Column_Name': '{i}', 'Column_Description': {get_substring_before_colon(cdesc)}" + "}"
            else:
              output+= ", {" + f"'Table_Name': '{result.get('Table_Name')}' ,'Column_Name': '{i}', 'Column_Description': {get_substring_before_colon(cdesc)}" + "}"
        
      else:
        ls = ast.literal_eval(column_details)
        encoded_values=result.get('Encoded_Values')
        encoded_values=ast.literal_eval(encoded_values)
        code_value =''
        for i in ls:
          if encoded_values.get(i, 'N/A')!= 'N/A':
            if isinstance(encoded_values.get(i), str):
              code_value= ast.literal_eval(encoded_values.get(i))
            elif isinstance(encoded_values.get(i), dict):
              code_value = encoded_values.get(i)
            code_value = get_table_info(question, encoded_values_retriver_prompt, code_value)
            cdesc=get_table_info(i, column_desc_retriver_prompt ,context["Column_Description"])
            if count_col == 0:
              count_col+=1
              output+= "{"+ f"'Table_Name': '{result.get('Table_Name')}' , 'Column_Name': '{i}', 'Column_Description': {get_substring_before_colon(cdesc)} , 'Encoded_Values': {fetch_value(code_value)}" + "}"
            else:
              output+= ", {"+ f"'Table_Name': '{result.get('Table_Name')}' , 'Column_Name': '{i}', 'Column_Description': {get_substring_before_colon(cdesc)}, 'Encoded_Values': {fetch_value(code_value)}" + "}"
          else:
            cdesc=get_table_info(i, column_desc_retriver_prompt ,context["Column_Description"])
            if count_col == 0:
              count_col+=1
              output+= "{" + f"'Table_Name': '{result.get('Table_Name')}' ,'Column_Name': '{i}', 'Column_Description': {get_substring_before_colon(cdesc)}" + "}"
            else:
              output+= ", {" + f"'Table_Name': '{result.get('Table_Name')}' , 'Column_Name': '{i}', 'Column_Description': {get_substring_before_colon(cdesc)}" + "}"
  output += "]"
  return output


llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

template = """Answer the question based solely on the following context:
{context}
Extract all relevant table names, column names, column descriptions, and encoded values (if available) from the context based on the question. Include all pertinent tables, column descriptions, column names, and encoded values, which will be used by the downstream Text-to-SQL Agent to generate SQL queries for answers.
Perform the following tasks:
1. Identify Table Names
2. Identify Column Names
3. Identify Column Descriptions
4. Identify Encoded Values

Finally, return only the table names, Column Descriptions, column names, and encoded values (if available).

Question: {question}
    """
retriever_prompt = ChatPromptTemplate.from_template(template)
