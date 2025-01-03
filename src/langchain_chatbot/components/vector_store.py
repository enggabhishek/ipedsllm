from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough,RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from prompts import column_name_retriver_prompt, encoded_values_retriver_prompt, column_desc_retriver_prompt
import re
import ast
import concurrent.futures
import openai
import os
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

class VectorStore:
    def __init__(self, collection):
        self.collection = collection

    #==================================================Text Emebedding Model Method======================================
    def get_embedding(self, text):
        
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
    def vector_search(self, user_query, collection):
    
        # Generate embedding for the user query
        query_embedding = self.get_embedding(user_query)

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

    def get_table_info(self, question: str, template: str, context: dict):
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
    def retrieve_list_objects(self, pattern, text):
        return re.findall(pattern, text)
    #====================================Fetching Coloumn Description=======================================================================
    def get_substring_before_colon(self, input_string):
        result = input_string.split(':', 1)[0]
        return result.strip()
    #=============================================Fetching Encoded Values==============================================================
    def fetch_value(self, input_string):
    
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
    def handle_user_query(self, question):
        get_knowledge = self.vector_search(question, self.collection)
        
        def process_result(result):
            context = {}
            column_details = ''
            output_list = []
            if result.get('Encoded_Values', '-1') != '-1':
                context["Table_Name"] = result.get('Table_Name')
                context["Column_Description"] = result.get('Column_Description')
                column_details += self.get_table_info(question, column_name_retriver_prompt, context)

                if "Column names related to the question" in column_details:
                    ls = self.retrieve_list_objects(self.patterns[1], column_details)
                else:
                    ls = ast.literal_eval(column_details)
                
                encoded_values = ast.literal_eval(result.get('Encoded_Values', '{}'))
                
                for i in ls:
                    cdesc = self.get_table_info(i, column_desc_retriver_prompt, context["Column_Description"])
                    if encoded_values.get(i, 'N/A') != 'N/A':
                        code_value = encoded_values.get(i)
                        if isinstance(code_value, str):
                            code_value = ast.literal_eval(code_value)
                        elif isinstance(code_value, dict):
                            pass
                        code_value = self.get_table_info(question, encoded_values_retriver_prompt, code_value)
                        output_list.append({
                            "Table_Name": result.get('Table_Name'),
                            "Column_Name": i,
                            "Column_Description": self.get_substring_before_colon(cdesc),
                            "Encoded_Values": self.fetch_value(code_value)
                        })
                    else:
                        output_list.append({
                            "Table_Name": result.get('Table_Name'),
                            "Column_Name": i,
                            "Column_Description": self.get_substring_before_colon(cdesc)
                        })
            return output_list

        # Multithreading for processing results
        output = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_result, result) for result in get_knowledge]
            for future in concurrent.futures.as_completed(futures):
                output.extend(future.result())

        return output


#====================MongoDB Client Vector DB Connection=====================================
client = MongoClient(os.getenv('MONGODB_URI'), tls=True,
    tlsAllowInvalidCertificates=True)
mongo_db = client.get_database(os.getenv('DB_NAME'))
c_name = os.getenv('COLLECTION_NAME')
collection= mongo_db.Data

#======================Create VectorStoreInstance=====================================================
vectorStoreInstance = VectorStore(collection)