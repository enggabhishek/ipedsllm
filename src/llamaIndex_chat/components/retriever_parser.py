from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core.query_pipeline import FnComponent
from llama_index.core.retrievers import SQLRetriever
from context import (
    ADM2022_prefix,
    HD2022_prefix,
    C2022_DEP_prefix,
    EFFY_2022_prefix,
    GR2022_prefix
)
from llama_index.core.llms import ChatResponse
from db_utils import sql_database
from llama_index.core import VectorStoreIndex

# set Logging to DEBUG for more detailed outputs
table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [  # list of SQLTableSchema objects
    SQLTableSchema(table_name="ADM2022", context_str=ADM2022_prefix),
    SQLTableSchema(table_name="HD2022", context_str=HD2022_prefix),
    SQLTableSchema(table_name="C2022DEP", context_str=C2022_DEP_prefix),
    SQLTableSchema(table_name="EFFY2022", context_str=EFFY_2022_prefix),
    SQLTableSchema(table_name="GR2022", context_str=GR2022_prefix),
]  # add a SQLTableSchema for each table

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)

sql_retriever = SQLRetriever(sql_database)


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
