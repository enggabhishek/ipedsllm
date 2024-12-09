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
    # C2022_A_prefix,
    # C2022_B_prefix,
    # C2022_C_prefix,
    EFFY_2022_prefix,
    # EFFY2022_DIST_prefix,
    # EFIA2022_prefix,
    # FLAGS2022_prefix,
    # GR200_22_prefix,
    GR2022_prefix,
    # GR2022_PEL_SSL_prefix,
    # IC2022_prefix,
    # IC2022_AY_prefix,
    # IC2022_CAMPUSES_prefix,
    #     IC2022_PY_prefix,
    #     OM2022_prefix,
    #     SFA2122_prefix,
    #     SFAV2122_prefix,
)
from llama_index.core.llms import ChatResponse


from db_utils import sql_database
from llama_index.core import VectorStoreIndex

# set Logging to DEBUG for more detailed outputs
table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [  # list of SQLTableSchema objects
    SQLTableSchema(table_name="adm2022", context_str=ADM2022_prefix),
    SQLTableSchema(table_name="hd2022", context_str=HD2022_prefix),
    SQLTableSchema(table_name="c2022dep", context_str=C2022_DEP_prefix),
    # SQLTableSchema(table_name="C2022_A", context_str=C2022_A_prefix),
    # SQLTableSchema(table_name="C2022_B", context_str=C2022_B_prefix),
    # SQLTableSchema(table_name="C2022_C", context_str=C2022_C_prefix),
    SQLTableSchema(table_name="effy2022", context_str=EFFY_2022_prefix),
    # SQLTableSchema(table_name="EFFY2022_DIST",
    #                context_str=EFFY2022_DIST_prefix),
    # SQLTableSchema(table_name="EFIA2022", context_str=EFIA2022_prefix),
    # SQLTableSchema(table_name="FLAGS2022", context_str=FLAGS2022_prefix),
    # SQLTableSchema(table_name="GR200_22", context_str=GR200_22_prefix),
    SQLTableSchema(table_name="gr2022", context_str=GR2022_prefix),
    # SQLTableSchema(table_name="GR2022_L2", context_str=GR2022_PEL_SSL_prefix),
    # SQLTableSchema(table_name="GR2022_PELL_SSL", context_str=GR2022_prefix),

    # SQLTableSchema(table_name="IC2022", context_str=IC2022_prefix),
    # SQLTableSchema(table_name="IC2022_AY", context_str=IC2022_AY_prefix),
    # SQLTableSchema(table_name="IC2022_CAMPUSES",
    #                context_str=IC2022_CAMPUSES_prefix),
    # SQLTableSchema(table_name="IC2022_PY", context_str=IC2022_PY_prefix),
    # SQLTableSchema(table_name="OM2022", context_str=OM2022_prefix),
    # SQLTableSchema(table_name="SFA2122", context_str=SFA2122_prefix),
    # SQLTableSchema(table_name="SFAV2122", context_str=SFAV2122_prefix),
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
