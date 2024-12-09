# This file contains the database connection and the SQLDatabase object

import os
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from llama_index.core import SQLDatabase

db_url = os.getenv("DB_URL")


def db_connect():
    # Creating a SQLAlchemy engine
    engine = create_engine(db_url)

    # Creating a session
    Session = sessionmaker(bind=engine)
    session = Session()

    # Connecting and creating a cursor
    connection = engine.connect()
    cursor = connection.connection.cursor()

    # Creating metadata object
    metadata_obj = MetaData()

    return engine, session, connection, cursor, metadata_obj


engine, session, connection, cursor, metadata_obj = db_connect()

sql_database = SQLDatabase(engine)
