"""Includes all the modules for setup db."""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()


APP_DB_USERNAME = os.getenv("APP_DB_USERNAME")
APP_DB_IP = os.getenv("APP_DB_IP")
APP_DB_PASSWORD = os.getenv("APP_DB_PASSWORD")
APP_DB_NAME = os.getenv("APP_DB_NAME")
APP_DB_PORT = os.getenv("APP_DB_PORT")

# Construct the connection string
URL_DATABASE = f"postgresql://{APP_DB_USERNAME}:{APP_DB_PASSWORD}@{APP_DB_IP}:{APP_DB_PORT}/{APP_DB_NAME}"

engine = create_engine(URL_DATABASE)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

metadata = MetaData()

metadata.reflect(bind=engine)

chats = Table("chats", metadata, autoload_with=engine)
user = Table("users", metadata, autoload_with=engine)
chat_feedback = Table("pdf_chat_feedback", metadata, autoload_with=engine)

def get_db():
    db = SessionLocal()
    return db