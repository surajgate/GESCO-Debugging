import os
import json
import smtplib
from datetime import datetime, timezone, timedelta, date

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import func, or_
from sqlalchemy.dialects.postgresql import JSONB
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings

from db import get_db, chats, chat_feedback, user, user_departments

load_dotenv()

MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME")
MILVUS_CONNECTION_URI = os.getenv("MILVUS_CONNECTION_URI")
OPENAI_EMBEDDINGS_MODEL = os.getenv("OPENAI_EMBEDDINGS_MODEL")
CHUNK_RETRIEVAL_ALGORITHM = os.getenv("CHUNK_RETRIEVAL_ALGORITHM")
NUM_CHUNKS_RETRIEVED = int(os.getenv("NUM_CHUNKS_RETRIEVED", default="50"))
MMR_LAMBDA_MULT = float(os.getenv("MMR_LAMBDA_MULT", default="0.2"))
NUM_CHUNKS_TO_MMR = int(os.getenv("NUM_CHUNKS_TO_MMR", default="100"))
DEPARTMENT_ID = os.getenv("DEPARTMENT_ID","")


def fetch_answer_without_citations():
    try:
        db = get_db()
        today = date.today()

        # Define the query and the filters upfront
        users_list = db.query(user_departments.c.user_id).filter(user_departments.c.department_id == DEPARTMENT_ID)
        user_ids = [user_id[0] for user_id in users_list]
        questions = (
            db.query(
                chats.c.sig_response.label("interpretted question"),
                chats.c.response
            )
            .filter(chats.c.user_id.in_(user_ids))
            .filter(
                or_(
                    func.cast(chats.c.citations, JSONB) == '{}',
                    func.jsonb_array_length(func.cast(chats.c.citations, JSONB)) == 0
                )
            )
            .order_by(chats.c.created_at.desc())
        )

        return (
            [ question[0] for question in questions]
        )
    except Exception as e:
        print(f"An error occurred while fetching the answer without citations information: {e}")
        return None
    finally:
        db.close()

def fetch_mmr_chunks_and_scores(question):
    try:
        embedding_function = OpenAIEmbeddings(model=OPENAI_EMBEDDINGS_MODEL)
        vectorstore = Milvus(
            connection_args={"uri": MILVUS_CONNECTION_URI},
            embedding_function=embedding_function,
            collection_name=MILVUS_COLLECTION_NAME,
            search_params={
                "metric_type": "L2",
                "params": {"ef": NUM_CHUNKS_TO_MMR + 100},
            },
            enable_dynamic_field=True
        )
        retrieved_docs = vectorstore.max_marginal_relevance_search(
            query=question, k=NUM_CHUNKS_RETRIEVED, fetch_k=NUM_CHUNKS_TO_MMR, lambda_mult=MMR_LAMBDA_MULT)

        chunks_metadata_list = []
        for chunk in retrieved_docs:
            chunk_metadata_dict = {
                "file_id": chunk.metadata["fileid"],
                "mmr_score": chunk.metadata["mmr_score"],
                "file_directory": chunk.metadata["file_directory"],
                "filename": chunk.metadata["filename"],
                "page_number": chunk.metadata["page_number"],
                "page_content": chunk.page_content
            }
            chunks_metadata_list.append(chunk_metadata_dict)

        return chunks_metadata_list
    except Exception as e:
        print(f"An error occurred while fetching MMR chunks and scores for question: {question}: {e}")
        return None
