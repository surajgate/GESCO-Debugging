import os
import json
import smtplib
from datetime import datetime, timezone, timedelta, date

import pandas as pd

from io import StringIO
from dotenv import load_dotenv
from sqlalchemy import func, or_
from sqlalchemy.dialects.postgresql import JSONB
from langchain_milvus import Milvus
from langchain_openai import AzureOpenAIEmbeddings

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

AZURE_OPENAI_EMBEDDINGS_MODEL = os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL", "")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")

def fetch_answer_without_citations():
    """
    Fetches questions from the database where responses do not have citations.
    The function determines the reporting window and retrieves questions
    created within the latest reporting period.
    
    Returns:
        list: A list of questions without citations.
    """
    try:
        db = get_db()
        now_utc = datetime.now(timezone.utc)
        
        REPORT_TIMES_UTC = [
            4,   # 10:00 AM IST -> 04:30 UTC
            7,   # 1:00 PM IST  -> 07:30 UTC
            10,  # 4:00 PM IST  -> 10:30 UTC
            13,  # 7:00 PM IST  -> 13:30 UTC
        ]
        
        # Current hour in UTC
        current_hour_utc = now_utc.hour
        
        # Find the most recent report time that has passed (in UTC)
        last_report_time = max(
            [hour for hour in REPORT_TIMES_UTC if hour <= current_hour_utc], 
            default=REPORT_TIMES_UTC[-1]
        )
        
        # Calculate start_time based on report windows (all in UTC)
        if last_report_time == REPORT_TIMES_UTC[0]:  # 04:30 UTC (10 AM IST) case
            # For 10 AM IST report, start from previous day's 7 PM IST (13:30 UTC)
            start_time = (now_utc - timedelta(days=1)).replace(
                hour=REPORT_TIMES_UTC[-1],
                minute=30,
                second=0,
                microsecond=0,
                tzinfo=timezone.utc
            )
        else:
            prev_report_idx = REPORT_TIMES_UTC.index(last_report_time) - 1
            prev_report_time = REPORT_TIMES_UTC[prev_report_idx]
            
            start_time = now_utc.replace(
                hour=prev_report_time,
                minute=30,
                second=0,
                microsecond=0,
                tzinfo=timezone.utc
            )
        
        # Query for questions without citations
        users_list = db.query(user_departments.c.user_id).filter(
            user_departments.c.department_id == DEPARTMENT_ID
        )
        user_ids = [user_id[0] for user_id in users_list]
        
        questions = (
            db.query(
                chats.c.sig_response.label("interpreted question"),
                chats.c.response
            )
            .filter(chats.c.user_id.in_(user_ids))
            .filter(
                or_(
                    func.cast(chats.c.citations, JSONB) == '{}',
                    func.jsonb_array_length(func.cast(chats.c.citations, JSONB)) == 0
                )
            )
            .filter(chats.c.created_at >= start_time)
            .order_by(chats.c.created_at.desc())
        )
        
        return [question[0] for question in questions]
        
    except Exception as e:
        print(f"An error occurred while fetching answers without citations: {e}")
        return None
    finally:
        db.close()

def fetch_mmr_chunks_and_scores(question):
    """
    Retrieves the most relevant text chunks for a given question using Maximal Marginal Relevance (MMR).
    
    Args:
        question (str): The input question for which relevant chunks are retrieved.
    
    Returns:
        list: A list of dictionaries containing chunk metadata and content.
    """
    try:
        embedding_function = AzureOpenAIEmbeddings(
            api_key=AZURE_OPENAI_API_KEY,
            model=AZURE_OPENAI_EMBEDDINGS_MODEL,
            azure_deployment=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT
        )
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

def save_chunks_to_stringio():
    """
    Fetches questions without citations, retrieves relevant chunks using MMR,
    and stores the results in a StringIO object.
    
    Returns:
        StringIO: A StringIO object containing formatted text data.
    """
    questions = fetch_answer_without_citations()
    if not questions:
        print("No questions found without citations.")
        return None

    file_content = StringIO()

    for idx_q, question in enumerate(questions, start=1):
        file_content.write(f"\n\n=== Question {idx_q}: {question} ===\n\n")

        chunks = fetch_mmr_chunks_and_scores(question)
        if not chunks:
            file_content.write("No relevant chunks found.\n")
            continue

        for idx_c, chunk in enumerate(chunks, start=1):
            file_content.write(f"\nChunk {idx_c}:\n")
            file_content.write(f"File ID: {chunk['file_id']}\n")
            file_content.write(f"MMR Score: {chunk['mmr_score']}\n")
            file_content.write(f"File Directory: {chunk['file_directory']}\n")
            file_content.write(f"Filename: {chunk['filename']}\n")
            file_content.write(f"Page Number: {chunk['page_number']}\n")
            file_content.write(f"Page Content:\n{chunk['page_content']}\n")
            file_content.write("\n" + "-" * 100 + "\n") 

    file_content.seek(0)

    return file_content