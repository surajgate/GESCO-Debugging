import os
import json
import smtplib
import pandas as pd

from datetime import datetime, timezone, timedelta, date
from sqlalchemy import func, or_
from sqlalchemy.dialects.postgresql import JSONB

from db import get_db, chats, chat_feedback, user, user_departments

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