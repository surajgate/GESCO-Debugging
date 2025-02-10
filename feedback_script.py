import os
import json
import smtplib
import pandas as pd

from datetime import datetime, timezone, timedelta, date
from sqlalchemy import func, or_
from sqlalchemy.dialects.postgresql import JSONB

from db import get_db, chats, chat_feedback, user

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

EMAIL_LIST_FOR_CHAT_QUERY = os.getenv("EMAIL_LIST_FOR_CHAT_QUERY", [])
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
RECEIVER_EMAILS = os.getenv("RECEIVER_EMAILS", [])
SMTP_SERVER = os.getenv("SMTP_SERVER", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", 0))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMPT_PASSWORD = os.getenv("SMPT_PASSWORD", "")

current_date = datetime.now().date().strftime("%d-%m-%Y")
current_time = datetime.now(timezone.utc)  # Ensures current_time is timezone-aware (UTC)

def convert_str_to_list(input_str):
    """
    Convert a string representation of a list to an actual list.
    
    Args:
        input_str (str): The string to convert.
        
    Returns:
        list: The converted list of strings.
    """
    if isinstance(input_str, str):
        return [item.strip() for item in input_str.strip('[]').replace('"', '').replace("'", '').split(',')]
    return input_str

def fetch_feedback_data_in_chunks(chunk_size=10000):
    """
    Fetch feedback data from the database in chunks based on a predefined chunk size.
    
    Args:
        chunk_size (int): The number of records to fetch in each chunk. Default is 10,000.
        
    Returns:
        pandas.DataFrame: Dataframe of feedback data, or None if there is an error.
    """
    try:
        db = get_db()
        today = date.today()

        # Define the query and the filters upfront
        email_list = convert_str_to_list(EMAIL_LIST_FOR_CHAT_QUERY)
        
        query = (
            db.query(
                user.c.id.label("user_id"),
                user.c.username,
                user.c.email,
                chats.c.id.label("chat_id"),
                chats.c.conv_id,
                chats.c.question,
                chats.c.sig_response.label("interpretted question"),
                chat_feedback.c.rating.label("Is response factually correct?"),
                chat_feedback.c.rating_2.label("Is response relevant and focused?"),
                chat_feedback.c.rating_3.label("Accurate references?"),
                chat_feedback.c.comment,
                chats.c.response,
                chats.c.citations,
                chats.c.created_at,
            )
            .join(chats, chats.c.user_id == user.c.id)
            .outerjoin(chat_feedback, chat_feedback.c.chat_id == chats.c.id)
            .filter(user.c.email.in_(email_list))
            .order_by(chats.c.created_at.desc())
        )

        # Define aggregate queries in a more efficient way
        base_query = db.query(chats.c.id).join(user, chats.c.user_id == user.c.id).filter(user.c.email.in_(email_list))

        total_number_of_chats = base_query
        total_number_of_feedback = (
            db.query(chat_feedback.c.id)
            .join(chats, chat_feedback.c.chat_id == chats.c.id)
            .join(user, chats.c.user_id == user.c.id)
            .filter(user.c.email.in_(email_list))
        )
        total_number_of_answers_without_citations = (
            db.query(chats.c.id)
            .join(user, chats.c.user_id == user.c.id)
            .filter(user.c.email.in_(email_list))
            .filter(
                or_(
                    func.cast(chats.c.citations, JSONB) == '{}',
                    func.jsonb_array_length(func.cast(chats.c.citations, JSONB)) == 0
                )
            )
        )

        today_start = datetime.combine(today, datetime.min.time())
        today_end = datetime.combine(today + timedelta(days=1), datetime.min.time())

        # Today's metrics
        total_number_of_chats_today = base_query.filter(chats.c.created_at >= today_start, chats.c.created_at < today_end)
        total_number_of_feedback_today = (
            total_number_of_feedback
            .filter(chat_feedback.c.created_at >= today_start, chat_feedback.c.created_at < today_end)
        )
        
        total_number_of_answers_without_citations_today = (
            total_number_of_answers_without_citations
            .filter(chats.c.created_at >= today_start, chats.c.created_at < today_end)
        )

        return (
            pd.read_sql(query.statement, db.bind, chunksize=chunk_size),
            total_number_of_chats.count(), total_number_of_feedback.count(),
            total_number_of_answers_without_citations.count(),
            total_number_of_chats_today.count(),
            total_number_of_feedback_today.count(),
            total_number_of_answers_without_citations_today.count()
        )
    except Exception as e:
        print(f"An error occurred while fetching the feedback information: {e}")
        return None, 0, 0, 0
    finally:
        db.close()

def export_to_csv(feedback_data):
    """
    Export the feedback data to a CSV file in chunks without storing the file locally.
    
    Args:
        feedback_data (pandas.DataFrame): The feedback data to export.
        
    Returns:
        bool: True if export is successful, False if there is an error.
    """
    if feedback_data is None:
        print("No feedback data to export")
        return False
    
    try:
        first_chunk = True
        from io import StringIO
        output = StringIO()

        for chunk in feedback_data:
            chunk["response"] = chunk["response"].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.strip() else x
            )
            chunk.to_csv(
                output, 
                mode="a", 
                header=first_chunk, 
                index=False, 
                encoding="utf-8"
            )
            first_chunk = False

        output.seek(0)
        return output
    except Exception as e:
        print(f"An error occurred while exporting to CSV: {e}")
        return None

def send_email_with_attachment(
    sender_email, 
    receiver_emails, 
    subject, 
    body, 
    file_content, 
    filename, 
    smtp_server, 
    smtp_port, 
    smtp_username, 
    smtp_password
):
    """
    Send an email with an in-memory CSV attachment.

    Args:
        sender_email (str): The sender's email address.
        receiver_emails (list): List of recipient email addresses.
        subject (str): The subject of the email.
        body (str): The body text of the email.
        file_content (StringIO): The content of the CSV file to attach.
        filename (str): The name of the file to attach.
        smtp_server (str): The SMTP server to use for sending the email.
        smtp_port (int): The SMTP port to use.
        smtp_username (str): The SMTP username.
        smtp_password (str): The SMTP password.
        
    Returns:
        bool: True if the email was sent successfully, False otherwise.
    """
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = ", ".join(receiver_emails)
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'html'))

        # Attaching the file content (in-memory file)
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(file_content.getvalue())
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition', 
            f'attachment; filename={filename}'
        )
        msg.attach(part)

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)

        print(f"Email sent successfully to {', '.join(receiver_emails)}")
        return True
    except Exception as e:
        print(f"An error occurred while sending email: {e}")
        return False

def main():
    """
    Main function to orchestrate the workflow of fetching feedback data,
    exporting it to an in-memory CSV, and sending it via email.
    """
    output_file_name = f"feedback_data_{current_date}.csv"

    # Fetch feedback data in chunks
    (
        feedback_data, total_number_of_chats, total_number_of_feedback, 
        total_number_of_answers_without_citations, total_number_of_chats_today, 
        total_number_of_feedback_today, total_number_of_answers_without_citations_today
    ) = fetch_feedback_data_in_chunks()

    file_content = export_to_csv(feedback_data)
    
    if not file_content:
        print("Failed to export feedback data")
        return
    
    utc_time = current_time
    if utc_time.tzinfo is None:
        utc_time = utc_time.replace(tzinfo=timezone.utc)  # Make sure it is timezone-aware (UTC)

    # Convert UTC to IST (UTC + 5:30)
    ist_time = utc_time + timedelta(hours=5, minutes=30)
    
    time_info = f"<strong>Time:</strong> {utc_time.time().strftime('%H:%M:%S')} (UTC) / {ist_time.time().strftime('%H:%M:%S')} (IST)"

    subject = f"Feedback Data CSV {current_date}"
    body = f"""
            <html>
            <body>
                <p>Hello,</p>

                <p>Please find the attached feedback data CSV file for GESCO.</p>

                <p><strong>Key Metrics:</strong></p>
                <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 50%;">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Total</th>
                            <th>Today</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Questions Asked</strong></td>
                            <td>{total_number_of_chats}</td>
                            <td>{total_number_of_chats_today}</td>
                        </tr>
                        <tr>
                            <td><strong>Feedback Given</strong></td>
                            <td>{total_number_of_feedback}</td>
                            <td>{total_number_of_feedback_today}</td>
                        </tr>
                        <tr style="color: red;">
                            <td><strong>Answers without Citations</strong></td>
                            <td>{total_number_of_answers_without_citations}</td>
                            <td>{total_number_of_answers_without_citations_today}</td>
                        </tr>
                    </tbody>
                </table>

                <p><strong>Date:</strong> {current_date}<br>
                {time_info}</p>

                <p>Best regards,<br>
                GenRPT Team</p>
            </body>
            </html>
            """

    # Send the email with attachment
    email_success = send_email_with_attachment(
        sender_email=SENDER_EMAIL,
        receiver_emails=convert_str_to_list(RECEIVER_EMAILS),
        subject=subject,
        body=body,
        file_content=file_content,
        filename=output_file_name,
        smtp_server=SMTP_SERVER,
        smtp_port=SMTP_PORT,
        smtp_username=SMTP_USERNAME,
        smtp_password=SMPT_PASSWORD
    )

    if not email_success:
        print("Failed to send email")

if __name__ == "__main__":
    main()
