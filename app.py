import sqlite3
import os
import re
import random
#import mysql.connector
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import PyPDF2
import docx
# import google.generativeai as genai # Removed as we are replacing Gemini
# from google.generativeai.types import GenerateContentResponse # Removed
import csv
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import tempfile
from datetime import datetime
import cv2
import numpy as np
import logging
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from PIL import Image
import io
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing, Rect
import time
import logging
import ast
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from difflib import SequenceMatcher
from tenacity import wait_fixed
from flask import session
# from google.generativeai import GenerativeModel # Removed
import subprocess
import signal
from functools import wraps
import traceback
from flask import request, send_file
import requests
from io import BytesIO
import pandas as pd # Added for get_analysis_data
import speech_recognition as sr
import threading
import queue
import sounddevice as sd
from scipy.io import wavfile
import noisereduce as nr
from flask_session import Session
from datetime import datetime, timedelta
app = Flask(__name__)
app.secret_key = '7340d01377d428f7b9a5608a3a8b46d3'

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Session config
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)
app.config['SESSION_FILE_THRESHOLD'] = 100
Session(app)

app.config.update(
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False,  # True in production with HTTPS
    SESSION_COOKIE_HTTPONLY=True,
    PERMANENT_SESSION_LIFETIME=timedelta(hours=2)
)

ELEVENLABS_API_KEY = "sk_1023bd0bdb0b18345fcfa94e04d372840c84cfc1e2816c99"
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"

# Face++ API Configuration
FACEPP_API_KEY = "QuT7DlOQyJokapsSsLGr36TpRb5Fejg3"
FACEPP_API_SECRET = "2yHpn5LRsyHlY_hHtfTKjo8xz10h3mlL"
FACEPP_API_URL = "https://api-us.faceplusplus.com/facepp/v3/"

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Add OpenAI import:
import openai
from dotenv import load_dotenv
load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Directory to store CSV files
CSV_FOLDER = r"F:/INTERNSHIP - IV SEM/interview_15-09-2025/project5_interview/output_csv"
os.makedirs(CSV_FOLDER, exist_ok=True)

# Directory to store captured images
IMAGE_FOLDER = r"F:/INTERNSHIP - IV SEM/interview_15-09-2025/project5_interview/images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Face recognition setup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add these global variables for email verification
email_verification_codes = {}

# Email configuration (replace with your actual email credentials)
EMAIL_ADDRESS = "sharathsivakumar610@gmail.com"
EMAIL_PASSWORD = "nrdz lyxy xekq fszx"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

def check_session(user_id):
    """Simple function to validate user session"""
    if 'user_id' not in session or session['user_id'] != user_id:
        return False
    return True


def openai_generate_content(prompt, model="gpt-3.5-turbo", max_tokens=1000):
    """
    Replace Gemini's generate_content with OpenAI's chat completion
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise

def openai_extract_technical_skills(text):
    """
    Replace extract_technical_skills_with_gemini
    """
    prompt = f"""
    The following text is extracted from a resume or CV. Please identify and list only the technical skills mentioned in this text.
    Focus on hard skills related to technology, programming languages, software, tools, and specific technical knowledge.
    Provide the result as a comma-separated list of skills.

    Text:
    {text[:3000]}  # Limit text length to avoid token limits

    Return only the comma-separated list of technical skills, nothing else.
    """

    try:
        response = openai_generate_content(prompt)
        skills_list = [skill.strip() for skill in response.split(',') if skill.strip()]
        logger.info(f"Successfully extracted {len(skills_list)} technical skills: {skills_list}")
        return skills_list
    except Exception as e:
        logger.error(f"Failed to extract technical skills: {str(e)}")
        # Fallback to simple extraction
        return find_technical_skills(text) or []

def openai_generate_questions_from_skills(skills):
    """
    Replace generate_questions_and_answers_from_skills
    """
    if not skills:
        return []

    prompt = f"""Given the following technical skills: {', '.join(skills)},
    generate exactly 5 interview questions that would be appropriate for assessing
    a candidate's proficiency in these skills. For each question, also provide
    an answer in 40-50 words. The questions should be challenging but fair,
    and should help evaluate both theoretical knowledge and practical application
    of these skills. Format the output as follows:

    Q1: [Question 1]
    A1: [Answer 1 (40-50 words)]

    Q2: [Question 2]
    A2: [Answer 2 (40-50 words)]

    Q3: [Question 3]
    A3: [Answer 3 (40-50 words)]

    Q4: [Question 4]
    A4: [Answer 4 (40-50 words)]

    Q5: [Question 5]
    A5: [Answer 5 (40-50 words)]

    Make sure to generate exactly 5 questions and answers."""

    try:
        response = openai_generate_content(prompt, max_tokens=1500)
        qa_pairs = parse_qa_pairs(response)
        logger.info(f"Generated {len(qa_pairs)} Q&A pairs")
        return qa_pairs
    except Exception as e:
        logger.error(f"Failed to generate questions and answers: {str(e)}")
        return []

def parse_qa_pairs(text):
    """
    Parse Q&A pairs from OpenAI response with better error handling
    """
    qa_pairs = []
    lines = text.split('\n')
    current_question = ""
    current_answer = ""

    for line in lines:
        line = line.strip()

        # Handle question lines (Q1:, Q2:, etc.)
        if re.match(r'^Q\d+:', line):
            if current_question and current_answer:
                qa_pairs.append((current_question, current_answer))
            parts = line.split(':', 1)
            if len(parts) > 1:
                current_question = parts[1].strip()
            else:
                current_question = line
            current_answer = ""

        # Handle answer lines (A1:, A2:, etc.)
        elif re.match(r'^A\d+:', line):
            parts = line.split(':', 1)
            if len(parts) > 1:
                current_answer = parts[1].strip()
            else:
                current_answer = line

    # Add the last pair
    if current_question and current_answer:
        qa_pairs.append((current_question, current_answer))

    logger.info(f"Parsed {len(qa_pairs)} Q&A pairs from response")
    return qa_pairs[:5]  # Return up to 5 pairs

@app.route('/check-email', methods=['GET'])
def check_email():
    email = request.args.get('email')
    
    if not email:
        return jsonify({'exists': False})
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "SELECT * FROM details WHERE email = ?"
        cursor.execute(query, (email,))
        existing_user = cursor.fetchone()
        conn.close()
        
        return jsonify({'exists': existing_user is not None})
    except Exception as e:
        print(f"Error checking email: {str(e)}")
        return jsonify({'exists': False})

@app.route('/send-verification-code', methods=['POST'])
def send_verification_code():
    data = request.get_json()
    email = data.get('email')
    code = data.get('code')

    if not email or not code:
        return jsonify({'success': False, 'error': 'Email and code are required'})

    try:
        # Check if email already exists
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "SELECT * FROM details WHERE email = ?"
        cursor.execute(query, (email,))
        existing_user = cursor.fetchone()
        conn.close()

        if existing_user:
            return jsonify({'success': False, 'error': 'Email already registered'})

        # Store the verification code with timestamp
        email_verification_codes[email] = {
            'code': code,
            'timestamp': time.time()
        }

        # Send the email
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = email
        msg['Subject'] = 'Email Verification Code'

        body = f"""
        Your verification code is: {code}

        This code will expire in 1 minute.

        If you didn't request this code, please ignore this email.
        """

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        return jsonify({'success': True})

    except Exception as e:
        print(f"Error sending verification email: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})
@app.route('/logout')
def logout():
    session.clear()  # Completely clear the session
    return redirect(url_for('home'))

@app.route('/forgot-userid', methods=['POST'])
def forgot_userid():
    data = request.get_json()
    email = data.get('email')

    if not email:
        return jsonify({'success': False, 'error': 'Email is required'})

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if email exists
        query = "SELECT user_id FROM details WHERE email = ?"
        cursor.execute(query, (email,))
        user = cursor.fetchone()

        if not user:
            return jsonify({'success': False, 'error': 'Email not found'})

        user_id = user['user_id']

        # Send email with user ID
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = email
        msg['Subject'] = 'Your User ID'

        body = f"""
        Your User ID is: {user_id}

        You can use this User ID to login to your account.

        If you didn't request this information, please ignore this email.
        """

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        return jsonify({'success': True})

    except Exception as e:
        print(f"Error sending user ID email: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})
    finally:
        conn.close()

@app.route('/send-password-reset-code', methods=['POST'])
def send_password_reset_code():
    data = request.get_json()
    user_id = data.get('user_id')

    if not user_id:
        return jsonify({'success': False, 'error': 'User ID is required'})

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if user ID exists and get email
        query = "SELECT email FROM details WHERE user_id = ?"
        cursor.execute(query, (user_id,))
        user = cursor.fetchone()

        if not user:
            return jsonify({'success': False, 'error': 'User ID not found'})

        email = user['email']

        # Generate a random 6-digit code
        reset_code = str(random.randint(100000, 999999))

        # Store the reset code with timestamp (in a real app, use a database)
        # For simplicity, we'll just return it to the frontend
        # In production, store this in a database with an expiration time

        # Send email with reset code
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = email
        msg['Subject'] = 'Password Reset Code'

        body = f"""
        Your password reset code is: {reset_code}

        This code will expire in 10 minutes.

        If you didn't request a password reset, please ignore this email.
        """

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        return jsonify({'success': True, 'code': reset_code})

    except Exception as e:
        print(f"Error sending password reset email: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})
    finally:
        conn.close()

@app.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json()
    user_id = data.get('user_id')
    new_password = data.get('new_password')

    if not user_id or not new_password:
        return jsonify({'success': False, 'error': 'User ID and new password are required'})

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Update password in database
        query = "UPDATE details SET password = ? WHERE user_id = ?"
        cursor.execute(query, (new_password, user_id))

        if cursor.rowcount == 0:
            return jsonify({'success': False, 'error': 'User not found'})

        conn.commit()
        return jsonify({'success': True})

    except Exception as e:
        print(f"Error resetting password: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})
    finally:
        conn.close()


def generate_pdf(user_id, results):
    pdf_filename = os.path.join(CSV_FOLDER, f'{user_id}_current_interview_results.pdf')
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=24, textColor=colors.darkblue, spaceAfter=30)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading1'], fontSize=18, textColor=colors.darkblue, spaceAfter=12)
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=12, spaceAfter=6)
    question_style = ParagraphStyle('Question', parent=styles['Normal'], fontSize=14, textColor=colors.darkgreen, spaceBefore=12)
    answer_style = ParagraphStyle('Answer', parent=styles['Normal'], fontSize=12, leftIndent=20, textColor=colors.black)
    remark_style = ParagraphStyle('Remark', parent=styles['Normal'], fontSize=12, textColor=colors.red, spaceBefore=6)
    feedback_style = ParagraphStyle('Feedback', parent=styles['Normal'], fontSize=12, textColor=colors.purple, spaceBefore=6, leftIndent=20)

    content = []
    content.append(Paragraph("CURRENT INTERVIEW RESULTS", title_style))
    content.append(Spacer(1, 2*inch))
    content.append(Paragraph(f"User ID: {user_id}", normal_style))
    content.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", normal_style))
    content.append(PageBreak())

    for result in results:
        content.append(Paragraph(result['question'], question_style))
        content.append(Paragraph(f"Model Answer: {result['model_answer']}", answer_style))
        content.append(Paragraph(f"Your Answer: {result['user_answer']}", answer_style))
        content.append(Paragraph(f"Similarity Score: {result['similarity_score']}", normal_style))
        content.append(Paragraph(f"Remark: {result['remark']}", remark_style))
        content.append(Paragraph("Feedback:", normal_style))
        feedback_points = result['feedback'].split('\n')
        for point in feedback_points:
            content.append(Paragraph(f"• {point}", feedback_style))
        content.append(Spacer(1, 0.2*inch))

    doc.build(content)
    return pdf_filename

def send_email_with_attachments(to_email, current_pdf_path, all_attempts_pdf_path, user_id):
    from_email = "sharathsivakumar610@gmail.com"  # Replace with your email
    password = "nrdz lyxy xekq fszx"  # Replace with your app password

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = f"Interview Results for User ID: {user_id}"

    body = "Please find attached your current interview results and a summary of all your attempts."
    msg.attach(MIMEText(body, 'plain'))

    # Attach current attempt PDF
    with open(current_pdf_path, "rb") as file:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(file.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(current_pdf_path)}")
    msg.attach(part)

    # Attach all attempts PDF
    with open(all_attempts_pdf_path, "rb") as file:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(file.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(all_attempts_pdf_path)}")
    msg.attach(part)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(filepath):
    if filepath.lower().endswith('.pdf'):
        try:
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text.strip()
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return None
    elif filepath.lower().endswith('.docx'):
        try:
            doc = docx.Document(filepath)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            print(f"Error extracting text from DOCX: {str(e)}")
            return None
    else:
        print(f"Unsupported file format: {filepath}")
        return None
def parse_skills_list(text):
    try:
        # Try to parse as a Python list
        skills = ast.literal_eval(text.strip())
        if isinstance(skills, list):
            return skills
    except (SyntaxError, ValueError):
        pass

    # If parsing fails, fall back to simple string splitting
    return [skill.strip() for skill in text.split(',') if skill.strip()]

# This function is not used in the provided context, but if it were, it would need 'client' defined.
# For now, it's commented out to avoid errors.
# def extract_technical_skills_with_openai(text):
#     prompt = f"""
#     Extract all technical skills (programming languages, frameworks, libraries, tools) from the following resume text.
#     Return them as a comma-separated list:

#     {text}
#     """

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "You are an AI assistant that extracts skills from resumes."},
#             {"role": "user", "content": prompt}
#         ]
#     )

#     skills_text = response.choices[0].message.content
#     return [s.strip() for s in skills_text.split(",") if s.strip()]


def find_technical_skills(text):
    skills = []
    lines = text.split('\n')
    inside_skills_section = False

    for line in lines:
        if any(word in line.lower() for word in ["technical skill", "technical skills", "skill", "skills"]):
            inside_skills_section = True
        elif inside_skills_section:
            potential_skills = line.strip().split()
            for skill in potential_skills:
                if len(skill) > 1 and len(skill) <= 20 and skill not in skills:
                    skills.append(skill)
            if len(skills) >= 4:
                break

    return skills[:4] if skills else None

# This function is not used in the provided context, but if it were, it would need 'client' defined.
# For now, it's commented out to avoid errors.
# @retry(
#     stop=stop_after_attempt(5),
#     wait=wait_exponential(multiplier=1, min=4, max=60),
#     retry=retry_if_exception_type((SyntaxError, ValueError))
# )
# def generate_questions_and_answers_from_skills(skills):
#     prompt = f"""
#     Based on these skills: {', '.join(skills)},
#     generate 5 interview questions with answers (40–50 words each).
#     Format:
#     Q1: ...
#     A1: ...
#     Q2: ...
#     A2: ...
#     """

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "You are an AI interviewer."},
#             {"role": "user", "content": prompt}
#         ]
#     )

#     lines = response.choices[0].message.content.split("\n")
#     qa_pairs = []
#     question, answer = None, None

#     for line in lines:
#         if line.startswith("Q"):
#             if question and answer:
#                 qa_pairs.append((question, answer))
#             question = line.split(":", 1)[-1].strip()
#         elif line.startswith("A"):
#             answer = line.split(":", 1)[-1].strip()

#     if question and answer:
#         qa_pairs.append((question, answer))

#     return qa_pairs[:5]


def save_questions_and_answers_to_csv(qa_pairs, user_id, attempt_number, csv_filename):
    filepath = os.path.join(CSV_FOLDER, csv_filename)
    with open(filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Question Number', 'Question', 'Model Answer', 'User Answer'])  # CSV header
        for i, (question, answer) in enumerate(qa_pairs, 1):
            writer.writerow([f"Question {i}", question, answer, ''])
    return filepath

def user_has_face_capture(user_id):
    """Check if user has a valid faceset_token."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = """
            SELECT faceset_token FROM face_capture_sessions 
            WHERE user_id = ? ORDER BY capture_time DESC LIMIT 1
        """
        cursor.execute(query, (user_id,))
        result = cursor.fetchone()
        conn.close()
        return result and result[0] is not None
    except Exception as e:
        logger.error(f"Error checking face capture: {e}")
        return False

def add_missing_faceset_token():
    """Helper to add faceset_token if missing."""
    conn = sqlite3.connect('reg.db')
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA table_info(face_capture_sessions)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'faceset_token' not in columns:
            cursor.execute("ALTER TABLE face_capture_sessions ADD COLUMN faceset_token TEXT")
            conn.commit()
            logger.info("Added faceset_token column.")
    except Exception as e:
        logger.error(f"Failed to add faceset_token: {e}")
    finally:
        conn.close()

def facepp_request(endpoint, data):
    """Helper to make authenticated POST to Face++ API."""
    url = f"{FACEPP_API_URL}{endpoint}"
    try:
        response = requests.post(url, data=data)
        result = response.json()
        if response.status_code != 200:
            logger.error(f"Face++ API error at {endpoint}: {response.status_code} - {result}")
            print(f"DEBUG: Full response for {endpoint}: {result}")
            return None
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Face++ API request failed at {endpoint}: {e}")
        return None

def detect_face_in_image(image_path_or_url):
    """Detect face; accepts path or URL."""
    data = {
        'api_key': FACEPP_API_KEY,
        'api_secret': FACEPP_API_SECRET,
        'return_attributes': 'none'
    }
    if image_path_or_url.startswith('http'):
        data['image_url'] = image_path_or_url
    else:
        with open(image_path_or_url, 'rb') as f:
            data['image_base64'] = base64.b64encode(f.read()).decode('utf-8')
    
    result = facepp_request('detect', data)
    if result and 'faces' in result and len(result['faces']) > 0:
        return result['faces'][0]['face_token']
    return None
    
# Test Face++ keys on startup (optional; remove in production)
def test_facepp_keys():
    """Quick test: Detect on a sample image URL."""
    data = {
        'api_key': FACEPP_API_KEY,
        'api_secret': FACEPP_API_SECRET,
        'image_url': 'https://www.faceplusplus.com/static/img/fpp-v3/head-portrait.jpg'  # Sample public image
    }
    result = facepp_request('detect', data)
    if result and 'faces' in result:
        logger.info("Face++ keys valid!")
        return True
    else:
        logger.error("Face++ keys invalid! Check credentials.")
        print("DEBUG: Test detect response:", result)
        return False

# Call on startup
test_facepp_keys()

def detect_face_in_image(image_path_or_url):
    """Detect face; accepts path or URL."""
    data = {
        'api_key': FACEPP_API_KEY,
        'api_secret': FACEPP_API_SECRET,
        'return_attributes': 'none'
    }
    if image_path_or_url.startswith('http'):
        data['image_url'] = image_path_or_url
    else:
        with open(image_path_or_url, 'rb') as f:
            data['image_base64'] = base64.b64encode(f.read()).decode('utf-8')
    
    result = facepp_request('detect', data)
    if result and 'faces' in result and len(result['faces']) > 0:
        return result['faces'][0]['face_token']
    return None

@app.route('/start_face_capture/<user_id>', methods=['GET', 'POST'])
def start_face_capture(user_id):
    if not check_session(user_id):
        return redirect(url_for('login'))
    if request.method == 'GET':
        return render_template('before.html', user_id=user_id)
    
    if request.method == 'POST':
        # Receive 3 base64 images from JS
        images_b64 = [
            request.form.get(f'image_{i}') for i in range(3)
        ]
        if len(images_b64) < 3 or any(not img for img in images_b64):
            return jsonify({'success': False, 'error': 'Insufficient images'}), 400
        
        session_id = str(uuid.uuid4())
        capture_time = datetime.now()
        
        # Temp save & detect faces
        temp_paths = []
        face_tokens = []
        for i, img_b64 in enumerate(images_b64):
            if img_b64.startswith('data:image'):
                img_b64 = img_b64.split(',')[1]  # Clean prefix
            img_data = base64.b64decode(img_b64)
            img_path = os.path.join(IMAGE_FOLDER, f"{user_id}_capture_{i}_{session_id}.jpg")
            with open(img_path, 'wb') as f:
                f.write(img_data)
            temp_paths.append(img_path)
            
            # Detect face
            face_token = detect_face_in_image(img_path)
            if face_token:
                face_tokens.append(face_token)
            os.remove(img_path)  # Clean up
        
        if len(face_tokens) < 2:
            return jsonify({'success': False, 'error': f'Only {len(face_tokens)} faces detected. Try again.'}), 400
        
        # Check for existing faceset
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT faceset_token FROM face_capture_sessions WHERE user_id = ? LIMIT 1", (user_id,))
        existing = cursor.fetchone()
        faceset_token = existing[0] if existing and existing[0] else None
        
        if not faceset_token:
            # Create new faceset with safer params
            display_name = f"faceset_{user_id[:8]}"  # Short & safe (max 100 chars, alphanumeric)
            tags = ""  # Empty to avoid issues; add later if needed
            data = {
                'api_key': FACEPP_API_KEY,
                'api_secret': FACEPP_API_SECRET,
                'display_name': display_name,
                'tags': tags
            }
            result = facepp_request('faceset/create', data)
            if not result or 'faceset_token' not in result:
                conn.close()
                return jsonify({'success': False, 'error': 'Failed to create faceset. Check API keys.'}), 500
            faceset_token = result['faceset_token']
            logger.info(f"Created faceset {faceset_token} for {user_id}")
        
        # Add faces to set
        data = {
            'api_key': FACEPP_API_KEY,
            'api_secret': FACEPP_API_SECRET,
            'faceset_token': faceset_token,
            'face_tokens': ','.join(face_tokens[:100])  # Limit to 100 per call
        }
        result = facepp_request('faceset/addface', data)
        if not result:
            conn.close()
            return jsonify({'success': False, 'error': 'Failed to add faces to set.'}), 500
        
        # Store in DB (overwrite if existing)
        cursor.execute("""
            INSERT OR REPLACE INTO face_capture_sessions (user_id, session_id, capture_time, faceset_token)
            VALUES (?, ?, ?, ?)
        """, (user_id, session_id, capture_time, faceset_token))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'faceset_token': faceset_token, 'message': 'Face capture completed!'})
    
@app.route('/verify_face/<user_id>', methods=['POST'])
def verify_face(user_id):
    if not check_session(user_id):
        return redirect(url_for('login'))
    # Receive single base64 image
    image_b64 = request.form.get('image')
    if not image_b64:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    # Clean base64 if it has data URL prefix
    if image_b64.startswith('data:image'):
        image_b64 = image_b64.split(',')[1]
    img_data = base64.b64decode(image_b64)
    img_path = os.path.join(IMAGE_FOLDER, f"{user_id}_verify_{uuid.uuid4()}.jpg")
    with open(img_path, 'wb') as f:
        f.write(img_data)
    
    # Detect face_token
    face_token = detect_face_in_image(img_path)
    os.remove(img_path)
    if not face_token:
        return jsonify({'success': False, 'error': 'No face detected'}), 400
    
    # Get faceset_token
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT faceset_token FROM face_capture_sessions WHERE user_id = ? LIMIT 1", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if not row or not row[0]:
        return jsonify({'success': False, 'error': 'No stored faceset'}), 400
    
    faceset_token = row[0]
    
    # Verify
    data = {
        'api_key': FACEPP_API_KEY,
        'api_secret': FACEPP_API_SECRET,
        'faceset_token': faceset_token,
        'face_token': face_token
    }
    result = facepp_request('faceset/verify', data)
    if result and 'results' in result and len(result['results']) > 0:
        confidence = result['results'][0].get('confidence', 0)
        is_match = confidence > 70  # Threshold
        return jsonify({'success': True, 'confidence': confidence, 'is_match': is_match})
    
    return jsonify({'success': False, 'error': 'Verification failed'}), 400

def get_user_attempt_number(user_id):
    conn = sqlite3.connect('reg.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM interview_attempts WHERE user_id = ?", (user_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count

def increment_user_attempt(user_id):
    current_attempt = get_user_attempt_number(user_id)
    new_attempt = current_attempt + 1
    csv_filename = f'{user_id}_questions_answers_attempt_{new_attempt}.csv'

    conn = sqlite3.connect('reg.db')
    cursor = conn.cursor()
    attempt_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cursor.execute(
        "INSERT INTO interview_attempts (user_id, attempt_number, attempt_date, csv_filename) VALUES (?, ?, ?, ?)",
        (user_id, new_attempt, attempt_date, csv_filename)
    )

    conn.commit()
    cursor.close()
    conn.close()

    return new_attempt, csv_filename

def get_questions_for_current_attempt(user_id):
    attempt_number = get_user_attempt_number(user_id)
    csv_filename = os.path.join(CSV_FOLDER, f'{user_id}_questions_answers_attempt_{attempt_number}.csv')
    questions = []

    try:
        with open(csv_filename, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            questions = [row['Question'] for row in reader]
    except FileNotFoundError:
        logger.error(f"CSV file not found for user {user_id}, attempt {attempt_number}")

    return questions

# New function to generate PDF for all previous attempts
def generate_all_attempts_pdf(user_id):
    pdf_filename = os.path.join(CSV_FOLDER, f'{user_id}_all_interview_results.pdf')
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

    # Styles (same as in the original generate_pdf function)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=24, textColor=colors.darkblue, spaceAfter=30)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading1'], fontSize=18, textColor=colors.darkblue, spaceAfter=12)
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=12, spaceAfter=6)
    question_style = ParagraphStyle('Question', parent=styles['Normal'], fontSize=14, textColor=colors.darkgreen, spaceBefore=12)
    answer_style = ParagraphStyle('Answer', parent=styles['Normal'], fontSize=12, leftIndent=20, textColor=colors.black)
    remark_style = ParagraphStyle('Remark', parent=styles['Normal'], fontSize=12, textColor=colors.red, spaceBefore=6)
    feedback_style = ParagraphStyle('Feedback', parent=styles['Normal'], fontSize=12, textColor=colors.purple, spaceBefore=6, leftIndent=20)

    content = []
    content.append(Paragraph("ALL INTERVIEW RESULTS", title_style))
    content.append(Spacer(1, 2*inch))
    content.append(Paragraph(f"User ID: {user_id}", normal_style))
    content.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", normal_style))
    content.append(PageBreak())

    current_attempt = get_user_attempt_number(user_id)
    for attempt in range(1, current_attempt + 1):
        csv_filename = os.path.join(CSV_FOLDER, f'{user_id}_questions_answers_attempt_{attempt}.csv')

        if os.path.exists(csv_filename):
            content.append(Paragraph(f"Attempt {attempt}", heading_style))

            with open(csv_filename, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    content.append(Paragraph(row['Question'], question_style))
                    content.append(Paragraph(f"Model Answer: {row['Model Answer']}", answer_style))
                    content.append(Paragraph(f"User Answer: {row['User Answer']}", answer_style))

                    # Calculate similarity and get remark
                    similarity = calculate_similarity(row['Model Answer'], row['User Answer'])
                    remark = get_remark(similarity)

                    content.append(Paragraph(f"Similarity Score: {similarity:.2f}%", normal_style))
                    content.append(Paragraph(f"Remark: {remark}", remark_style))
                    content.append(Spacer(1, 0.2*inch))

            content.append(PageBreak())

    doc.build(content)
    return pdf_filename


TIMEOUT_SECONDS = 10
SUPPORTED_LANGUAGES = ['python', 'cpp', 'java', 'html']
language_map = {
    'python': 'python',
    'cpp': 'c_cpp', 
    'java': 'java',
    'html': 'html'
}


def handle_timeout(signum, frame):
    raise TimeoutError("Code execution timed out")

def with_error_handling(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            error_message = str(e)
            traceback.print_exc()  # For server-side logging
            return jsonify({
                'error': error_message,
                'has_error': True
            })
    return decorated_function

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form.get('user_id')  # From form
        password = request.form.get('password')

        print(f"Login attempt: user_id={user_id}, password provided={bool(password)}")  # Debug

        if not user_id or not password:
            flash('User ID and password are required.', 'error')
            return redirect(url_for('login'))

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT password FROM details WHERE user_id = ?", (user_id,))
            user = cursor.fetchone()
            conn.close()

            if user and user['password'] == password:
                session['user_id'] = user_id
                flash('Login successful! Welcome back.', 'success')
                return redirect(url_for('both', user_id=user_id))  # To dashboard or home
            else:
                flash('Invalid User ID or password. Please try again.', 'error')
                logger.error(f"Login failed for user_id: {user_id} - User exists: {user is not None}, Password match: {user['password'] == password if user else False}")

        except Exception as e:
            flash('Login failed. Please try again.', 'error')
            logger.error(f"Login error: {e}")

    # GET: Render login form
    return render_template('sec.html')  # Your login template (sec.html from docs)
    

@app.route('/remove_skill/<user_id>', methods=['POST'])
def remove_skill(user_id):
    data = request.get_json()
    skill_to_remove = data.get("skill")

    if not skill_to_remove:
        return jsonify({"success": False, "message": "No skill provided"}), 400

    # If you store skills in session
    skills = session.get('skills', [])

    if skill_to_remove in skills:
        skills.remove(skill_to_remove)
        session['skills'] = skills
        return jsonify({"success": True, "message": f"Removed {skill_to_remove}"})
    else:
        return jsonify({"success": False, "message": "Skill not found"}), 404
    
    
@app.route('/check-user-id', methods=['GET'])
def check_user_id():
    user_id = request.args.get('user_id')

    # Establish SQLite connection
    conn = sqlite3.connect('reg.db')  # or the full path to your DB file
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM details WHERE user_id = ?"
    cursor.execute(query, (user_id,))
    result = cursor.fetchone()

    cursor.close()
    conn.close()

    if result:
        return f"User {user_id} exists."
    else:
        return f"User {user_id} not found."

@app.route('/registration-success')
def registration_success():
    user_id = request.args.get('user_id')
    if not user_id:
        flash('Registration failed. Please try again.', 'error')
        return redirect(url_for('register'))
    return render_template('success.html', user_id=user_id)

def get_db_connection():
    conn = sqlite3.connect('reg.db')
    conn.row_factory = sqlite3.Row
    return conn

def generate_unique_user_id():
    """Generate a unique user_id in the format LPxxxxx (e.g., LP00001), based on the highest existing ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM details ORDER BY user_id DESC LIMIT 1")
    last_user = cursor.fetchone()

    if last_user and last_user['user_id'].startswith('LP'):
        last_number = int(last_user['user_id'].replace('LP', '').lstrip('0') or 0)
        new_number = last_number + 1
    else:
        new_number = 1  # Start with LP00001 if no records exist

    # Format with leading zeros to ensure five digits
    user_id = f"LP{new_number:05d}"
    conn.close()
    return user_id

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        password = request.form.get('password')

        print(f"Processed data: name={name}, email={email}, phone={phone}")  # Keep your debug print

        if not all([name, email, phone, password]):
            flash('All fields are required.', 'error')
            return redirect(url_for('register'))

        # Check email verification
        verified_email = session.get('verified_email')
        if verified_email != email:
            flash('Please verify your email first.', 'error')
            return redirect(url_for('register'))

        try:
            # Generate unique user_id with LPxxxxx format
            user_id = generate_unique_user_id()

            # Store plain password (no hashing)
            plain_password = password

            # Insert into DB
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO details (name, email, user_id, password, phone)
                VALUES (?, ?, ?, ?, ?)
            """, (name, email, user_id, plain_password, phone))
            conn.commit()
            conn.close()

            # Clear verification
            session.pop('verified_email', None)

            # Render success page with user_id
            return render_template('success.html', user_id=user_id)

        except sqlite3.IntegrityError as e:
            flash('Registration failed. Please try again.', 'error')
            logger.error(f"DB IntegrityError during registration: {e}")
        except Exception as e:
            flash('Registration failed. Please try again.', 'error')
            logger.error(f"Registration error: {e}")
        finally:
            if 'conn' in locals():
                conn.close()

    # GET: Render the form
    return render_template('third.html')  # Your registration template

@app.route('/verify-email-code', methods=['POST'])
def verify_email_code():
    data = request.get_json()
    email = data.get('email')
    code = data.get('code')

    if not email or not code:
        return jsonify({'success': False, 'error': 'Email and code are required'})

    # Check if code is valid and not expired (1 min = 60s)
    stored = email_verification_codes.get(email)
    if not stored or stored['code'] != code or (time.time() - stored['timestamp']) > 60:
        return jsonify({'success': False, 'error': 'Invalid or expired code'})

    # Mark as verified in session
    session['verified_email'] = email
    # Optional: Clear the code after verification
    del email_verification_codes[email]

    return jsonify({'success': True})

@app.route('/four/<user_id>', methods=['GET', 'POST'])
def four(user_id):
    conn = sqlite3.connect('reg.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    technical_skills = None
    questions_generated = False
    job_description = ""
    face_capture_completed = user_has_face_capture(user_id)
    error_message = None

    # Fetch the existing job description
    cursor.execute("SELECT job_description FROM details WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    if result:
        job_description = result[0] or ""

    if request.method == 'POST':
        if not face_capture_completed:
            return render_template('four.html', user_id=user_id, show_face_capture=True)
        else:
            # Handle job description update
            new_job_description = request.form.get('job_description')
            if new_job_description:
                cursor.execute("UPDATE details SET job_description = ? WHERE user_id = ?", (new_job_description, user_id))
                conn.commit()
                job_description = new_job_description

            if 'file' in request.files:
                file = request.files['file']
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)

                    cursor.execute("UPDATE details SET upload_file = ? WHERE user_id = ?", (filename, user_id))
                    conn.commit()

                    extracted_text = extract_text_from_file(filepath)
                    if extracted_text:
                        try:
                            # Save the extracted resume into the resumes table
                            cursor.execute("INSERT INTO resumes (user_id, resume_text) VALUES (?, ?)", (user_id, extracted_text))
                            conn.commit()

                            # Use OpenAI instead of Gemini
                            logger.info("Extracting technical skills with OpenAI...")
                            technical_skills = openai_extract_technical_skills(extracted_text)
                            logger.info(f"Extracted skills: {technical_skills}")

                            if technical_skills:
                                logger.info("Generating questions with OpenAI...")
                                qa_pairs = openai_generate_questions_from_skills(technical_skills)
                                logger.info(f"Generated {len(qa_pairs)} Q&A pairs")

                                if qa_pairs and len(qa_pairs) >= 3:  # Require at least 3 questions
                                    new_attempt, csv_filename = increment_user_attempt(user_id)
                                    save_questions_and_answers_to_csv(qa_pairs, user_id, new_attempt, csv_filename)
                                    questions_generated = True
                                    flash("Questions generated successfully!")
                                else:
                                    error_message = "Failed to generate enough questions. Please try again."
                            else:
                                error_message = "No technical skills found in the uploaded file. Please upload a technical resume."
                        except Exception as e:
                            error_message = f"An error occurred: {str(e)}. Please try again."
                            logger.error(f"Error in resume processing: {str(e)}")
                    else:
                        error_message = "Unable to extract text from the uploaded file. Please try a different file format."

    cursor.close()
    return render_template('four.html',
                           job_description=job_description,
                           technical_skills=technical_skills,
                           questions_generated=questions_generated,
                           user_id=user_id,
                           face_capture_completed=face_capture_completed,
                           error_message=error_message)

@app.route('/test_openai')
def test_openai():
    try:
        test_prompt = "Hello, are you working?"
        response = openai_generate_content(test_prompt, max_tokens=50)
        return jsonify({"success": True, "response": response})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/check_face_capture/<user_id>')
def check_face_capture(user_id):
    face_capture_completed = user_has_face_capture(user_id)
    return jsonify({'completed': face_capture_completed})

@app.route('/previous_attempts/<user_id>')
def previous_attempts(user_id):
    return render_template('previous_attempts.html', user_id=user_id)

@app.route('/get_previous_attempts/<user_id>')
def get_previous_attempts(user_id):
    try:
        conn = sqlite3.connect('reg.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Use the existing query but calculate percentage from similarity
        query = """
            SELECT attempt_number, attempt_date,
                   COALESCE(overall_percentage, overall_similarity, 0) AS overall_percentage,
                   COALESCE(feedback, '') AS feedback,
                   COALESCE(question_count, 0) AS question_count,
                   COALESCE(duration_minutes, 0) AS duration_minutes,
                   COALESCE(interview_type, 'text') AS interview_type
            FROM interview_attempts
            WHERE user_id = ?
            ORDER BY attempt_number ASC
        """
        cursor.execute(query, (user_id,))
        attempts = cursor.fetchall()
        conn.close()

        return jsonify([
            {
                "attemptNumber": row["attempt_number"],
                "date": row["attempt_date"],
                "overallPercentage": row["overall_percentage"],
                "feedback": row["feedback"],
                "questionCount": row["question_count"],
                "duration": row["duration_minutes"],
                "interviewType": row["interview_type"]
            }
            for row in attempts
        ])

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
def save_attempt_to_database(user_id, attempt_number, overall_percentage, feedback, question_count=0, duration_minutes=0):
    conn = sqlite3.connect('reg.db')
    cursor = conn.cursor()

    # Check if the new columns exist
    cursor.execute("PRAGMA table_info(interview_attempts)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'overall_percentage' in columns:  # Check for overall_percentage
        cursor.execute(
            "SELECT id FROM interview_attempts WHERE user_id = ? AND attempt_number = ?",
            (user_id, attempt_number)
        )
        existing = cursor.fetchone()

        if existing:
            cursor.execute(
                "UPDATE interview_attempts SET overall_percentage = ?, feedback = ?, question_count = ?, duration_minutes = ? WHERE user_id = ? AND attempt_number = ?",
                (overall_percentage, feedback, question_count, duration_minutes, user_id, attempt_number)
            )
        else:
            cursor.execute(
                "INSERT INTO interview_attempts (user_id, attempt_number, attempt_date, overall_percentage, feedback, question_count, duration_minutes, csv_filename) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id,
                    attempt_number,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    overall_percentage,
                    feedback,
                    question_count,
                    duration_minutes,
                    f"{user_id}_questions_answers_attempt_{attempt_number}.csv"
                )
            )
    else:
        # Fallback for older schema
        cursor.execute(
            "SELECT id FROM interview_attempts WHERE user_id = ? AND attempt_number = ?",
            (user_id, attempt_number)
        )
        existing = cursor.fetchone()

        if existing:
            cursor.execute(
                "UPDATE interview_attempts SET csv_filename = ? WHERE user_id = ? AND attempt_number = ?",
                (f"{user_id}_questions_answers_attempt_{attempt_number}.csv", user_id, attempt_number)
            )
        else:
            cursor.execute(
                "INSERT INTO interview_attempts (user_id, attempt_number, attempt_date, csv_filename) "
                "VALUES (?, ?, ?, ?)",
                (
                    user_id,
                    attempt_number,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    f"{user_id}_questions_answers_attempt_{attempt_number}.csv"
                )
            )

    conn.commit()
    conn.close()

@app.route('/before/<user_id>')
def before(user_id):
    if user_has_face_capture(user_id):
        flash('Face capture already completed.')
        return redirect(url_for('four', user_id=user_id))
    return render_template('before.html', user_id=user_id)


def generate_session_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

@app.route('/capture_complete', methods=['POST'])
def capture_complete():
    data = request.get_json()
    session_id = data.get('session_id')
    user_id = data.get('user_id')
    
    if not session_id or not user_id:
        logger.error("No session ID or user ID provided")
        return jsonify({"error": "No session ID or user ID provided"}), 400

    # Create a faceset in Face++
    faceset_token, outer_id = create_faceset_facepp()
    
    if not faceset_token:
        return jsonify({
            "success": False,
            "message": "Failed to create face recognition model."
        }), 500

    # Add all captured faces to the faceset
    face_tokens = session.get('face_tokens', [])
    for face_token in face_tokens:
        if not add_face_to_faceset_facepp(face_token, faceset_token):
            logger.warning(f"Failed to add face token {face_token} to faceset")

    # Connect to SQLite and insert session record
    db = sqlite3.connect('reg.db')
    cursor = db.cursor()

    query = "INSERT INTO face_capture_sessions (user_id, session_id, capture_time, faceset_token) VALUES (?, ?, ?, ?)"
    current_time = datetime.now()
    cursor.execute(query, (user_id, session_id, current_time, faceset_token))
    db.commit()
    cursor.close()
    db.close()

    # Clear face tokens from session
    session.pop('face_tokens', None)
    session.modified = True

    return jsonify({
        "success": True,
        "message": "Face recognition model trained successfully",
        "redirect": url_for('four', user_id=user_id)
    }), 200

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        session_id = request.form.get('session_id')
        if not session_id:
            session_id = generate_session_id()

        session_folder = os.path.join(IMAGE_FOLDER, session_id)
        if not os.path.exists(session_folder):
            os.makedirs(session_folder)

        filename = secure_filename(file.filename)
        filepath = os.path.join(session_folder, filename)
        file.save(filepath)

        # Use Face++ API for face detection
        faces_detected, faces = detect_faces_facepp(filepath)
        
        # Save face tokens in session for later training
        if 'face_tokens' not in session:
            session['face_tokens'] = []
            
        for face in faces:
            session['face_tokens'].append(face['face_token'])
            
        session.modified = True

        return jsonify({
            "success": True,
            "filename": filename,
            "session_id": session_id,
            "faces_detected": faces_detected
        }), 200
    
@app.route('/both/<user_id>')
def both(user_id):
    if not check_session(user_id):
        return redirect(url_for('login'))
    return render_template('both.html', user_id=user_id)

@app.route('/different/<user_id>')
def different(user_id):
    if not check_session(user_id):
        return redirect(url_for('login'))
    return render_template('different.html', user_id=user_id)


# Update the code route to use OpenAI functions
@app.route('/code/<user_id>', methods=['GET', 'POST'])
@with_error_handling
def code(user_id):
    if request.method == 'GET':
        session['user_id'] = user_id
        session['current_question'] = 1
        # Automatically generate first question
        return render_template('code.html', user_id=user_id)
    
    elif request.method == 'POST':
        action = request.json.get('action')
        
        if action == 'generate_question':
            option = request.json.get('option')
            question_number = request.json.get('questionNumber')
            
            language = option if option in SUPPORTED_LANGUAGES else 'python'
            difficulty = 'intermediate'
            
            question = openai_generate_question(language, difficulty, question_number)
            session[f'question_{question_number}'] = question
            
            return jsonify({'question': question})
            
        elif action == 'check_code':
            user_code = request.json.get('code')
            option = request.json.get('option')
            question_number = request.json.get('questionNumber')
            question = session.get(f'question_{question_number}')
            
            if not user_code.strip():
                return jsonify({'output': "No code submitted for checking."})
            
            language = option if option in SUPPORTED_LANGUAGES else 'python'
            output = openai_check_code(user_code, language, question)
            
            return jsonify({'output': output})
            
        elif action == 'submit_code':
            user_code = request.json.get('code')
            option = request.json.get('option')
            question_number = request.json.get('questionNumber')
            question = session.get(f'question_{question_number}')
            
            if not user_code.strip():
                return jsonify({'evaluation': 'Score: 0/100\nNo code submitted.'})
            
            language = option if option in SUPPORTED_LANGUAGES else 'python'
            evaluation = openai_evaluate_code(user_code, language, question)
            
            return jsonify({'evaluation': evaluation})
            
        elif action == 'run_code':
            user_code = request.json.get('code')
            user_input = request.json.get('input')
            language = request.json.get('option')
            
            output, has_error = openai_run_code(user_code, language, user_input)
            
            return jsonify({
                'output': output,
                'has_error': has_error
            })

    return jsonify({'error': 'Invalid request'}), 400

def openai_generate_question(language, difficulty, question_number):
    """Generate a coding question using OpenAI"""
    prompt = f"""
    Generate a {difficulty} level coding question #{question_number} for {language}.
    Requirements:
    1. Suitable for {difficulty} level
    2. Tests problem-solving and {language} knowledge
    3. Clear problem statement
    4. Specific input/output formats
    5. Example test cases
    6. Unique problem (not commonly found)

    Format the question with clear HTML structure:
    <div class="question-section">
        <h3>Problem Description:</h3>
        <p>[Clear and concise problem statement]</p>

        <h3>Input Format:</h3>
        <ul>
            <li>[Input parameter 1 with type and constraints]</li>
            <li>[Input parameter 2 with type and constraints]</li>
            ...
        </ul>

        <h3>Output Format:</h3>
        <p>[Clear description of expected output format]</p>

        <h3>Constraints:</h3>
        <ul>
            <li>[Constraint 1]</li>
            <li>[Constraint 2]</li>
            ...
        </ul>

        <h3>Example:</h3>
        <pre>
Input:
[Sample input exactly as it should be entered]

Output:
[Sample output exactly as it should appear]
        </pre>

        <h3>Explanation:</h3>
        <p>[Step-by-step explanation of how the example input leads to the output]</p>
    </div>
    
    Ensure:
    1. All sections are clearly divided with headers
    2. Input/Output formats are shown exactly as they should be entered/displayed
    3. Constraints are specific and clear
    4. Example includes actual formatting that users should follow
    5. Explanation helps users understand the logic
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a coding interview question generator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error in generate_question: {str(e)}")
        return f"Error generating question: {str(e)}"

def openai_check_code(code, language, question):
    code_lines = code.split('\n')
    line_numbers = [i + 1 for i in range(len(code_lines))]
    
    # Build the lines separately
    formatted_code = "\n".join(f"Line {num}: {line}" for num, line in zip(line_numbers, code_lines))
    
    prompt = f"""
    As an expert {language} programmer, analyze this code line by line for errors.
    Question: {question}

    Analyze each line individually and specify:
    1. Whether there's an error (syntax error, logical error, or potential runtime error)
    2. What the error is (if any)
    3. How to fix it (if there's an error)

    For each line, respond in this exact format:
    Line [number]: [code]
    Status: [ERROR/OK]
    [If ERROR] Issue: [Description of the error]
    [If ERROR] Fix: [How to fix the error]

    Here's the code:
    {formatted_code}

    After the line-by-line analysis, provide:
    Overall Assessment:
    - Total number of errors found
    - General code quality
    - Suggestions for improvement 
    """

    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a code reviewer analyzing code for errors."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error in check_code: {str(e)}")
        return f"Error analyzing code: {str(e)}"

def openai_run_code(code, language, user_input):
    """Run code and predict output using OpenAI"""
    if not code.strip():
        return "No code submitted. Please enter some code before running.", True

    prompt = f"""
    Given this {language} code and input, tell me what would be the output.
    If there are any syntax errors or runtime errors, respond with "CHECK THE CODE! THERE ARE ERRORS IN THE CODE"
    Otherwise, just show the exact output the code would produce.

    CODE:
    {code}

    INPUT:
    {user_input}

    Respond only with the output or error message, nothing else.
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You analyze code and predict its output."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        output = response.choices[0].message.content.strip()
        has_error = "CHECK THE CODE" in output
        return output, has_error
    except Exception as e:
        logger.error(f"OpenAI API error in run_code: {str(e)}")
        return "An error occurred while analyzing the code", True

def openai_evaluate_code(code, language, question):
    """Evaluate code submission using OpenAI"""
    if not code.strip():
        return 'Score: 0/100\nNo code submitted.'
    
    prompt = f"""
    Evaluate this {language} solution:
    Question: {question}
    Code: {code}

    Provide:
    1. Score (0-100)
    2. Detailed explanation
    3. Pros/Cons
    4. Suggestions

    Scoring:
    0-5: Irrelevant/wrong language
    6-10: Relevant but incorrect
    11-30: Some understanding
    31-50: Partially solves
    51-70: Mostly solves
    71-90: Solves with minor issues
    91-100: Perfect solution

    Format:
    Score: [X]/100
    Explanation: [Details]
    Pros:
    - [Pro points]
    Cons:
    - [Con points]
    Feedback:
    [Additional comments]
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You evaluate code solutions for technical interviews."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error in evaluate_code: {str(e)}")
        return f"Error evaluating code: {str(e)}"
    
# @with_error_handling
# def submit_code():
#     user_code = request.json.get('code')
#     option = request.json.get('option')
#     question_number = request.json.get('questionNumber')
#     question = session.get(f'question_{question_number}')
    
#     if not user_code.strip():
#         return jsonify({'evaluation': 'Score: 0/100\nNo code submitted.'})
    
#     language = option if option in SUPPORTED_LANGUAGES else 'python'
    
#     prompt = f"""
#     Evaluate this {language} solution:
#     Question: {question}
#     Code: {user_code}

#     Provide:
#     1. Score (0-100)
#     2. Detailed explanation
#     3. Pros/Cons
#     4. Suggestions

#     Scoring:
#     0-5: Irrelevant/wrong language
#     6-10: Relevant but incorrect
#     11-30: Some understanding
#     31-50: Partially solves
#     51-70: Mostly solves
#     71-90: Solves with minor issues
#     91-100: Perfect solution

#     Format:
#     Score: [X]/100
#     Explanation: [Details]
#     Pros:
#     - [Pro points]
#     Cons:
#     - [Con points]
#     Feedback:
#     [Additional comments]
#     """
    
#     response = model.generate_content(prompt)
#     return jsonify({'evaluation': response.text})


# Helper functions
def run_python_code(file_path, input_data):
    return subprocess.Popen(
        ['python', file_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

def run_cpp_code(file_path, input_data):
    # Compile
    compile_result = subprocess.run(
        ['g++', file_path, '-o', f'{file_path}.out'],
        capture_output=True,
        text=True
    )
    if compile_result.returncode != 0:
        raise Exception("Compilation error")

    # Run
    return subprocess.Popen(
        [f'{file_path}.out'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

def run_java_code(file_path, input_data):
    # Compile
    compile_result = subprocess.run(
        ['javac', file_path],
        capture_output=True,
        text=True
    )
    if compile_result.returncode != 0:
        raise Exception("Compilation error")

    # Run
    return subprocess.Popen(
        ['java', '-cp', os.path.dirname(file_path), os.path.splitext(os.path.basename(file_path))[0]],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

def run_test_cases(file_path, language, test_cases):
    results = []
    for test_case in test_cases:
        if not test_case.strip():
            continue

        test_input, expected_output = test_case.strip().split('|||')

        if language == 'python':
            process = run_python_code(file_path, test_input)
        elif language == 'cpp':
            process = run_cpp_code(file_path, test_input)
        else:  # java
            process = run_java_code(file_path, test_input)

        try:
            stdout, stderr = process.communicate(input=test_input.strip(), timeout=TIMEOUT_SECONDS)
            if stderr:
                results.append(False)
            else:
                results.append(stdout.strip() == expected_output.strip())
        except:
            results.append(False)

    return results

def cleanup_files(file_path, language):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        if language == 'cpp' and os.path.exists(f'{file_path}.out'):
            os.remove(f'{file_path}.out')
        if language == 'java':
            class_file = f"{os.path.splitext(file_path)[0]}.class"
            if os.path.exists(class_file):
                os.remove(class_file)
    except:
        pass  # Cleanup errors shouldn't affect the response


@app.route('/practical_results/<user_id>', methods=['GET', 'POST'])
@with_error_handling
def practical_results(user_id):
    if request.method == 'POST':
        results = request.get_json()
        session[f'practical_results_{user_id}'] = {
            'language': results.get('language', ''),
            'responses': results.get('responses', [])
        }
        print("✅ Practical results saved:", session[f'practical_results_{user_id}'])
        
        # Generate PDF and send email
        language = results.get('language', '')
        responses = results.get('responses', [])
        
        if responses:  # Only send email if there are results
            # Get user email from database
            conn = sqlite3.connect('reg.db')
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT email FROM details WHERE user_id = ?", (user_id,))
            user = cursor.fetchone()
            conn.close()
            
            if user and user['email']:
                pdf_filename = generate_practical_pdf(user_id, language, responses)
                email_sent = send_practical_results_email(user['email'], pdf_filename, user_id, language)
                
                if email_sent:
                    print("✅ Practical results email sent successfully")
                else:
                    print("❌ Failed to send practical results email")
        
        return jsonify({'status': 'success'})
    
    results = session.get(f'practical_results_{user_id}')
    print("🔍 Retrieved session data:", results)

    if not results or not results.get('responses'):
        return render_template('practical_results.html', language='', responses=[])

    return render_template(
        'practical_results.html',
        language=results.get('language', ''),
        responses=results.get('responses', [])
    )

# Add this function to generate practical results PDF
def generate_practical_pdf(user_id, language, responses):
    pdf_filename = os.path.join(CSV_FOLDER, f'{user_id}_practical_results.pdf')
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=24, textColor=colors.darkblue, spaceAfter=30)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading1'], fontSize=18, textColor=colors.darkblue, spaceAfter=12)
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=12, spaceAfter=6)
    question_style = ParagraphStyle('Question', parent=styles['Normal'], fontSize=14, textColor=colors.darkgreen, spaceBefore=12)
    code_style = ParagraphStyle('Code', parent=styles['Code'], fontSize=10, textColor=colors.black, backColor=colors.lightgrey, borderPadding=5)
    score_style = ParagraphStyle('Score', parent=styles['Normal'], fontSize=12, textColor=colors.darkblue, spaceBefore=12)

    content = []
    content.append(Paragraph("PRACTICAL CODING RESULTS", title_style))
    content.append(Spacer(1, 2*inch))
    content.append(Paragraph(f"User ID: {user_id}", normal_style))
    content.append(Paragraph(f"Language: {language.upper()}", normal_style))
    content.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", normal_style))
    content.append(PageBreak())

    for i, response in enumerate(responses, 1):
        content.append(Paragraph(f"Question {i}", heading_style))
        content.append(Paragraph(response['question'], question_style))

        content.append(Paragraph("Your Code:", normal_style))
        content.append(Paragraph(response['code'], code_style))

        # Extract and display score
        evaluation = response.get('evaluation', '')
        score_line = ""
        for line in evaluation.split('\n'):
            if line.startswith('Score:'):
                score_line = line
                break

        content.append(Paragraph(score_line, score_style))
        content.append(Spacer(1, 0.2*inch))
        content.append(PageBreak())

    doc.build(content)
    return pdf_filename

# Add this function to send practical results email
def send_practical_results_email(to_email, pdf_path, user_id, language):
    from_email = "sharathsivakumar610@gmail.com"  # Replace with your email
    password = "nrdz lyxy xekq fszx"  # Replace with your app password

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = f"Practical Coding Results for User ID: {user_id}"

    body = f"""
    Practical Coding Results for User ID: {user_id}

    Language: {language}
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    Please find attached the detailed results of your practical coding test.

    Thank you for using our testing platform.
    """

    msg.attach(MIMEText(body, 'plain'))

    # Attach PDF
    with open(pdf_path, "rb") as file:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(file.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(pdf_path)}")
    msg.attach(part)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

# Modify the five route to use the current attempt's questions
"""@app.route('/five/<user_id>')
def five(user_id):
    questions = get_questions_for_current_attempt(user_id)

    if not questions:
        flash("No questions found for this attempt. Please generate questions first.")
        return redirect(url_for('four', user_id=user_id))

    # Check if face recognition model exists for this user
    session_id = get_user_session_id(user_id)
    if not session_id:
        flash("Face recognition model not found. Please complete face capture first.")
        return redirect(url_for('before', user_id=user_id))

    return render_template('five.html', user_id=user_id, questions=questions)
"""

@app.route('/speak', methods=['POST'])
def speak():
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        url = f'https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}'

        headers = {
            'xi-api-key': ELEVENLABS_API_KEY,
            'Content-Type': 'application/json'
        }

        payload = {
            'text': text,
            'voice_settings': {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            # Create a BytesIO object from the response content
            audio_data = BytesIO(response.content)

            # Return the audio data with correct headers
            return send_file(
                audio_data,
                mimetype='audio/mpeg',
                as_attachment=False,
                download_name='speech.mp3'
            )
        else:
            logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
            return jsonify({"error": "Failed to generate audio", "details": response.text}), 500

    except Exception as e:
        logger.error(f"Error in speak endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


# Modify the get_questions route to use the current attempt's questions
@app.route('/get_questions/<user_id>', methods=['GET'])
def get_questions(user_id):
    questions = get_questions_for_current_attempt(user_id)
    return jsonify({"questions": questions})

# Modify the save_answer route to use the current attempt's CSV file
@app.route('/save_answer/<user_id>', methods=['POST'])
def save_answer(user_id):
    data = request.get_json()
    question = data['question']
    answer = data['answer']

    attempt_number = get_user_attempt_number(user_id)
    csv_filename = os.path.join(CSV_FOLDER, f'{user_id}_questions_answers_attempt_{attempt_number}.csv')
    updated_rows = []

    # Read and update the CSV file
    with open(csv_filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Question'] == question:
                row['User Answer'] = answer
            updated_rows.append(row)

    # Write the updated content back to the CSV
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['Question Number', 'Question', 'Model Answer', 'User Answer']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    return jsonify({"success": True})

def detect_faces_facepp(image_path):
    """Detect faces using Face++ API"""
    url = FACEPP_API_URL + "detect"
    
    with open(image_path, 'rb') as f:
        files = {'image_file': f}
        data = {
            'api_key': FACEPP_API_KEY,
            'api_secret': FACEPP_API_SECRET,
            'return_landmark': 0,
            'return_attributes': 'none'
        }
        
        try:
            response = requests.post(url, data=data, files=files)
            result = response.json()
            
            if 'faces' in result:
                return len(result['faces']), result.get('faces', [])
            else:
                logger.error(f"Face++ API error: {result.get('error_message', 'Unknown error')}")
                return 0, []
                
        except Exception as e:
            logger.error(f"Error calling Face++ API: {str(e)}")
            return 0, []

def create_faceset_facepp():
    """Create a new faceset in Face++"""
    url = FACEPP_API_URL + "faceset/create"
    
    data = {
        'api_key': FACEPP_API_KEY,
        'api_secret': FACEPP_API_SECRET,
        'display_name': 'InterviewApp_Faceset',
        'outer_id': f'interview_app_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    try:
        response = requests.post(url, data=data)
        result = response.json()
        
        if 'faceset_token' in result:
            return result['faceset_token'], result.get('outer_id')
        else:
            logger.error(f"Failed to create Face++ faceset: {result.get('error_message', 'Unknown error')}")
            return None, None
            
    except Exception as e:
        logger.error(f"Error creating Face++ faceset: {str(e)}")
        return None, None

def add_face_to_faceset_facepp(face_token, faceset_token):
    """Add a face to a Face++ faceset"""
    url = FACEPP_API_URL + "faceset/addface"
    
    data = {
        'api_key': FACEPP_API_KEY,
        'api_secret': FACEPP_API_SECRET,
        'faceset_token': faceset_token,
        'face_tokens': face_token
    }
    
    try:
        response = requests.post(url, data=data)
        result = response.json()
        
        if 'face_added' in result and result['face_added'] > 0:
            return True
        else:
            logger.error(f"Failed to add face to Face++ faceset: {result.get('error_message', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"Error adding face to Face++ faceset: {str(e)}")
        return False

def search_face_in_faceset_facepp(image_path, faceset_token):
    """Search for a face in a Face++ faceset"""
    url = FACEPP_API_URL + "search"
    
    with open(image_path, 'rb') as f:
        files = {'image_file': f}
        data = {
            'api_key': FACEPP_API_KEY,
            'api_secret': FACEPP_API_SECRET,
            'faceset_token': faceset_token,
            'return_result_count': 1
        }
        
        try:
            response = requests.post(url, data=data, files=files)
            result = response.json()
            
            if 'results' in result and len(result['results']) > 0:
                confidence = result['results'][0]['confidence']
                return True, confidence
            else:
                return False, 0
                
        except Exception as e:
            logger.error(f"Error searching face in Face++ faceset: {str(e)}")
            return False, 0
        
def delete_faceset_facepp(faceset_token):
    """Delete a Face++ faceset"""
    url = FACEPP_API_URL + "faceset/delete"
    
    data = {
        'api_key': FACEPP_API_KEY,
        'api_secret': FACEPP_API_SECRET,
        'faceset_token': faceset_token
    }
    
    try:
        response = requests.post(url, data=data)
        result = response.json()
        
        if 'faceset_token' in result:
            return True
        else:
            logger.error(f"Failed to delete Face++ faceset: {result.get('error_message', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"Error deleting Face++ faceset: {str(e)}")
        return False


@app.route('/recognize_face/<user_id>', methods=['POST'])
def recognize_face(user_id):
    image_data = request.json['image']
    image_data = image_data.split(',')[1]  # Remove the "data:image/jpeg;base64," part

    # Save the image to a temporary file
    temp_image_path = os.path.join(IMAGE_FOLDER, f'temp_{user_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')
    
    with open(temp_image_path, 'wb') as f:
        f.write(base64.b64decode(image_data))

    # Get the faceset token for this user
    session_id = get_user_session_id(user_id)
    if not session_id:
        os.remove(temp_image_path)
        return jsonify({"error": "No trained model found for this user"}), 400

    # Get the faceset token from database
    conn = sqlite3.connect('reg.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = "SELECT faceset_token FROM face_capture_sessions WHERE session_id = ?"
    cursor.execute(query, (session_id,))
    result = cursor.fetchone()
    conn.close()
    
    if not result or not result['faceset_token']:
        os.remove(temp_image_path)
        return jsonify({"error": "Face recognition model not found"}), 400
        
    faceset_token = result['faceset_token']

    # Search for the face in the faceset using Face++
    is_user, confidence = search_face_in_faceset_facepp(temp_image_path, faceset_token)
    
    # Clean up temporary file
    os.remove(temp_image_path)

    # For demonstration, we'll just return a simple result
    # In a real application, you might want to process the image to detect face location
    return jsonify({
        "faces": [{
            "isUser": is_user,
            "confidence": confidence,
            "message": "YES IT'S THE USER" if is_user and confidence > 70 else "NO IT'S NOT THE USER"
        }]
    })

def get_user_session_id(user_id):
    conn = sqlite3.connect('reg.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = """
        SELECT session_id, faceset_token
        FROM face_capture_sessions
        WHERE user_id = ?
        ORDER BY capture_time DESC
        LIMIT 1
    """
    cursor.execute(query, (user_id,))
    result = cursor.fetchone()

    cursor.close()
    conn.close()

    return result['session_id'] if result else None

@app.route('/six/<user_id>')
def six(user_id):
    return render_template('six.html', user_id=user_id)

@app.route('/seven/<user_id>')
def seven(user_id):
    return render_template('seven.html', user_id=user_id)

def calculate_similarity(model_answer, user_answer):
    vectorizer = TfidfVectorizer().fit_transform([model_answer, user_answer])
    vectors = vectorizer.toarray()
    cos_sim = cosine_similarity(vectors)
    return cos_sim[0][1] * 100  # Return percentage

def get_remark(similarity_score):
    if similarity_score >= 90:
        return "Excellent"
    elif 80 <= similarity_score < 90:
        return "Very Good"
    elif 70 <= similarity_score < 80:
        return "Good"
    elif 60 <= similarity_score < 70:
        return "Fair"
    elif 50 <= similarity_score < 60:
        return "You can do better"
    else:
        return "Next time try harder"

@app.route('/get_results/<user_id>', methods=['GET'])
def get_results(user_id):
    attempt_number = get_user_attempt_number(user_id)
    csv_filename = os.path.join(CSV_FOLDER, f'{user_id}_questions_answers_attempt_{attempt_number}.csv')
    results = []
    results_with_feedback = []

    # Read the CSV file for the current attempt
    with open(csv_filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            model_answer = row['Model Answer']
            user_answer = row['User Answer']
            similarity = calculate_similarity(model_answer, user_answer)
            remark = get_remark(similarity)

            # Generate feedback using OpenAI
            feedback = openai_generate_feedback(row['Question'], model_answer, user_answer)

            result = {
                'question': row['Question'],
                'model_answer': model_answer,
                'user_answer': user_answer,
                'similarity_score': f'{similarity:.2f}%',
                'remark': remark
            }
            results.append(result)

            result_with_feedback = result.copy()
            result_with_feedback['feedback'] = feedback
            results_with_feedback.append(result_with_feedback)

    # ... rest of the function remains the same
    # The original code had '... rest of the function remains the same' here.
    # Assuming the intent was to return results_with_feedback or similar.
    # For now, I'll just return a success message, as the context doesn't show
    # what happens after results_with_feedback is populated.
    return jsonify({"success": True, "results": results_with_feedback})


def extract_technical_keywords(resume_text):
    """Extract technical keywords from resume to inform question generation"""
    prompt = f"""
    Extract the top 10 most important technical skills, technologies, and programming languages
    from this resume. Return them as a comma-separated list:

    {resume_text}

    Technical keywords:
    """

    try:
        # Changed from model.generate_content to openai_generate_content
        result_text = openai_generate_content(prompt)
        keywords = [kw.strip() for kw in result_text.split(',')]
        return keywords[:10]  # Return top 10
    except Exception as e: # Added specific exception for better logging
        logger.error(f"Error extracting technical keywords with OpenAI: {str(e)}")
        return ["programming", "development", "technical", "software", "coding"]

# Voice-based interview initial question & fallback questions
initial_question = "Tell me about yourself"
fallback_questions = [
    "Tell me about your experience with Python projects.",
    "What challenges have you faced while coding?",
    "Can you describe any group work or team collaboration?",
    "What are your hobbies?",
    "Have you built anything using Flask or Django?",
    "What motivates you to learn Python?",
    "What's your favorite feature in Python?"
]


@app.route('/voice_interview/<user_id>')
def voice_interview(user_id):
    try:
        # Clear previous session data
        session.clear()
        session['user_id'] = user_id
        session['interview_started'] = True
        session['question_count'] = 0
        session['answers'] = []
        session['asked_questions'] = []
        session['all_questions'] = []  # Initialize all_questions
        session['current_round'] = 'technical'  # Track interview round
        session['used_keywords'] = []  # Track used technical keywords

        conn = sqlite3.connect('reg.db')
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT r.resume_text, d.name
            FROM resumes r
            JOIN details d ON r.user_id = d.user_id
            WHERE r.user_id = ?
            ORDER BY r.id DESC LIMIT 1
        """, (user_id,))
        data = cur.fetchone()
        conn.close()

        session['resume_text'] = data['resume_text'] if data else ""
        session['user_name'] = data['name'] if data else "Candidate"

        # Extract resume keywords for later use
        if session['resume_text']:
            session['resume_keywords'] = extract_technical_keywords(session['resume_text'])
        else:
            session['resume_keywords'] = []

        return render_template("voice_interview.html", user_id=user_id)
    except Exception as e:
        logger.error(f"Error in voice_interview: {str(e)}")
        flash("An error occurred while starting the interview")
        return redirect(url_for('four', user_id=user_id))

# Update the initial question generation - ALWAYS start with "Tell me about yourself"
@app.route('/start_interview', methods=['POST'])
def start_interview():
    user_name = session.get('user_name', 'Candidate')

    # FIXED: Always start with "Tell me about yourself" question
    initial_question = f"Tell me about yourself, {user_name}."

    # Initialize session tracking
    session['answers'] = []
    session['all_questions'] = [initial_question]  # Track all questions
    session['asked_questions'] = [initial_question]  # Track asked questions
    session['current'] = 0
    session['last_question'] = initial_question
    session['question_count'] = 1
    session.modified = True

    return jsonify({'question': initial_question})

def openai_generate_next_question(prev_question, prev_answer, is_hr_round=False):
    """Generate next question using OpenAI"""
    prompt = f"""
    You are conducting a {'HR' if is_hr_round else 'technical'} interview.

    Previous question: {prev_question}
    Candidate's answer: {prev_answer}

    Generate the next appropriate interview question that:
    1. Is relevant to the previous answer
    2. Is open-ended and encourages detailed response
    3. Keeps under 20 words
    4. Ends with a question mark
    5. Is unique and not repetitive

    Return only the question, nothing else.
    """

    try:
        response = openai_generate_content(prompt, max_tokens=50)
        question = response.strip()

        if not question.endswith('?'):
            question += '?'

        return question

    except Exception as e:
        logger.error(f"Error generating next question: {str(e)}")
        # Fallback to a general question if OpenAI fails
        return get_fallback_question(is_hr_round)

# This function was not provided in the original context, but is called by openai_generate_next_question
def get_fallback_question(is_hr_round):
    if is_hr_round:
        return random.choice([
            "How do you handle pressure?",
            "Tell me about a time you failed.",
            "What are your strengths?"
        ])
    else:
        return random.choice([
            "What is your favorite programming language?",
            "Describe a challenging technical problem you solved.",
            "How do you debug code?"
        ])


def openai_generate_feedback(question, model_answer, user_answer):
    """Generate feedback using OpenAI"""
    prompt = f"""
    Question: {question}
    Model Answer: {model_answer}
    User Answer: {user_answer}

    Provide 5 short, clear feedback points on how the user could improve their answer.
    Each point should be concise and actionable.
    Return the feedback as bullet points, one per line.
    """

    try:
        response = openai_generate_content(prompt)
        return response.strip()
    except Exception as e:
        logger.error(f"Error generating feedback: {str(e)}")
        return "Error generating feedback. Please try again later."

def openai_generate_overall_feedback(questions_answers):
    """Generate overall feedback using OpenAI"""
    prompt = f"""
    Analyze the following set of interview questions and answers:

    {questions_answers}

    Provide a concise overall feedback (about 100 words) on the candidate's performance,
    highlighting strengths and areas for improvement. Focus on:
    1. Technical knowledge
    2. Communication skills
    3. Problem-solving approach
    4. Areas that need more attention

    Format the feedback as a paragraph without bullet points.
    """

    try:
        response = openai_generate_content(prompt)
        return response.strip()
    except Exception as e:
        logger.error(f"Error generating overall feedback: {str(e)}")
        return "Unable to generate feedback at this time."

def generate_first_technical_question(self_intro_answer, resume_text, user_name):
    """Generate the first technical question based on self-introduction and resume"""
    try:
        prompt = f"""
        The candidate just introduced themselves. Based on their introduction and resume,
        generate the first technical question.

        CANDIDATE'S SELF INTRODUCTION:
        {self_intro_answer}

        RESUME CONTENT:
        {resume_text[:2000] if resume_text else "No resume available"}

        Generate a technical question that:
        1. Builds on what they mentioned in their introduction
        2. Is specific to their technical skills/experience
        3. Is open-ended and encourages detailed response
        4. Keeps under 20 words
        5. Address them as {user_name}
        6. Ends with a question mark
        7. Is unique and not repetitive

        Technical question:
        """
        # Changed from model.generate_content to openai_generate_content
        question = openai_generate_content(prompt)

        # Clean up the response
        question = question.replace('"', '').replace("'", "")
        if not question.endswith('?'):
            question += '?'

        return question

    except Exception as e:
        logger.error(f"Error generating first technical question: {str(e)}")
        # Fallback to a general technical question
        return f"Can you tell me about your most challenging technical project, {user_name}?"

def generate_technical_followup(prev_question, prev_answer, asked_questions, user_name):
    """Generate follow-up technical questions based on previous answers and resume"""
    resume_text = session.get('resume_text', '')
    resume_keywords = session.get('resume_keywords', [])
    used_keywords = session.get('used_keywords', [])

    # If no answer was provided, generate a new question without following up
    if not prev_answer or prev_answer == "(No answer provided)":
        # Use a different approach to generate a fresh question
        available_keywords = [kw for kw in resume_keywords if kw not in used_keywords]

        if available_keywords:
            # Use a new keyword
            focus_keyword = random.choice(available_keywords)
            used_keywords.append(focus_keyword)
            session['used_keywords'] = used_keywords
            session.modified = True

            prompt = f"""
            Based on the candidate's resume, generate a technical interview question focused on {focus_keyword}.

            RESUME CONTEXT:
            {resume_text[:2000] if resume_text else "No resume available"}

            ALREADY ASKED QUESTIONS (avoid these):
            {', '.join(asked_questions[-5:]) if asked_questions else 'None'}

            Generate a technical question that:
            1. Is specifically about {focus_keyword} or related technologies
            2. Is open-ended and encourages detailed response
            3. Keeps under 20 words
            4. Address them as {user_name}
            5. Ends with a question mark
            6. Is different from previously asked questions

            Technical question:
            """
        else:
            # All keywords used, generate a general technical follow-up
            prompt = f"""
            You are conducting a technical interview. Generate the next technical question.

            PREVIOUS QUESTION: {prev_question}
            CANDIDATE'S ANSWER: {prev_answer}

            RESUME CONTENT:
            {resume_text[:2000] if resume_text else "No resume available"}

            ALREADY ASKED QUESTIONS (avoid these):
            {', '.join(asked_questions[-5:]) if asked_questions else 'None'}

            Generate a technical follow-up question that:
            1. Builds naturally on their previous answer
            2. Dives deeper into technical details
            3. References specific technologies/skills from their resume when relevant
            4. Keeps under 20 words
            5. Address them as {user_name}
            6. Ends with a question mark
            7. Is different from previously asked questions

            Follow-up question:
            """
    else: # If an answer was provided, generate a follow-up based on it
        prompt = f"""
        You are conducting a technical interview. Generate the next technical question.

        PREVIOUS QUESTION: {prev_question}
        CANDIDATE'S ANSWER: {prev_answer}

        RESUME CONTENT:
        {resume_text[:2000] if resume_text else "No resume available"}

        ALREADY ASKED QUESTIONS (avoid these):
        {', '.join(asked_questions[-5:]) if asked_questions else 'None'}

        Generate a technical follow-up question that:
        1. Builds naturally on their previous answer
        2. Dives deeper into technical details
        3. References specific technologies/skills from their resume when relevant
        4. Keeps under 20 words
        5. Address them as {user_name}
        6. Ends with a question mark
        7. Is different from previously asked questions

        Follow-up question:
        """

    try:
        # Changed from model.generate_content to openai_generate_content
        question = openai_generate_content(prompt)

        # Clean up
        question = question.replace('"', '').replace("'", "")
        if not question.endswith('?'):
            question += '?'

        return question

    except Exception as e:
        logger.error(f"Error generating follow-up technical question: {str(e)}")
        return get_fallback_technical_question(asked_questions, user_name)

def generate_hr_question(prev_question, prev_answer, asked_questions, user_name):
    """Generate HR questions"""
    try:
        if not prev_answer or len(prev_answer.split()) < 3:
            return get_fallback_hr_question(asked_questions, user_name)

        prompt = f"""
        Generate a follow-up HR question based on this conversation:
        Previous question: {prev_question}
        Candidate answer: {prev_answer}

        Requirements:
        1. Must be relevant to the previous answer
        2. Should explore behavioral aspects, teamwork, or soft skills
        3. Keep it under 20 words
        4. Address the candidate as {user_name}
        5. End with a question mark
        6. Should not be any of these: {', '.join(asked_questions)}
        7. Focus on leadership, problem-solving, or interpersonal skills
        8. Ensure the question is unique and not repetitive
        """
        # Changed from model.generate_content to openai_generate_content
        question = openai_generate_content(prompt)

        if not question.endswith('?'):
            question += '?'

        return question

    except Exception as e:
        logger.error(f"Error generating HR question: {str(e)}")
        return get_fallback_hr_question(asked_questions, user_name)
    
@app.route('/process_audio/<user_id>', methods=['POST'])
def process_audio(user_id):
    try:
        audio_file = request.files['audio']
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            audio_file.save(tmp.name)
            # Optional: Noise reduction
            rate, data = wavfile.read(tmp.name)
            reduced = nr.reduce_noise(y=data, sr=rate)
            wavfile.write(tmp.name, rate, reduced)
            # Recognize
            r = sr.Recognizer()
            with sr.AudioFile(tmp.name) as source:
                audio = r.record(source)
                text = r.recognize_google(audio)  # Or use recognize_sphinx for offline
        os.unlink(tmp.name)
        # Store/process text as needed (e.g., evaluate answer)
        return jsonify({'success': True, 'text': text})
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def get_fallback_technical_question(asked_questions, user_name):
    """Fallback technical questions when generation fails"""
    fallback_tech_questions = [
        f"How do you approach debugging complex code issues, {user_name}?",
        f"What's your process for learning new technologies, {user_name}?",
        f"How do you ensure code quality in your projects, {user_name}?",
        f"Describe your experience with version control systems, {user_name}?",
        f"How do you handle performance optimization, {user_name}?",
        f"What testing strategies do you use in your development, {user_name}?",
        f"How do you stay updated with technology trends, {user_name}?",
        f"Describe a challenging technical problem you solved, {user_name}?"
    ]

    # Get unasked questions first
    asked_set = set(asked_questions)
    available = [q for q in fallback_tech_questions if q not in asked_set]

    return random.choice(available) if available else random.choice(fallback_tech_questions)

def get_fallback_hr_question(asked_questions, user_name):
    """Fallback HR questions"""
    fallback_hr_questions = [
        f"How do you handle pressure and tight deadlines, {user_name}?",
        f"Tell me about a time you disagreed with a team member, {user_name}?",
        f"What motivates you in your work, {user_name}?",
        f"How do you prioritize multiple tasks, {user_name}?",
        f"Describe your ideal work environment, {user_name}?",
        f"How do you handle constructive criticism, {user_name}?",
        f"What are your career goals, {user_name}?",
        f"How do you contribute to team collaboration, {user_name}?"
    ]

    asked_set = set(asked_questions)
    available = [q for q in fallback_hr_questions if q not in asked_set]

    return random.choice(available) if available else random.choice(fallback_hr_questions)


@app.route('/submit_voice_answer', methods=['POST'])
def submit_voice_answer():
    try:
        data = request.get_json()
        answer = data.get('answer', '').strip()
        is_hr_round = data.get('isHrRound', False)

        # Ensure session tracking exists
        if 'all_questions' not in session:
            session['all_questions'] = []
            session['answers'] = []
            session['question_count'] = 0

        # Get the current question (last one asked)
        current_question = session['all_questions'][-1] if session['all_questions'] else ""

        # Store answer with question reference
        session['answers'].append({
            'question': current_question,
            'answer': answer
        })

        # Generate next question using OpenAI
        # The original code called openai_generate_next_question directly here.
        # However, the logic for generating technical/HR questions is split across
        # generate_first_technical_question, generate_technical_followup, and generate_hr_question.
        # This part needs careful handling to ensure the correct function is called.
        # For simplicity, I'll assume a general next question generation, but you might
        # need to implement a state machine or more complex logic here to switch between
        # technical and HR questions based on your interview flow.

        user_name = session.get('user_name', 'Candidate')
        asked_questions = session.get('asked_questions', [])

        if session.get('current_round') == 'technical':
            # If it's the first technical question after self-intro
            if session['question_count'] == 1: # Assuming self-intro is Q1
                next_question = generate_first_technical_question(answer, session.get('resume_text', ''), user_name)
            else:
                next_question = generate_technical_followup(current_question, answer, asked_questions, user_name)
            # Add logic to potentially switch to HR round after N technical questions
            if session['question_count'] >= 5: # Example: switch after 5 technical questions
                session['current_round'] = 'hr'
        elif session.get('current_round') == 'hr':
            next_question = generate_hr_question(current_question, answer, asked_questions, user_name)
        else: # Default or initial state, should ideally be handled by start_interview
            next_question = openai_generate_next_question(current_question, answer, False) # Default to technical if state is unclear


        # Track the new question
        session['all_questions'].append(next_question)
        session['asked_questions'].append(next_question) # Keep track of all asked questions
        session['question_count'] += 1
        session.modified = True

        return jsonify({
            'question': next_question,
            'question_number': session['question_count'],
            'status': 'success'
        })

    except Exception as e:
        logger.error(f"Error in submit_voice_answer: {str(e)}")
        is_hr_round = session.get('current_round') == 'hr'
        fallback_question = get_fallback_hr_question(session.get('asked_questions', []), session.get('user_name', 'Candidate')) if is_hr_round else get_fallback_technical_question(session.get('asked_questions', []), session.get('user_name', 'Candidate'))
        return jsonify({
            'question': fallback_question,
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/end_interview', methods=['POST'])
def end_interview():
    try:
        # Get user_id from session
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'User not identified'}), 400

        # Get all questions and answers
        all_questions = session.get('all_questions', [])
        answers = session.get('answers', [])

        # Create a dictionary of answered questions for easy lookup
        answered_dict = {a['question']: a['answer'] for a in answers}

        results = []
        total_score = 0
        max_possible_score = len(all_questions) * 10  # Each question worth 10 points

        for i, question in enumerate(all_questions):
            answer = answered_dict.get(question, '(No answer provided)')

            # Generate score and feedback
            if answer != '(No answer provided)':
                score = min(10, len(answer.split()) // 3)  # Simple scoring
                # Assuming 'Model answer' is a placeholder, as it's not stored in session for voice interview
                feedback = openai_generate_feedback(question, "A comprehensive and relevant answer.", answer)
            else:
                score = 0
                feedback = "No response was provided for this question."

            results.append({
                'question': question,
                'answer': answer,
                'score': score,
                'feedback': feedback
            })
            total_score += score

        # Calculate overall percentage
        overall_percentage = round((total_score / max_possible_score) * 100, 2) if max_possible_score > 0 else 0

        # Store evaluated answers in session temporarily
        session['evaluated_answers'] = results
        session['total_score'] = total_score
        session['overall_percentage'] = overall_percentage 

        # Save to database - NEW CODE
        attempt_number = get_user_attempt_number(user_id) + 1
        feedback_text = "Voice interview completed. See detailed results for feedback."
        
        # Save the attempt to database
        save_attempt_to_database(
            user_id=user_id,
            attempt_number=attempt_number,
            overall_percentage=overall_percentage,
            feedback=feedback_text,
            question_count=len(all_questions),
            duration_minutes=0  # You might want to track duration
        )
        
        # Also update the interview_attempts table to mark this as a voice interview
        conn = sqlite3.connect('reg.db')
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE interview_attempts SET interview_type = 'voice' WHERE user_id = ? AND attempt_number = ?",
            (user_id, attempt_number)
        )
        conn.commit()
        conn.close()

        # Generate PDF with results
        pdf_filename = generate_voice_results_pdf(user_id, results, total_score)

        # Get user email from database
        conn = sqlite3.connect('reg.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT email FROM details WHERE user_id = ?", (user_id,))
        user = cursor.fetchone()
        conn.close()

        # Send email if user has an email address
        email_sent = False
        if user and user['email']:
            email_sent = send_voice_results_email(user['email'], pdf_filename, user_id, total_score, len(results))

        # Clear interview-specific session data but keep user_id for results page
        interview_keys = ['all_questions', 'answers', 'question_count', 'resume_text',
                         'user_name', 'interview_started', 'asked_questions', 'current_round', 'used_keywords']
        for key in interview_keys:
            session.pop(key, None)

        session.modified = True

        return jsonify({
            'success': True,
            'redirect_url': url_for('voice_results'),
            'email_sent': email_sent
        })

    except Exception as e:
        logger.error(f"Error ending interview: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def recognize_speech_from_mic():
    # Initialize the recognizer
    r = sr.Recognizer()

    # Use the microphone as the source
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    # Recognize speech using Google's Web Speech API
    try:
        text = r.recognize_google(audio)
        print("You said: " + text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""

@app.route('/voice_results')
def voice_results():
    try:
        # Get evaluated answers from session
        evaluated_answers = session.get('evaluated_answers', [])
        total_score = session.get('total_score', 0)
        overall_percentage = session.get('overall_percentage', 0)  # Get the percentage
        user_id = session.get('user_id', '')

        if not evaluated_answers:
            flash("No interview results found. Please complete an interview first.")
            return redirect(url_for('four', user_id=user_id))

        # Clear the remaining session data after getting what we need
        results_keys = ['evaluated_answers', 'total_score', 'overall_percentage']  # Add overall_percentage
        for key in results_keys:
            session.pop(key, None)
        session.modified = True

        return render_template(
            "voice_results.html",
            answers=evaluated_answers,
            total_score=total_score,
            overall_percentage=overall_percentage  # Pass to template
        )

    except Exception as e:
        logger.error(f"Error generating voice results: {str(e)}")
        flash("An error occurred while generating results")
        return redirect(url_for('four', user_id=session.get('user_id', '')))

def generate_voice_results_pdf(user_id, results, total_score):
    pdf_filename = os.path.join(CSV_FOLDER, f'{user_id}_voice_interview_results.pdf')
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=24, textColor=colors.darkblue, spaceAfter=30)
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=12, spaceAfter=6)
    question_style = ParagraphStyle('Question', parent=styles['Normal'], fontSize=14, textColor=colors.darkgreen, spaceBefore=12)
    answer_style = ParagraphStyle('Answer', parent=styles['Normal'], fontSize=12, leftIndent=20, textColor=colors.black)
    feedback_style = ParagraphStyle('Feedback', parent=styles['Normal'], fontSize=12, textColor=colors.purple, spaceBefore=6, leftIndent=20)

    content = []
    content.append(Paragraph("VOICE INTERVIEW RESULTS", title_style))
    content.append(Spacer(1, 2*inch))
    content.append(Paragraph(f"User ID: {user_id}", normal_style))
    content.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", normal_style))
    content.append(Paragraph(f"Total Score: {total_score}/{(len(results) * 10)}", normal_style))
    content.append(PageBreak())

    for result in results:
        content.append(Paragraph(result['question'], question_style))
        content.append(Paragraph(f"Your Answer: {result['answer']}", answer_style))
        content.append(Paragraph(f"Score: {result['score']}/10", normal_style))
        content.append(Paragraph("Feedback:", normal_style))
        feedback_points = result['feedback'].split('\n')
        for point in feedback_points:
            content.append(Paragraph(f"• {point}", feedback_style))
        content.append(Spacer(1, 0.2*inch))

    doc.build(content)
    return pdf_filename


def send_voice_results_email(to_email, pdf_path, user_id, total_score, total_questions):
    from_email = "sharathsivakumar610@gmail.com"  # Replace with your email
    password = "nrdz lyxy xekq fszx"  # Replace with your app password

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = f"Voice Interview Results for User ID: {user_id}"

    body = f"""
    Voice Interview Results for User ID: {user_id}

    Total Score: {total_score}/{total_questions * 10}
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    Please find attached the detailed results of your voice interview.

    Thank you for using our interview platform.
    """

    msg.attach(MIMEText(body, 'plain'))

    # Attach PDF
    with open(pdf_path, "rb") as file:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(file.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(pdf_path)}")
    msg.attach(part)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

def cleanup_interview_session():
    """Clear all interview-related session data"""
    interview_keys = [
        'all_questions', 'answers', 'question_count', 'resume_text',
        'user_name', 'interview_started', 'asked_questions',
        'evaluated_answers', 'total_score', 'current_round', 'used_keywords'
    ]
    for key in interview_keys:
        session.pop(key, None)
    session.modified = True

@app.route('/cleanup_interview_session', methods=['POST'])
def cleanup_interview_session_route():
    """API endpoint to explicitly clean up interview session"""
    try:
        cleanup_interview_session()
        return jsonify({'success': True, 'message': 'Session cleaned up successfully'})
    except Exception as e:
        logger.error(f"Error cleaning up session: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/get_total_attempts/<user_id>')
def get_total_attempts(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM interview_attempts WHERE user_id = ?", (user_id,))
    total_attempts = cursor.fetchone()[0]
    conn.close()
    return jsonify({'total_attempts': total_attempts})

@app.route('/clear_session')
def clear_session():
    # Clear all session data
    session.clear()
    return jsonify({'success': True})

@app.route('/five/<user_id>')
def five(user_id):
    return redirect(url_for('voice_interview', user_id=user_id))

# Aptitude round routes
@app.route('/aptitude/<user_id>')
def aptitude(user_id):
    return render_template('aptitude.html', user_id=user_id)

@app.route('/save_aptitude_results/<user_id>', methods=['POST'])
def save_aptitude_results(user_id):
    try:
        data = request.get_json()
        score = data.get('score')
        total = data.get('total')
        difficulty = data.get('difficulty', 'unknown')
        answers = data.get('answers', [])
        skipped = data.get('skipped', 0)
        
        # Store results in session instead of database
        session['aptitude_results'] = {
            'user_id': user_id,
            'score': score,
            'total': total,
            'difficulty': difficulty,
            'answers': answers,
            'skipped': skipped,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify({
            'success': True, 
            'message': 'Results processed successfully'
        })
    
    except Exception as e:
        print(f"Error processing aptitude results: {str(e)}")
        return jsonify({
            'success': False, 
            'error': str(e),
            'message': 'Failed to process results. Please try again.'
        }), 500
    
import json
from flask import session

@app.route('/generate_aptitude_questions/<user_id>/<difficulty>', methods=['GET'])
def generate_aptitude_questions(user_id, difficulty):
    try:
        # Validate difficulty level
        if difficulty not in ['beginner', 'intermediate', 'hard']:
            return jsonify({
                "success": False,
                "error": "Invalid difficulty level"
            }), 400

        prompt = f"""Generate exactly 30 {difficulty} level aptitude test questions with these specifications:
        
        Include a variety of question types:
        1. Technical questions (programming concepts, algorithms, data structures)
        2. Logical reasoning questions
        3. Quantitative aptitude questions (math problems)
        4. Verbal ability questions (grammar, vocabulary)
        5. Analytical questions (pattern recognition, data interpretation)

        Required Format for Each Question:
        {{
            "question": "Clear question text",
            "type": "mcq" or "oneword",
            "options": ["Option1", "Option2", "Option3", "Option4"] (only for mcq),
            "answer": "Correct answer",
            "explanation": "Brief explanation",
            "code_snippet": "Optional code snippet if relevant to question"
        }}

        Rules:
        1. For 'mcq' type: Must include exactly 4 options and the answer must match one option
        2. For 'oneword' type: Answer must be a single word/short phrase
        3. All questions must have all required fields
        4. Return ONLY the JSON array, no additional text or markdown
        5. Difficulty level: {difficulty} - adjust complexity accordingly
        6. Include code snippets only for technical questions where appropriate

        Example Valid Response:
        [
            {{
                "question": "What is the time complexity of binary search?",
                "type": "mcq",
                "options": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
                "answer": "O(log n)",
                "explanation": "Binary search divides the search space in half each time, resulting in logarithmic time complexity.",
                "code_snippet": "def binary_search(arr, target):\\n    low, high = 0, len(arr)-1\\n    while low <= high:\\n        mid = (low + high) // 2\\n        if arr[mid] == target:\\n            return mid\\n        elif arr[mid] < target:\\n            low = mid + 1\\n        else:\\n            high = mid - 1\\n    return -1"
            }},
            {{
                "question": "Which data structure uses LIFO principle?",
                "type": "oneword",
                "answer": "Stack",
                "explanation": "Stack follows Last-In-First-Out (LIFO) principle where the last element added is the first one to be removed.",
                "code_snippet": ""
            }},
            {{
                "question": "If a train travels at 60 km/h, how far will it travel in 2.5 hours?",
                "type": "mcq",
                "options": ["120 km", "150 km", "125 km", "130 km"],
                "answer": "150 km",
                "explanation": "Distance = Speed × Time = 60 km/h × 2.5 h = 150 km",
                "code_snippet": ""
            }},
            {{
                "question": "Choose the correctly spelled word:",
                "type": "mcq",
                "options": ["Occurrence", "Occurence", "Ocurrence", "Occurrance"],
                "answer": "Occurrence",
                "explanation": "The correct spelling is 'Occurrence' with double 'r' and double 'c'.",
                "code_snippet": ""
            }}
        ]
        """

        # Use OpenAI instead of Gemini
        response = openai_generate_content(prompt, max_tokens=4000)
        response_text = response.strip()

        # Clean response text
        for prefix in ['```json', '```']:
            if response_text.startswith(prefix):
                response_text = response_text[len(prefix):].strip()
        if response_text.endswith('```'):
            response_text = response_text[:-3].strip()

        questions = json.loads(response_text)
        
        # Enhanced validation
        valid_questions = []
        validation_errors = []
        
        for i, q in enumerate(questions, 1):
            try:
                # Check required fields
                required = ['question', 'type', 'answer', 'explanation']
                if not all(key in q for key in required):
                    validation_errors.append(f"Q{i}: Missing required fields")
                    continue

                # Validate type
                if q['type'] not in ['mcq', 'oneword']:
                    validation_errors.append(f"Q{i}: Invalid type '{q['type']}'")
                    continue

                # Validate MCQ questions
                if q['type'] == 'mcq':
                    if 'options' not in q:
                        validation_errors.append(f"Q{i}: Missing options for MCQ")
                        continue
                    if len(q['options']) != 4:
                        validation_errors.append(f"Q{i}: Needs exactly 4 options")
                        continue
                    if q['answer'] not in q['options']:
                        validation_errors.append(f"Q{i}: Answer not in options")
                        continue

                # Validate one-word answers
                if q['type'] == 'oneword' and isinstance(q['answer'], list):
                    validation_errors.append(f"Q{i}: One-word answer should be string")
                    continue

                valid_questions.append(q)

            except Exception as e:
                validation_errors.append(f"Q{i}: Validation error - {str(e)}")
                continue

        if len(valid_questions) >= 15:  # Lowered threshold to 15
            session[f'aptitude_questions_{user_id}'] = valid_questions[:30]
            return jsonify({
                "success": True,
                "questions": valid_questions[:30],
                "valid_count": len(valid_questions),
                "total_received": len(questions),
                "warnings": validation_errors[:5]  # Return first 5 errors
            })
        else:
            logger.error(f"Validation failed - Received: {len(questions)} Valid: {len(valid_questions)}")
            logger.error(f"Validation errors: {validation_errors}")
            return jsonify({
                "success": False,
                "error": "Insufficient valid questions",
                "valid_count": len(valid_questions),
                "total_received": len(questions),
                "validation_errors": validation_errors[:10],  # Return first 10 errors
                "sample_question": questions[0] if questions else None
            }), 400

    except json.JSONDecodeError as e:
        logger.error(f"JSON Parse Error: {str(e)}\nResponse: {response_text[:200]}...")
        return jsonify({
            "success": False,
            "error": "Invalid JSON format",
            "response_sample": response_text[:200] + ("..." if len(response_text) > 200 else "")
        }), 400
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": "Question generation failed",
            "details": str(e)
        }), 500

@app.route('/send_aptitude_results_email/<user_id>', methods=['POST'])
def send_aptitude_results_email(user_id):
    try:
        # Get results from session instead of request
        results_data = session.get('aptitude_results', {})

        if not results_data:
            return jsonify({
                'success': False,
                'message': 'No results found to send'
            }), 404

        score = results_data.get('score')
        total = results_data.get('total')
        difficulty = results_data.get('difficulty', 'unknown')
        answers = results_data.get('answers', [])

        # Get user email from database
        conn = sqlite3.connect('reg.db')
        cursor = conn.cursor()
        cursor.execute("SELECT email FROM details WHERE user_id = ?", (user_id,))
        user = cursor.fetchone()
        conn.close()

        if not user:
            return jsonify({
                'success': False,
                'message': 'User not found'
            }), 404

        user_email = user[0]

        # Generate PDF with results
        pdf_filename = generate_aptitude_pdf(user_id, score, total, difficulty, answers)

        # Send email with PDF attachment
        email_sent = send_email_with_attachment(
            user_email,
            pdf_filename,
            user_id,
            score,
            total,
            difficulty
        )

        if email_sent:
            # Clear the session data after successful email sending
            session.pop('aptitude_results', None)
            return jsonify({
                'success': True,
                'message': 'Results sent to your email successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to send email. Please try again.'
            }), 500

    except Exception as e:
        print(f"Error sending aptitude results email: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to send email. Please try again.'
        }), 500

def generate_aptitude_pdf(user_id, score, total, difficulty, answers):
    pdf_filename = os.path.join(CSV_FOLDER, f'{user_id}_aptitude_results.pdf')
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=24, textColor=colors.darkblue, spaceAfter=30)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading1'], fontSize=18, textColor=colors.darkblue, spaceAfter=12)
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=12, spaceAfter=6)
    question_style = ParagraphStyle('Question', parent=styles['Normal'], fontSize=14, textColor=colors.darkgreen, spaceBefore=12)
    answer_style = ParagraphStyle('Answer', parent=styles['Normal'], fontSize=12, leftIndent=20, textColor=colors.black)

    content = []
    content.append(Paragraph("APTITUDE TEST RESULTS", title_style))
    content.append(Spacer(1, 2*inch))
    content.append(Paragraph(f"User ID: {user_id}", normal_style))
    content.append(Paragraph(f"Difficulty: {difficulty}", normal_style))
    content.append(Paragraph(f"Score: {score}/{total} ({round((score/total)*100)}%)", normal_style))
    content.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", normal_style))
    content.append(PageBreak())

    for i, result in enumerate(answers, 1):
        # Add defensive coding to handle different data structures
        user_answer = result.get('user_answer', result.get('userAnswer', 'No answer provided'))
        correct_answer = result.get('correct_answer', result.get('correctAnswer', 'Unknown'))
        is_correct = result.get('is_correct', result.get('isCorrect', False))
        explanation = result.get('explanation', 'No explanation available')
        question = result.get('question', f'Question {i}')

        content.append(Paragraph(f"Question {i}: {question}", question_style))
        content.append(Paragraph(f"Your Answer: {user_answer}", answer_style))
        content.append(Paragraph(f"Correct Answer: {correct_answer}", answer_style))
        content.append(Paragraph(f"Result: {'Correct' if is_correct else 'Incorrect'}", normal_style))
        content.append(Paragraph(f"Explanation: {explanation}", normal_style))
        content.append(Spacer(1, 0.2*inch))

    doc.build(content)
    return pdf_filename

def send_email_with_attachment(to_email, pdf_path, user_id, score, total, difficulty):
    from_email = "sharathsivakumar610@gmail.com"  # Replace with your email
    password = "nrdz lyxy xekq fszx"  # Replace with your app password

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = f"Aptitude Test Results for User ID: {user_id}"

    body = f"""
    Aptitude Test Results for User ID: {user_id}

    Difficulty: {difficulty}
    Score: {score}/{total} ({round((score/total)*100)}%)
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    Please find attached the detailed results of your aptitude test.

    Thank you for using our testing platform.
    """

    msg.attach(MIMEText(body, 'plain'))

    # Attach PDF
    with open(pdf_path, "rb") as file:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(file.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(pdf_path)}")
    msg.attach(part)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

@app.route('/scorecard/<user_id>')
def scorecard(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM details WHERE user_id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()

    total_score = 0  # <-- calculate if needed
    max_possible_score = 0  # <-- calculate if needed

    overall_percentage = session.get("voice_overall_percentage", 0)

    return render_template(
        "scorecard.html",
        name=user["name"],
        total_score=total_score,
        max_possible_score=max_possible_score,
        overall_percentage=overall_percentage,
        user_id=user_id
    )


@app.route('/save_voice_percentage', methods=['POST'])
def save_voice_percentage():
    data = request.get_json()
    percentage = data.get('percentage')

    if percentage is not None:
        session['voice_overall_percentage'] = percentage
        return jsonify({'success': True, 'percentage': percentage})
    return jsonify({'success': False, 'error': 'No percentage provided'}), 400


@app.route('/get_analysis_data/<user_id>')
def get_analysis_data(user_id):
    conn = sqlite3.connect('reg.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # User info
    cursor.execute("SELECT name FROM details WHERE user_id=?", (user_id,))
    user = cursor.fetchone()

    # Attempts & scores
    cursor.execute("""
        SELECT attempt_number, attempt_date, overall_percentage
        FROM interview_attempts
        WHERE user_id=? AND overall_percentage IS NOT NULL
        ORDER BY attempt_number
    """, (user_id,))
    attempts = cursor.fetchall()

    scores = []
    dates = []
    for a in attempts:
        scores.append(a["overall_percentage"])
        dates.append(a["attempt_date"].split()[0] if a["attempt_date"] else f"Attempt {a['attempt_number']}")

    # Skills distribution (sample data - you'll need to implement this properly)
    skills_data = [
        {"skill": "Python", "score": 75},
        {"skill": "SQL", "score": 68},
        {"skill": "Flask", "score": 82},
        {"skill": "JavaScript", "score": 60},
        {"skill": "HTML/CSS", "score": 78}
    ]
    
    skills_labels = [s["skill"] for s in skills_data]
    skills_scores = [s["score"] for s in skills_data]

    # Question type distribution (sample data - you'll need to implement this properly)
    question_types = [
        {"type": "Technical", "score": 75},
        {"type": "Practical", "score": 68},
        {"type": "Aptitude", "score": 82}
    ]

    conn.close()

    total_attempts = len(attempts)
    avg_score = sum(scores)//total_attempts if total_attempts else 0
    best_score = max(scores) if scores else 0
    improvement = (scores[-1] - scores[0]) if len(scores) > 1 else 0

    return jsonify({
        "name": user["name"] if user else user_id,
        "total_attempts": total_attempts,
        "average_score": avg_score,
        "best_score": best_score,
        "improvement": improvement,
        "attempt_dates": dates,
        "scores": scores,
        "skills_labels": skills_labels,
        "skills_data": skills_scores,
        "question_types": question_types
    })
    
@app.route('/download_scorecard/<user_id>')
def download_scorecard(user_id):
    attempt_number = get_user_attempt_number(user_id)
    pdf_filename = os.path.join(CSV_FOLDER, f'{user_id}_scorecard.pdf')

    # Get user details
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM details WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    name = row['name'] if row else "Candidate"

    # Calculate score
    total = 0
    score = 0
    csv_filename = os.path.join(CSV_FOLDER, f'{user_id}_questions_answers_attempt_{attempt_number}.csv')
    import csv
    with open(csv_filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 10
            similarity = calculate_similarity(row['Model Answer'], row['User Answer'])
            if similarity >= 50:
                score += 10

    # Build PDF with same design
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors

    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=28, textColor=colors.darkblue, alignment=1)
    score_style = ParagraphStyle('Score', parent=styles['Normal'], fontSize=22, textColor=colors.green, alignment=1)
    normal_center = ParagraphStyle('Center', parent=styles['Normal'], fontSize=14, alignment=1)

    content = []

    # Add logo
    logo_path = os.path.join(app.static_folder, "logo.png")
    if os.path.exists(logo_path):
        content.append(Image(logo_path, width=200, height=200))
        content.append(Spacer(1, 0.3*inch))

    # Title
    content.append(Paragraph("Score Card", title_style))
    content.append(Spacer(1, 0.3*inch))

    # Name & Score
    content.append(Paragraph(f"Name: {name}", normal_center))
    content.append(Paragraph(f"Final Score: {score}/{total}", score_style))
    content.append(Spacer(1, 0.5*inch))

    # Greeting
    content.append(Paragraph(f"Congratulations {name}! 🎉<br/>Your performance has been evaluated successfully.", normal_center))

    doc.build(content)
    return send_file(pdf_filename, as_attachment=True)

@app.route('/get_user_data/<user_id>')
def get_user_data(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get user details
    cursor.execute("SELECT name FROM details WHERE user_id = ?", (user_id,))
    user = cursor.fetchone()

    # Get attempt count
    cursor.execute("SELECT COUNT(*) FROM interview_attempts WHERE user_id = ?", (user_id,))
    attempt_count = cursor.fetchone()[0]

    # Get average score
    cursor.execute("SELECT AVG(overall_similarity) FROM interview_attempts WHERE user_id = ?", (user_id,))
    avg_score = cursor.fetchone()[0] or 0

    # Check face capture status
    face_capture_completed = user_has_face_capture(user_id)

    conn.close()

    return jsonify({
        'name': user['name'] if user else None,
        'attempt_count': attempt_count,
        'average_score': round(avg_score, 1),
        'face_capture_completed': face_capture_completed,
        'theory_completion': min(100, attempt_count * 20),  # Simplified calculation
        'practical_completion': 0,  # You'll need to implement tracking for this
        'aptitude_completion': 0    # You'll need to implement tracking for this
    })

@app.route('/analysis/<user_id>')
def analysis(user_id):
    return render_template('analysis.html', user_id=user_id)


def get_skills_data(user_id):
    # This function should extract skills data from your database
    # For now, return sample data
    return [
        {"skill": "Python", "count": 3},
        {"skill": "SQL", "count": 2},
        {"skill": "Flask", "count": 1}
    ]

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)

