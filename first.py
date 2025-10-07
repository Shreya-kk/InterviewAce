import sqlite3
import os
import re
import random  # Added this import
import mysql.connector
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import PyPDF2
import docx
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
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
from google.generativeai import GenerativeModel
import subprocess
import signal
from functools import wraps
import traceback
from flask import request, send_file
import requests
from io import BytesIO

app = Flask(__name__)
app.secret_key = '7340d01377d428f7b9a5608a3a8b46d3'

from flask_session import Session
from datetime import datetime, timedelta

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

# ELEVENLABS_API_KEY = "sk_1023bd0bdb0b18345fcfa94e04d372840c84cfc1e2816c99"
# VOICE_ID = "iWq9tCtTw1SYmVpvsRjY"

ELEVENLABS_API_KEY = "sk_1023bd0bdb0b18345fcfa94e04d372840c84cfc1e2816c99"
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# genai.configure(api_key="AIzaSyDVE9ObZlAkrrelgEq9zfEUfM9p6TSWsh4")
# genai.configure(api_key="AIzaSyDlbrdvDOnhx3T2ezo8sAPaHUwoOVpcpBU")
# genai.configure(api_key="AIzaSyBgm_XijhBtpIGmepmXCBRJ3JzTPGg80no")
# genai.configure(api_key="AIzaSyC_WXI8fi4EqedWNoaMdbMYocPW16d2ybw")
# genai.configure(api_key="AIzaSyDkGmyQjFede_KlTl1l3vgcJglMbNNuvM8")
genai.configure(api_key="AIzaSyCI3mwoIsjHIm94rJ4FBuN8bt4-QaIkrJk")
model = genai.GenerativeModel("gemini-1.5-flash")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Directory to store CSV files
CSV_FOLDER = r"E:/Project25-26/Interview/project5_interview/output_csv"
os.makedirs(CSV_FOLDER, exist_ok=True)

# Directory to store captured images
IMAGE_FOLDER = r"E:/Project25-26/Interview/project5_interview/images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Face recognition setup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

model = GenerativeModel('gemini-1.5-flash')

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

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((SyntaxError, ValueError))
)
def extract_technical_skills_with_gemini(text):
    prompt = f"""
    The following text is extracted from a resume or CV. Please identify and list only the technical skills mentioned in this text. Focus on hard skills related to technology, programming languages, software, tools, and specific technical knowledge. Provide the result as a comma-separated list of skills. 

    Text:
    {text}

    Technical Skills:
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        if isinstance(response, GenerateContentResponse) and response.text:
            skills_list = [skill.strip() for skill in response.text.split(',') if skill.strip()]
            logger.info(f"Successfully extracted {len(skills_list)} technical skills")
            return skills_list
        else:
            logger.warning("Empty or invalid response from Gemini API")
            raise ValueError("Empty or invalid response from Gemini API")
    except Exception as e:
        logger.error(f"Failed to extract technical skills: {str(e)}")
        raise

    return []

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

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((SyntaxError, ValueError))
)
def generate_questions_and_answers_from_skills(skills):
    prompt = f"""Given the following technical skills: {', '.join(skills)},
    generate 5 interview questions that would be appropriate for assessing 
    a candidate's proficiency in these skills. For each question, also provide 
    an answer in 40-50 words. The questions should be challenging but fair, 
    and should help evaluate both theoretical knowledge and practical application 
    of these skills. Format the output as follows:

    Q1: [Question 1]
    A1: [Answer 1 (40-50 words)]

    Q2: [Question 2]
    A2: [Answer 2 (40-50 words)]

    ... and so on for 5 questions and answers."""

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        if isinstance(response, GenerateContentResponse) and response.text:
            qa_pairs = []
            lines = response.text.split('\n')
            current_question = ""
            current_answer = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith('Q'):
                    if current_question and current_answer:
                        qa_pairs.append((current_question, current_answer))
                    current_question = line[line.find(':') + 1:].strip()
                elif line.startswith('A'):
                    current_answer = line[line.find(':') + 1:].strip()
            
            if current_question and current_answer:
                qa_pairs.append((current_question, current_answer))
            
            if len(qa_pairs) == 5:
                logger.info(f"Successfully generated 5 question-answer pairs")
                return qa_pairs
            else:
                logger.warning(f"Generated {len(qa_pairs)} question-answer pairs instead of 5")
                raise ValueError(f"Generated {len(qa_pairs)} question-answer pairs instead of 5")
        else:
            logger.warning("Empty or invalid response from Gemini API")
            raise ValueError("Empty or invalid response from Gemini API")
    except Exception as e:
        logger.error(f"Failed to generate questions and answers: {str(e)}")
        raise

    return []

def save_questions_and_answers_to_csv(qa_pairs, user_id, attempt_number, csv_filename):
    filepath = os.path.join(CSV_FOLDER, csv_filename)
    with open(filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Question Number', 'Question', 'Model Answer', 'User Answer'])  # CSV header
        for i, (question, answer) in enumerate(qa_pairs, 1):
            writer.writerow([f"Question {i}", question, answer, ''])
    return filepath

def user_has_face_capture(user_id):
    conn = sqlite3.connect('reg.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = """
        SELECT session_id 
        FROM face_capture_sessions 
        WHERE user_id = ? 
        ORDER BY capture_time DESC 
        LIMIT 1
    """
    cursor.execute(query, (user_id,))
    result = cursor.fetchone()

    cursor.close()
    conn.close()

    if result and result[0]:  # result[0] is session_id
        yml_file = f"{result[0]}_model.yml"
        return os.path.exists(os.path.join(IMAGE_FOLDER, yml_file))

    return False

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
SUPPORTED_LANGUAGES = ['python', 'cpp', 'java']

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
        user_id = request.form['user_id']
        password = request.form['password']

        conn = sqlite3.connect('reg.db')  # or use get_db_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Use ? placeholders for SQLite
        query = "SELECT * FROM details WHERE user_id = ? AND password = ?"
        cursor.execute(query, (user_id, password))
        user = cursor.fetchone()

        conn.close()

        if user:
            return redirect(url_for('both', user_id=user_id))
        else:
            flash('Invalid login credentials')
            return redirect(url_for('login'))

    return render_template('sec.html')

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
    return render_template('success.html', user_id=user_id)

def get_db_connection():
    conn = sqlite3.connect('reg.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name'].strip()
        email = request.form['email'].strip()
        phone = request.form['phone'].strip()
        password = request.form['password'].strip()

        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if email or phone already exists
        check_query = "SELECT * FROM details WHERE email = ? OR phone = ?"
        cursor.execute(check_query, (email, phone))
        existing_user = cursor.fetchone()

        if existing_user:
            if existing_user['email'] == email:
                flash('Email is already registered.', 'error')
            elif existing_user['phone'] == phone:
                flash('Phone number is already registered.', 'error')
            conn.close()
            return redirect(url_for('register'))

        # Generate new user_id like LP00001
        cursor.execute("SELECT COUNT(*) as count FROM details")
        row = cursor.fetchone()
        count = row['count'] + 1
        user_id = f"LP{count:05d}"

        try:
            insert_query = '''
                INSERT INTO details (name, email, phone, user_id, password, upload_time, upload_file)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, '')
            '''
            cursor.execute(insert_query, (name, email, phone, user_id, password))
            conn.commit()

            flash('Registration successful!', 'success')
            return redirect(url_for('registration_success', user_id=user_id))
        except Exception as e:
            conn.rollback()
            flash(f'An error occurred: {e}', 'error')
        finally:
            conn.close()

    return render_template('third.html')


@app.route('/four/<user_id>', methods=['GET', 'POST'])
def four(user_id):
    conn = sqlite3.connect('reg.db')  # Connect to your SQLite database
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
            error_message = "Face capture must be completed before uploading a resume."
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
                            # ✅ Save the extracted resume into the resumes table
                            cursor.execute("INSERT INTO resumes (user_id, resume_text) VALUES (?, ?)", (user_id, extracted_text))

                            conn.commit()

                            # Continue with your existing logic
                            technical_skills = extract_technical_skills_with_gemini(extracted_text)

                            if technical_skills:
                                qa_pairs = generate_questions_and_answers_from_skills(technical_skills)
                                if qa_pairs:
                                    new_attempt, csv_filename = increment_user_attempt(user_id)
                                    save_questions_and_answers_to_csv(qa_pairs, user_id, new_attempt, csv_filename)
                                    questions_generated = True
                                else:
                                    error_message = "Failed to generate questions. Please try again."
                            else:
                                error_message = "No technical skills found in the uploaded file."
                        except Exception as e:
                            error_message = f"An error occurred: {str(e)}. Please try again."
                    else:
                        error_message = "Unable to extract text from the uploaded file."

    cursor.close()
    return render_template('four.html', 
                           job_description=job_description, 
                           technical_skills=technical_skills, 
                           questions_generated=questions_generated, 
                           user_id=user_id,
                           face_capture_completed=face_capture_completed,
                           error_message=error_message)

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

        cursor.execute("""
            SELECT attempt_number, attempt_date, 
                   COALESCE(overall_similarity, 0) AS overall_similarity, 
                   COALESCE(feedback, '') AS feedback
            FROM interview_attempts 
            WHERE user_id = ? 
            ORDER BY attempt_number ASC
        """, (user_id,))
        
        attempts = cursor.fetchall()
        conn.close()

        return jsonify([
            {
                "attemptNumber": row["attempt_number"],
                "date": row["attempt_date"],
                "overallSimilarity": row["overall_similarity"],
                "feedback": row["feedback"]
            }
            for row in attempts
        ])

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# @app.route('/get_attempt_feedback/<user_id>/<int:attempt_number>')
# def get_attempt_feedback(user_id, attempt_number):
#     conn = sqlite3.connect('reg.db')
#     conn.row_factory = sqlite3.Row
#     cursor = conn.cursor()
    
#     cursor.execute(
#         "SELECT * FROM interview_attempts_details WHERE user_id = ? AND attempt_number = ?",
#         (user_id, attempt_number)
#     )
#     attempt_data = cursor.fetchone()
#     conn.close()
    
#     if attempt_data:
#         return jsonify({
#             'feedback': attempt_data['feedback'],
#             'overallSimilarity': attempt_data['overall_similarity'],
#             'date': attempt_data['attempt_date']
#         })
#     else:
#         return jsonify({'error': 'Attempt not found'}), 404

def save_attempt_to_database(user_id, attempt_number, overall_similarity, feedback):
    conn = sqlite3.connect('reg.db')
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id FROM interview_attempts WHERE user_id = ? AND attempt_number = ?",
        (user_id, attempt_number)
    )
    existing = cursor.fetchone()

    if existing:
        cursor.execute(
            "UPDATE interview_attempts SET overall_similarity = ?, feedback = ? WHERE user_id = ? AND attempt_number = ?",
            (overall_similarity, feedback, user_id, attempt_number)
        )
    else:
        cursor.execute(
            "INSERT INTO interview_attempts (user_id, attempt_number, attempt_date, overall_similarity, feedback, csv_filename) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                user_id,
                attempt_number,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                overall_similarity,
                feedback,
                f"{user_id}_questions_answers_attempt_{attempt_number}.csv"
            )
        )

    conn.commit()
    conn.close()


def generate_overall_feedback(questions_answers):
    prompt = f"""
    Analyze the following set of interview questions and answers:

    {questions_answers}

    Provide a concise overall feedback (about 100 words) on the candidate's performance, highlighting strengths and areas for improvement. Focus on:
    1. Technical knowledge
    2. Communication skills
    3. Problem-solving approach
    4. Areas that need more attention

    Format the feedback as a paragraph without bullet points.
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        if isinstance(response, GenerateContentResponse) and response.text:
            return response.text.strip()
        else:
            return "Unable to generate feedback at this time."
    except Exception as e:
        print(f"Error generating overall feedback: {str(e)}")
        return "Error generating feedback. Please try again later."

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
    
    session_folder = os.path.join(IMAGE_FOLDER, session_id)
    logger.info(f"Processing images in folder: {session_folder}")
    DATABASE = 'reg.db' 


    # Train the face recognition model
    faces = []
    labels = []
    for filename in os.listdir(session_folder):
        if filename.startswith('face_'):
            img_path = os.path.join(session_folder, filename)
            face = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if face is not None:
                faces.append(face)
                labels.append(1)  # Assuming all faces belong to the same person
            else:
                logger.warning(f"Failed to read image: {img_path}")
    
    if faces:
        logger.info(f"Training model with {len(faces)} face images")
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(faces, np.array(labels))

            # Save trained model
            yml_path = os.path.join(IMAGE_FOLDER, f'{session_id}_model.yml')
            recognizer.save(yml_path)
            logger.info(f"Model saved to: {yml_path}")

            # Connect to SQLite and insert session record
            db = sqlite3.connect(DATABASE)
            cursor = db.cursor()

            query = "INSERT INTO face_capture_sessions (user_id, session_id, capture_time) VALUES (?, ?, ?)"
            current_time = datetime.now()
            cursor.execute(query, (user_id, session_id, current_time))
            db.commit()
            cursor.close()
            db.close()

            return jsonify({
                "success": True,
                "message": "Face recognition model trained successfully",
                "redirect": url_for('four', user_id=user_id)
            }), 200

        except Exception as e:
            logger.error(f"Error during training or saving model: {e}")
            return jsonify({
                "success": False,
                "message": "Failed to train and save face recognition model."
            }), 500

    else:
        logger.warning("No valid face images found for training")
        return jsonify({"error": "No valid face images found for training"}), 400


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
        
        # Process the uploaded image for face detection
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_filename = f'face_{filename}'
            face_filepath = os.path.join(session_folder, face_filename)
            cv2.imwrite(face_filepath, face)
            logger.info(f"Face image saved: {face_filepath}")
        
        return jsonify({
            "success": True,
            "filename": filename,
            "session_id": session_id,
            "faces_detected": len(faces)
        }), 200
    
@app.route('/both/<user_id>')
def both(user_id):
    return render_template('both.html', user_id=user_id)

@app.route('/different/<user_id>')
def different(user_id):
    return render_template('different.html', user_id=user_id)


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
            return generate_question()
        elif action == 'check_code':
            return check_code()
        elif action == 'submit_code':
            return submit_code()
        elif action == 'run_code':
            return run_code()

    return jsonify({'error': 'Invalid request'}), 400

@with_error_handling
def run_code():
    user_code = request.json.get('code')
    user_input = request.json.get('input')
    language = request.json.get('option')

    if not user_code.strip():
        return jsonify({
            'output': "No code submitted. Please enter some code before running.",
            'has_error': True
        })

    # Analyze code and predict output using Gemini API
    prompt = f"""
    Given this {language} code and input, tell me what would be the output.
    If there are any syntax errors or runtime errors, respond with "CHECK THE CODE! THERE ARE ERRORS IN THE CODE"
    Otherwise, just show the exact output the code would produce.

    CODE:
    {user_code}

    INPUT:
    {user_input}

    Respond only with the output or error message, nothing else.
    """

    try:
        response = model.generate_content(prompt)
        output = response.text.strip()
        
        has_error = "CHECK THE CODE" in output
        
        return jsonify({
            'output': output,
            'has_error': has_error
        })
        
    except Exception as e:
        return jsonify({
            'output': "An error occurred while analyzing the code",
            'has_error': True
        })

@with_error_handling
def generate_question():
    option = request.json.get('option')
    question_number = request.json.get('questionNumber')
    
    language = option if option in SUPPORTED_LANGUAGES else 'python'
    difficulty = 'intermediate'  # Default difficulty

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
    
    response = model.generate_content(prompt)
    question = response.text
    session[f'question_{question_number}'] = question
    
    return jsonify({'question': question})

@with_error_handling
def check_code():
    user_code = request.json.get('code')
    option = request.json.get('option')
    question_number = request.json.get('questionNumber')
    question = session.get(f'question_{question_number}')

    if not user_code.strip():
        return jsonify({'output': "No code submitted for checking."})

    language = option if option in SUPPORTED_LANGUAGES else 'python'

    # Split code into lines for line-by-line analysis
    code_lines = user_code.split('\n')
    line_numbers = [i + 1 for i in range(len(code_lines))]

    prompt = """
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
    {'\n'.join(f"Line {num}: {line}" for num, line in zip(line_numbers, code_lines))}

    After the line-by-line analysis, provide:
    Overall Assessment:
    - Total number of errors found
    - General code quality
    - Suggestions for improvement 
   """

    response = model.generate_content(prompt)
    return jsonify({'output': response.text})

@with_error_handling
def submit_code():
    user_code = request.json.get('code')
    option = request.json.get('option')
    question_number = request.json.get('questionNumber')
    question = session.get(f'question_{question_number}')
    
    if not user_code.strip():
        return jsonify({'evaluation': 'Score: 0/100\nNo code submitted.'})
    
    language = option if option in SUPPORTED_LANGUAGES else 'python'
    
    prompt = f"""
    Evaluate this {language} solution:
    Question: {question}
    Code: {user_code}

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
    
    response = model.generate_content(prompt)
    return jsonify({'evaluation': response.text})

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


'''
@app.route('/practical_results/<user_id>', methods=['GET', 'POST'])
@with_error_handling  # Use your existing error handling decorator
def practical_results(user_id):
    if request.method == 'POST':
        # Get the results data from request
        results = request.get_json()
        # Store in session
        session[f'practical_results_{user_id}'] = {
            'language': results.get('language', ''),
            'responses': results.get('responses', [])
        }
        return jsonify({'status': 'success'})
    
    # GET request handling
    results = session.get(f'practical_results_{user_id}', {
        'language': '',
        'responses': []
    })
    
    # Return HTML or JSON based on Accept header
    if request.headers.get('Accept') == 'application/json':
        return jsonify(results)
    
    return render_template(
        'practical_results.html',
        language=results.get('language', ''),
        responses=results.get('responses', [])
    )'''

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


@app.route('/recognize_face/<user_id>', methods=['POST'])
def recognize_face(user_id):
    image_data = request.json['image']
    image_data = image_data.split(',')[1]  # Remove the "data:image/jpeg;base64," part
    
    # Decode the base64 image
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # Load the trained model for this user (consider caching this for performance)
    session_id = get_user_session_id(user_id)
    if not session_id:
        return jsonify({"error": "No trained model found for this user"}), 400
    
    yml_path = os.path.join(IMAGE_FOLDER, f'{session_id}_model.yml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Check if the model file exists
    if not os.path.exists(yml_path):
        return jsonify({"error": "Face recognition model file not found"}), 400
    
    try:
        recognizer.read(yml_path)
    except Exception as e:
        logger.error(f"Error loading face recognition model: {str(e)}")
        return jsonify({"error": "Failed to load face recognition model"}), 500
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    result = {"faces": []}
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        try:
            id_, confidence = recognizer.predict(roi_gray)
            
            result["faces"].append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "isUser": confidence < 70,  # Lower confidence is better
                "message": "YES IT'S THE USER" if confidence < 70 else "NO IT'S NOT THE USER",
                "confidence": confidence
            })
        except Exception as e:
            logger.error(f"Error predicting face: {str(e)}")
            result["faces"].append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "isUser": False,
                "message": "Error during recognition",
                "confidence": 100
            })
    
    return jsonify(result)

def get_user_session_id(user_id):
    conn = sqlite3.connect('reg.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = """
        SELECT session_id 
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
            
            # Generate feedback using Gemini API
            feedback = generate_feedback(row['Question'], model_answer, user_answer)
            
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
    
    # Generate PDF for the current attempt
    current_attempt_pdf = generate_pdf(user_id, results_with_feedback)
    
    # Generate PDF for all attempts
    all_attempts_pdf = generate_all_attempts_pdf(user_id)
    
    # Fetch user's email from the database
    cursor = db.cursor(dictionary=True)
    query = "SELECT email FROM details WHERE user_id = %s"
    cursor.execute(query, (user_id,))
    user = cursor.fetchone()
    cursor.close()
    
    if user and user['email']:
        # Send email with both PDF attachments
        email_sent = send_email_with_attachments(user['email'], current_attempt_pdf, all_attempts_pdf, user_id)
        if email_sent:
            return jsonify({"results": results, "message": "Results and feedback for current and all attempts sent to your email."})
        else:
            return jsonify({"results": results, "message": "Failed to send email. Please check with the administrator."})
    else:
        return jsonify({"results": results, "message": "User email not found. Results not sent."})
    
def extract_technical_keywords(resume_text):
    """Extract technical keywords from resume to inform question generation"""
    prompt = f"""
    Extract the top 10 most important technical skills, technologies, and programming languages 
    from this resume. Return them as a comma-separated list:
    
    {resume_text}
    
    Technical keywords:
    """
    
    try:
        result = model.generate_content(prompt)
        keywords = [kw.strip() for kw in result.text.split(',')]
        return keywords[:10]  # Return top 10
    except:
        return ["programming", "development", "technical", "software", "coding"]


def generate_feedback(question, model_answer, user_answer):
    prompt = f"""
    Question: {question}
    Model Answer: {model_answer}
    User Answer: {user_answer}
    
    Provide 5 short, clear feedback points on how the user could improve their answer. Each point should be concise and actionable.
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        if isinstance(response, GenerateContentResponse) and response.text:
            # Split the response into lines and take the first 5
            feedback_points = response.text.strip().split('\n')[:5]
            # Join the points into a single string
            return "\n".join(feedback_points)
        else:
            return "Unable to generate feedback at this time."
    except Exception as e:
        print(f"Error generating feedback: {str(e)}")
        return "Error generating feedback. Please try again later."

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

# @app.route('/voice_interview/<user_id>')
# def voice_interview(user_id):
#     try:
#         # Clear previous session data
#         session.clear()
#         session['user_id'] = user_id
#         session['interview_started'] = True
#         session['question_count'] = 0
#         session['answers'] = []
#         session['asked_questions'] = []
#         session['all_questions'] = []  # Initialize all_questions

#         conn = sqlite3.connect('reg.db')
#         conn.row_factory = sqlite3.Row
#         cur = conn.cursor()
#         cur.execute("""
#             SELECT r.resume_text, d.name 
#             FROM resumes r
#             JOIN details d ON r.user_id = d.user_id
#             WHERE r.user_id = ?
#             ORDER BY r.id DESC LIMIT 1
#         """, (user_id,))
#         data = cur.fetchone()
#         conn.close()

#         session['resume_text'] = data['resume_text'] if data else ""
#         session['user_name'] = data['name'] if data else "Candidate"
        
#         # Extract resume keywords for later use
#         if session['resume_text']:
#             session['resume_keywords'] = extract_technical_keywords(session['resume_text'])
#         else:
#             session['resume_keywords'] = []
        
#         return render_template("voice_interview.html", user_id=user_id)
#     except Exception as e:
#         logger.error(f"Error in voice_interview: {str(e)}")
#         flash("An error occurred while starting the interview")
#         return redirect(url_for('four', user_id=user_id))

# # Update the initial question generation - ALWAYS start with "Tell me about yourself"
# @app.route('/start_interview', methods=['POST'])
# def start_interview():
#     user_name = session.get('user_name', 'Candidate')
    
#     # FIXED: Always start with "Tell me about yourself" question
#     initial_question = f"Tell me about yourself, {user_name}."
    
#     # Initialize session tracking
#     session['answers'] = []
#     session['all_questions'] = [initial_question]  # Track all questions
#     session['asked_questions'] = [initial_question]  # Track asked questions
#     session['current'] = 0
#     session['last_question'] = initial_question
#     session['question_count'] = 1
#     session.modified = True
    
#     return jsonify({'question': initial_question})

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

def generate_next_question(prev_question, prev_answer):
    """Generate the next question based on previous question and answer"""
    asked_questions = session.get('asked_questions', [])
    user_name = session.get('user_name', 'Candidate')
    
    # If no answer was provided, skip to next question without follow-up
    if prev_answer == "(No answer provided)":
        if session.get('current_round') == 'technical':
            return generate_technical_followup(prev_question, "", asked_questions, user_name)
        else:
            return generate_hr_question(prev_question, "", asked_questions, user_name)
    
    # If this is the first question (tell me about yourself)
    if "tell me about yourself" in prev_question.lower():
        # Generate first technical question based on resume
        return generate_first_technical_question(prev_answer, session.get('resume_text', ''), user_name)
    
    # For subsequent questions, alternate between technical and HR questions
    # After 4 technical questions, switch to HR round
    if session.get('current_round') == 'technical' and session.get('question_count', 0) >= 5:
        session['current_round'] = 'hr'
    
    if session.get('current_round') == 'technical':
        return generate_technical_followup(prev_question, prev_answer, asked_questions, user_name)
    else:
        return generate_hr_question(prev_question, prev_answer, asked_questions, user_name)

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
        
        result = model.generate_content(prompt)
        question = result.text.strip()
        
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
    # If no answer was provided, generate a new question without following up
    if not prev_answer or prev_answer == "(No answer provided)":
        # Use a different approach to generate a fresh question
        resume_text = session.get('resume_text', '')
        resume_keywords = session.get('resume_keywords', [])
        used_keywords = session.get('used_keywords', [])
        
        # Get available keywords that haven't been used
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
    
    try:
        # Get available keywords that haven't been used
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
            
            PREVIOUS QUESTION: {prev_question}
            CANDIDATE'S ANSWER: {prev_answer}
            
            ALREADY ASKED QUESTIONS (avoid these):
            {', '.join(asked_questions[-5:]) if asked_questions else 'None'}
            
            Generate a technical question that:
            1. Is specifically about {focus_keyword} or related technologies
            2. Builds naturally on their previous answer when relevant
            3. Is open-ended and encourages detailed response
            4. Keeps under 20 words
            5. Address them as {user_name}
            6. Ends with a question mark
            7. Is different from previously asked questions
            
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
        
        result = model.generate_content(prompt)
        question = result.text.strip()
        
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
        
        response = model.generate_content(prompt)
        question = response.text.strip()
        
        if not question.endswith('?'):
            question += '?'
            
        return question
        
    except Exception as e:
        logger.error(f"Error generating HR question: {str(e)}")
        return get_fallback_hr_question(asked_questions, user_name)

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
        
        # Also add to asked_questions for tracking
        if current_question and current_question not in session.get('asked_questions', []):
            if 'asked_questions' not in session:
                session['asked_questions'] = []
            session['asked_questions'].append(current_question)
        
        # Generate next question based on conversation flow
        next_question = generate_next_question(current_question, answer)
        
        # Track the new question
        session['all_questions'].append(next_question)
        session['question_count'] = len(session['all_questions'])
        session.modified = True
        
        return jsonify({
            'question': next_question,
            'question_number': session['question_count'],
            'status': 'success'
        })

    except Exception as e:
        logger.error(f"Error in submit_voice_answer: {str(e)}")
        user_name = session.get('user_name', 'Candidate')
        asked_questions = session.get('asked_questions', [])
        
        # Determine which type of fallback question to use based on current round
        if session.get('current_round') == 'hr':
            fallback_question = get_fallback_hr_question(asked_questions, user_name)
        else:
            fallback_question = get_fallback_technical_question(asked_questions, user_name)
            
        return jsonify({
            'question': fallback_question,
            'status': 'error',
            'message': str(e)
        }), 500

# @app.route('/submit_voice_answer', methods=['POST'])
# def submit_voice_answer():
#     try:
#         data = request.get_json()
#         answer = data.get('answer', '').strip()
#         is_hr_round = data.get('isHrRound', False)
        
#         # Ensure session tracking exists
#         if 'all_questions' not in session:
#             session['all_questions'] = []
#             session['answers'] = []
#             session['question_count'] = 0
        
#         # Get the current question (last one asked)
#         current_question = session['all_questions'][-1] if session['all_questions'] else ""
        
#         # Store answer with question reference
#         session['answers'].append({
#             'question': current_question,
#             'answer': answer
#         })
        
#         # Also add to asked_questions for tracking
#         if current_question and current_question not in session.get('asked_questions', []):
#             if 'asked_questions' not in session:
#                 session['asked_questions'] = []
#             session['asked_questions'].append(current_question)
        
#         # Generate next question based on round type
#         if is_hr_round:
#             next_question = generate_hr_question(
#                 current_question,
#                 answer,
#                 session['all_questions']
#             )
#         else:
#             next_question = generate_technical_question_from_resume(
#                 current_question,
#                 answer,
#                 session['all_questions']
#             )
        
#         # Track the new question
#         session['all_questions'].append(next_question)
#         session['question_count'] += 1
#         session.modified = True
        
#         return jsonify({
#             'question': next_question,
#             'question_number': session['question_count'],
#             'status': 'success'
#         })

#     except Exception as e:
#         logger.error(f"Error in submit_voice_answer: {str(e)}")
#         fallback_question = get_fallback_hr_question(session.get('all_questions', []), session.get('user_name', 'Candidate')) if is_hr_round else get_fallback_technical_question(session.get('all_questions', []), session.get('user_name', 'Candidate'))
#         return jsonify({
#             'question': fallback_question,
#             'status': 'error',
#             'message': str(e)
#         }), 500

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
        
        # Evaluate all questions (answered and unanswered)
        results = []
        total_score = 0
        
        for i, question in enumerate(all_questions):
            answer = answered_dict.get(question, '(No answer provided)')
            
            # Generate score and feedback
            if answer != '(No answer provided)':
                score = min(10, len(answer.split()) // 3)  # Simple scoring
                feedback = generate_feedback(question, "Model answer", answer)
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
        
        # Store evaluated answers in session temporarily
        session['evaluated_answers'] = results
        session['total_score'] = total_score
        
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
                         'user_name', 'interview_started', 'asked_questions']
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
        user_id = session.get('user_id', '')
        
        if not evaluated_answers:
            flash("No interview results found. Please complete an interview first.")
            return redirect(url_for('four', user_id=user_id))
        
        # Clear the remaining session data after getting what we need
        results_keys = ['evaluated_answers', 'total_score']
        for key in results_keys:
            session.pop(key, None)
        session.modified = True
        
        return render_template(
            "voice_results.html",
            answers=evaluated_answers,
            total_score=total_score
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
        'evaluated_answers', 'total_score'
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

# Add this route to save results
@app.route('/save_aptitude_results/<user_id>', methods=['POST'])
def save_aptitude_results(user_id):
    try:
        data = request.get_json()
        score = data.get('score')
        total = data.get('total')
        difficulty = data.get('difficulty', 'unknown')
        answers = data.get('answers', [])
        
        # Store results in session instead of database
        session['aptitude_results'] = {
            'user_id': user_id,
            'score': score,
            'total': total,
            'difficulty': difficulty,
            'answers': answers,
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

        prompt = f"""Generate exactly 30 {difficulty} level technical aptitude test questions for computer science students with these specifications:
        
        Required Format for Each Question:
        {{
            "question": "Clear technical question text",
            "type": "mcq" or "oneword",
            "options": ["Option1", "Option2", "Option3", "Option4"] (only for mcq),
            "answer": "Correct answer",
            "explanation": "Brief technical explanation",
            "code_snippet": "Optional code snippet if relevant to question"
        }}

        Rules:
        1. Focus on programming concepts, algorithms, data structures, and technical problem-solving
        2. For 'mcq' type: Must include exactly 4 options and the answer must match one option
        3. For 'oneword' type: Answer must be a single word/short phrase
        4. All questions must have all required fields
        5. Return ONLY the JSON array, no additional text or markdown
        6. Difficulty level: {difficulty} - adjust technical complexity accordingly
        7. Include code snippets where appropriate for programming questions

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
            }}
        ]
        """

        response = model.generate_content(prompt)
        response_text = response.text.strip()

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
    # Fetch latest attempt data
    attempt_number = get_user_attempt_number(user_id)
    csv_filename = os.path.join(CSV_FOLDER, f'{user_id}_questions_answers_attempt_{attempt_number}.csv')
    
    total_score = 0
    total_questions = 0
    name = "Candidate"
    
    # Get candidate name
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM details WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    if row:
        name = row['name']
    conn.close()

    # Read CSV and calculate actual scores
    import csv
    with open(csv_filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_questions += 1
            similarity = calculate_similarity(row['Model Answer'], row['User Answer'])
            total_score += similarity
    
    # Calculate average score
    average_score = total_score / total_questions if total_questions > 0 else 0
    
    return render_template("scorecard.html", name=name, score=round(average_score), total=100, user_id=user_id)

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
        SELECT attempt_number, attempt_date, csv_filename
        FROM interview_attempts
        WHERE user_id=?
        ORDER BY attempt_number
    """, (user_id,))
    attempts = cursor.fetchall()

    scores, dates = [], []
    for a in attempts:
        try:
            df = pd.read_csv(a["csv_filename"])
            score = int(df["overallSimilarity"].mean())
            scores.append(score)
            dates.append(a["attempt_date"])
        except Exception as e:
            print(f"Error reading {a['csv_filename']}: {e}")
            scores.append(0)
            dates.append(a["attempt_date"])

    # ✅ Skills distribution
    cursor.execute("""
        SELECT skill, AVG(score) as avg_score
        FROM skills_performance
        WHERE user_id=?
        GROUP BY skill
    """, (user_id,))
    skills = cursor.fetchall()
    skills_labels = [s["skill"] for s in skills]
    skills_data = [int(s["avg_score"]) for s in skills]

    # Question type distribution
    cursor.execute("""
        SELECT question_type, AVG(score) as avg_score
        FROM question_performance
        WHERE user_id=?
        GROUP BY question_type
    """, (user_id,))
    qtypes = cursor.fetchall()

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
        # 👇 keys your analysis.html expects
        "skills_labels": skills_labels,
        "skills_data": skills_data,
        "question_types": [{"type": q["question_type"], "score": int(q["avg_score"])} for q in qtypes]
    })


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