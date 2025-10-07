import sqlite3
import os

# Create the database in the current folder
db_path = os.path.join(os.getcwd(), "reg.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# SQL to create all tables (including resumes)
cursor.executescript("""
-- Table: details
CREATE TABLE IF NOT EXISTS details (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    user_id TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    upload_file TEXT,
    job_description TEXT
);

-- Table: face_capture_sessions
CREATE TABLE IF NOT EXISTS face_capture_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    capture_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES details(user_id)
);

-- Table: interview_attempts
CREATE TABLE IF NOT EXISTS interview_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    attempt_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    csv_filename TEXT NOT NULL,
    overall_similarity REAL DEFAULT 0,
    feedback TEXT,
    question_count INTEGER DEFAULT 0,
    duration_minutes INTEGER DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES details(user_id)
);

-- ✅ New Table: resumes
CREATE TABLE IF NOT EXISTS resumes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    resume_text TEXT,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES details(user_id)
);
                     
-- Table: skills_performance
CREATE TABLE IF NOT EXISTS skills_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    skill TEXT NOT NULL,
    score INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES details(user_id)
);
                     
-- Table: question_performance
CREATE TABLE IF NOT EXISTS question_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    question_type TEXT NOT NULL,
    score INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES details(user_id)
);

-- Create a table for individual question results if it doesn't exist
CREATE TABLE IF NOT EXISTS question_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    user_answer TEXT,
    similarity_score REAL DEFAULT 0,
    feedback TEXT,
    FOREIGN KEY (attempt_id) REFERENCES interview_attempts(id)
);
""")

# Check if columns already exist and add them if they don't
def add_column_if_not_exists(table_name, column_name, column_type):
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [info[1] for info in cursor.fetchall()]
        
        if column_name not in columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
            print(f"✅ Added column '{column_name}' to table '{table_name}'")
        else:
            print(f"ℹ️ Column '{column_name}' already exists in table '{table_name}'")
    except Exception as e:
        print(f"❌ Error checking/adding column '{column_name}' to table '{table_name}': {e}")

# Add columns to interview_attempts if they don't exist
add_column_if_not_exists("interview_attempts", "overall_similarity", "REAL DEFAULT 0")
add_column_if_not_exists("interview_attempts", "feedback", "TEXT")
add_column_if_not_exists("interview_attempts", "question_count", "INTEGER DEFAULT 0")
add_column_if_not_exists("interview_attempts", "duration_minutes", "INTEGER DEFAULT 0")
add_column_if_not_exists("interview_attempts", "interview_type", "TEXT DEFAULT 'text'")
add_column_if_not_exists("details", "phone", "TEXT")

# NEW: Add faceset_token to face_capture_sessions if it doesn't exist
add_column_if_not_exists("face_capture_sessions", "faceset_token", "TEXT")

# Optional: Insert a sample resume for testing voice interview
try:
    cursor.execute("""
    INSERT OR IGNORE INTO resumes (user_id, resume_text)
    VALUES (?, ?)
    """, ("101", "Experienced Python developer with Flask, SQLite, and AI integrations."))
except:
    print("ℹ️ Sample resume already exists or couldn't be inserted")

conn.commit()
conn.close()

print("✅ Database setup completed successfully!")