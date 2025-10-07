SET FOREIGN_KEY_CHECKS=0;

DROP TABLE IF EXISTS `details`;
CREATE TABLE details (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    upload_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    upload_file VARCHAR(255),
    job_description VARCHAR(255)
);

DROP TABLE IF EXISTS `face_capture_sessions`;
CREATE TABLE face_capture_sessions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    capture_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES details(user_id)
);

DROP TABLE IF EXISTS `interview_attempts`;
CREATE TABLE interview_attempts (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id VARCHAR(255) NOT NULL,
    attempt_number INTEGER NOT NULL,
    attempt_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    csv_filename VARCHAR(255) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES details(user_id)
);

DROP TABLE IF EXISTS `resumes`;
CREATE TABLE resumes (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id VARCHAR(255) NOT NULL,
    resume_text VARCHAR(255),
    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES details(user_id)
);
INSERT INTO `resumes` (id, user_id, resume_text, uploaded_at) VALUES ('1', '101', 'Experienced Python developer with Flask, SQLite, and AI integrations.', '2025-07-07 06:40:32');
SET FOREIGN_KEY_CHECKS=1;
