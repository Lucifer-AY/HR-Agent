import streamlit as st
import requests
import json
import os
import sys
import asyncio
import websockets
import logging
import threading
import openai
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
import re
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta

project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

load_dotenv()

logging.basicConfig(filename=os.path.join(project_root, 'data', 'logs', 'frontend.log'), level=logging.INFO)
logging.info(f"Loaded SMTP_EMAIL: {os.getenv('SMTP_EMAIL')}")
logging.info(f"Loaded SMTP_PASSWORD: {os.getenv('SMTP_PASSWORD')[:4] if os.getenv('SMTP_PASSWORD') else None}... (masked)")

try:
    from src.parsing import parse_resume
    from src.interview import conduct_live_interview, generate_feedback
except ModuleNotFoundError as e:
    st.error(f"Failed to import from src: {e}")
    raise

API_URL = "http://localhost:8000"
WEBSOCKET_URL = "ws://localhost:8000/interview"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in .env file")
    raise ValueError("OPENAI_API_KEY not set")
if not SMTP_EMAIL or not SMTP_PASSWORD:
    st.error(f"SMTP_EMAIL or SMTP_PASSWORD not found in .env file: SMTP_EMAIL={SMTP_EMAIL}, SMTP_PASSWORD={'*' * len(SMTP_PASSWORD) if SMTP_PASSWORD else None}")
    raise ValueError("SMTP credentials not set")

openai.api_key = OPENAI_API_KEY
JOB_DESCRIPTIONS_PATH = os.path.join(project_root, "data", "job_descriptions.csv")

def parse_job_description(text):
    job_data = {
        "required_skills": [],
        "required_experience": 0.0,
        "required_education": [],
        "required_certifications": []
    }
    
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if any(kw in line.lower() for kw in ["skills", "experience in", "knowledge of", "proficient in", "expertise"]):
            skills_part = line.split(":", 1)[-1].strip() if ":" in line else line
            skills = re.split(r'[,\s;]+', skills_part)
            job_data["required_skills"].extend(s.strip() for s in skills if s.strip() and s.lower() not in ["in", "and", "of"])
        
        match = re.search(r"(\d+\.?\d*)\+?\s*years?", line, re.IGNORECASE)
        if match:
            job_data["required_experience"] = float(match.group(1))
        
        if "degree" in line.lower() or any(kw in line.lower() for kw in ["bachelor", "master", "phd", "diploma"]):
            job_data["required_education"].append(line.strip())
        
        if any(kw in line.lower() for kw in ["certification", "certified", "license", "credential"]):
            job_data["required_certifications"].append(line.strip())
    
    if not job_data["required_skills"]:
        job_data["required_skills"] = []
    if not job_data["required_experience"]:
        job_data["required_experience"] = 0.0
    
    logging.info(f"Parsed job_data: {json.dumps(job_data)}")
    return job_data

def parse_resumes(resume_files):
    resume_data_list = []
    for resume_file in resume_files:
        resume_data = parse_resume(resume_file)
        resume_data_list.append(resume_data)
        logging.info(f"Processed resume {resume_file.name}: {json.dumps(resume_data)}")
    return resume_data_list

def rank_candidates(job_data, resume_data_list):
    try:
        if not job_data or not resume_data_list:
            raise ValueError("Job data or resume data missing")
        
        payload = {"job_data": job_data, "resume_data_list": resume_data_list}
        logging.info(f"Sending request to {API_URL}/rank_candidates with payload: {json.dumps(payload, default=str)}")
        
        response = requests.post(
            f"{API_URL}/rank_candidates",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        
        logging.info(f"Response status: {response.status_code}, content: {response.text}")
        return response.json()["ranked_candidates"]
    
    except requests.ConnectionError as e:
        st.error(f"Cannot connect to backend at {API_URL}. Ensure the FastAPI server is running: {str(e)}")
        logging.error(f"Connection error: {str(e)}")
        return []
    except requests.Timeout as e:
        st.error(f"Request to {API_URL}/rank_candidates timed out: {str(e)}")
        logging.error(f"Timeout error: {str(e)}")
        return []
    except requests.RequestException as e:
        st.error(f"Error ranking candidates: {str(e)}")
        logging.error(f"Request exception: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Unexpected error in rank_candidates: {str(e)}")
        logging.error(f"Unexpected exception: {str(e)}")
        return []

def schedule_interviews(selected_candidates, job_title):
    schedules = []
    start_time = datetime.now() + timedelta(days=1)
    for i, candidate in enumerate(selected_candidates):
        interview_time = start_time + timedelta(hours=i)
        schedules.append({
            "email": candidate["resume"].get("email", ""),
            "time": interview_time.strftime("%Y-%m-%d %H:%M:%S"),
            "job_title": job_title
        })
    return schedules

def send_interview_email(schedule):
    subject = f"Interview Scheduled for {schedule['job_title']}"
    body = f"""
    Dear Candidate,

    We are pleased to inform you that you have been selected for an interview for the {schedule['job_title']} position.

    **Interview Details:**
    - Date & Time: {schedule['time']}
    - Mode: Virtual (Link to be shared closer to the date)

    Please confirm your availability by replying to this email.

    Best regards,
    HR Team
    """
    
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_EMAIL
    msg["To"] = schedule["email"]
    
    try:
        logging.info(f"Attempting to send email to {schedule['email']} from {SMTP_EMAIL}")
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.send_message(msg)
        logging.info(f"Email sent to {schedule['email']} for {schedule['job_title']} at {schedule['time']}")
        st.success(f"Email sent to {schedule['email']}")
    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"SMTP Authentication failed. Check SMTP_EMAIL={SMTP_EMAIL} and SMTP_PASSWORD in .env: {str(e)}"
        logging.error(error_msg)
        st.error(error_msg)
    except Exception as e:
        error_msg = f"Failed to send email to {schedule['email']}: {str(e)}"
        logging.error(error_msg)
        st.error(error_msg)

async def interview_websocket(audio_file):
    async with websockets.connect(WEBSOCKET_URL) as websocket:
        audio_data = audio_file.read()
        await websocket.send(audio_data)
        transcription = await websocket.recv()
        question_audio = await websocket.recv()
        return transcription, question_audio

def run_websocket(audio_file):
    def run_async_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(interview_websocket(audio_file))
        loop.close()
        return result
    
    with ThreadPoolExecutor() as pool:
        return pool.submit(run_async_in_thread).result()

def main():
    st.title("HR Tech Suite")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Candidate Ranking", "Audio Interview"])

    if page == "Candidate Ranking":
        st.header("Candidate Ranking")
        
        job_data_dict = {}
        try:
            df = pd.read_csv(JOB_DESCRIPTIONS_PATH, encoding='cp1252')
            if not all(col in df.columns for col in ["job title", "job description"]):
                st.error("CSV must contain 'job title' and 'job description'")
            else:
                job_data_dict = {row["job title"]: row["job description"] for _, row in df.iterrows()}
                st.write("Job Titles Loaded:", list(job_data_dict.keys()))
        except FileNotFoundError:
            st.error(f"Job descriptions file not found at {JOB_DESCRIPTIONS_PATH}")
            return
        except Exception as e:
            st.error(f"Error loading job descriptions: {str(e)}")
            return
        
        if job_data_dict:
            selected_job_title = st.selectbox("Select Job Title", options=list(job_data_dict.keys()))
            if selected_job_title:
                st.write(f"Selected Job Description for {selected_job_title}:")
                st.text(job_data_dict[selected_job_title])
        
        st.subheader("Upload Resumes (PDFs)")
        resume_files = st.file_uploader("Upload resumes (PDF)", type=["pdf"], accept_multiple_files=True)
        
        num_candidates = st.number_input("Number of Candidates to Select", min_value=1, value=1, step=1)
        
        if "ranked_candidates" not in st.session_state:
            st.session_state.ranked_candidates = []
            st.session_state.selected_job_title = None

        if st.button("Rank and Schedule Interviews") and resume_files and selected_job_title:
            if num_candidates > len(resume_files):
                st.error(f"Number of candidates ({num_candidates}) exceeds uploaded resumes ({len(resume_files)})")
            else:
                with st.spinner("Processing candidates..."):
                    job_description = job_data_dict[selected_job_title]
                    job_data = parse_job_description(job_description)
                    resume_data_list = parse_resumes(resume_files)
                    
                    if not resume_data_list:
                        st.error("No valid resumes parsed.")
                    else:
                        ranked_candidates = rank_candidates(job_data, resume_data_list)
                        if ranked_candidates:
                            st.session_state.ranked_candidates = ranked_candidates[:num_candidates]
                            st.session_state.selected_job_title = selected_job_title
                            st.subheader(f"Top {num_candidates} Ranked Candidates for {selected_job_title}")
                            table_data = [
                                {
                                    "Rank": candidate["rank"],
                                    "Score": candidate["score"],
                                    "ATS Score": candidate["ats_score"],
                                    "Skills": ", ".join(candidate["resume"]["skills"]) or "None",
                                    "Experience": ", ".join(candidate["resume"]["experience"]) or "None",
                                    "Education": ", ".join(candidate["resume"]["education"]) or "None",
                                    "Certifications": ", ".join(candidate["resume"]["certifications"]) or "None",
                                    "Email": candidate["resume"].get("email", "Not Found")
                                }
                                for candidate in st.session_state.ranked_candidates
                            ]
                            st.table(table_data)
                            
                            schedules = schedule_interviews(st.session_state.ranked_candidates, selected_job_title)
                            st.subheader("Interview Schedules")
                            for schedule in schedules:
                                st.write(f"Email: {schedule['email']}, Time: {schedule['time']}")
                                if schedule['email']:
                                    send_interview_email(schedule)
                                else:
                                    st.warning(f"No email found for candidate at rank {schedules.index(schedule) + 1}")
                            logging.info(f"Candidates ranked and interviews scheduled for {selected_job_title}")
                        else:
                            st.warning("No candidates ranked. Check backend logs for details.")

    elif page == "Audio Interview":
        st.header("Audio Interview")
        
        if not st.session_state.get("ranked_candidates") or not st.session_state.get("selected_job_title"):
            st.warning("Please rank candidates first in the 'Candidate Ranking' section.")
        else:
            job_role = st.session_state.selected_job_title
            st.write(f"Conducting real-time audio interview for: **{job_role}**")
            num_questions = st.slider("Number of Questions", 1, 5, 3)
            
            if st.button("Start Audio Interview"):
                with st.spinner("Starting live audio interview... Please answer each question aloud within 15 seconds when prompted."):
                    progress_text = st.empty()
                    progress_text.text("Preparing interview...")

                    def run_interview():
                        interview_data = {"job_role": job_role, "timestamp": datetime.now().isoformat(), "questions": [], "responses": [], "feedback": "Not generated yet"}
                        audio_files = []
                        try:
                            interview_data, audio_files = conduct_live_interview(job_role, num_questions)
                        except Exception as e:
                            logging.error(f"Interview thread failed: {str(e)}")
                            interview_data["feedback"] = f"Interview failed: {str(e)}"
                        finally:
                            st.session_state.interview_data = interview_data
                            st.session_state.audio_files = audio_files

                    interview_thread = threading.Thread(target=run_interview)
                    interview_thread.start()
                    
                    total_timeout = num_questions * 30
                    for i in range(total_timeout):
                        if not interview_thread.is_alive():
                            progress_text.text("Interview completed, displaying results...")
                            break
                        progress_text.text(f"Processing question {len(st.session_state.get('interview_data', {}).get('questions', [])) + 1}/{num_questions}... ({i}/{total_timeout}s)")
                        time.sleep(1)
                    else:
                        st.error(f"Interview process timed out after {total_timeout} seconds. Check interview.log for details.")
                        logging.error("Interview thread timed out.")
                        interview_thread.join(timeout=1)
                        progress_text.text("Interview timed out, displaying partial results...")

                    # Display results in the main thread
                    if "interview_data" in st.session_state:
                        with st.container():
                            st.subheader("Interview Results")
                            for i, (q, r) in enumerate(zip(st.session_state.interview_data["questions"], st.session_state.interview_data["responses"])):
                                st.write(f"**Question {i + 1}:** {q}")
                                st.write(f"**Transcribed Response:** {r['response']}")
                                st.write(f"**Sentiment:** {r['sentiment']:.2f}, **Score:** {r['score']}")
                                st.write(f"**Explanation:** {r['explanation']}")
                                audio_file = st.session_state.audio_files[i] if i < len(st.session_state.audio_files) else None
                                if audio_file and os.path.exists(audio_file):
                                    st.audio(audio_file, format="audio/wav")
                                else:
                                    st.warning(f"Audio file for Question {i + 1} not found.")
                            st.subheader("Interview Feedback")
                            st.write(st.session_state.interview_data["feedback"])
                        progress_text.text("Interview results displayed.")
                    else:
                        st.error("No interview data received. Check interview.log for details.")
                        progress_text.text("Interview failed.")

if __name__ == "__main__":
    main()