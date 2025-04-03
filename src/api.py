from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException
from pydantic import BaseModel
import shutil
import os
import logging
import asyncio
import whisper
import pyttsx3
import json
from src.parsing import parse_resume, extract_text_from_pdf, train_parsing_model
from typing import List

app = FastAPI()

logging.basicConfig(filename='data/logs/api.log', level=logging.INFO)

# Load Whisper model
whisper_model = whisper.load_model("base")

# Initialize text-to-speech
tts_engine = pyttsx3.init()

class JobData(BaseModel):
    required_skills: List[str]
    required_experience: float
    required_education: List[str]
    required_certifications: List[str]

class ResumeData(BaseModel):
    skills: List[str]
    experience: List[str]
    education: List[str]
    certifications: List[str]
    email: str

class RankRequest(BaseModel):
    job_data: JobData
    resume_data_list: List[ResumeData]

@app.post("/rank_candidates")
async def rank_candidates_endpoint(request: RankRequest):
    """Rank candidates based on parsed resume data and job description."""
    try:
        job_data = request.job_data.dict()
        resume_data_list = [rd.dict() for rd in request.resume_data_list]
        
        ranked_candidates = []
        for i, resume in enumerate(resume_data_list):
            skill_match = len(set(job_data["required_skills"]) & set(resume["skills"])) / len(job_data["required_skills"]) if job_data["required_skills"] else 0
            exp_match = 1.0 if any(job_data["required_experience"] <= len(exp) for exp in resume["experience"]) else 0.5
            score = (skill_match * 70) + (exp_match * 30)
            ats_score = skill_match * 100
            
            ranked_candidates.append({
                "rank": i + 1,
                "score": round(score, 2),
                "ats_score": round(ats_score, 2),
                "resume": resume
            })
        
        ranked_candidates.sort(key=lambda x: x["score"], reverse=True)
        for i, candidate in enumerate(ranked_candidates):
            candidate["rank"] = i + 1
        
        logging.info(f"Ranked candidates: {json.dumps(ranked_candidates)}")
        return {"ranked_candidates": ranked_candidates}
    except Exception as e:
        logging.error(f"Error in rank_candidates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/interview")
async def interview_websocket(websocket: WebSocket):
    """Handle live audio interview with real-time transcription and question generation."""
    await websocket.accept()
    try:
        previous_response = None
        while True:
            audio_chunk = await websocket.recv()
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_chunk)
            transcription = whisper_model.transcribe("temp_audio.wav")["text"]
            logging.info(f"Transcribed: {transcription}")
            question = "Tell me more about your experience."  # Placeholder
            tts_engine.save_to_file(question, "temp_question.wav")
            tts_engine.runAndWait()
            with open("temp_question.wav", "rb") as f:
                audio_response = f.read()
            await websocket.send(audio_response)
            os.remove("temp_audio.wav")
            os.remove("temp_question.wav")
    except Exception as e:
        logging.error(f"WebSocket error: {str(e)}")
        await websocket.close()

@app.post("/train_parsing_model")
async def train_parsing_model_endpoint(resumes: List[UploadFile] = File(...), labels: List[UploadFile] = File(...)):
    """Train the resume parsing model with multiple resumes and labels."""
    try:
        if len(resumes) != len(labels):
            raise ValueError("Number of resumes and label files must match.")
        
        training_data = []
        temp_files = []
        for resume, label in zip(resumes, labels):
            resume_path = f"data/resumes/{resume.filename}"
            os.makedirs("data/resumes", exist_ok=True)
            with open(resume_path, "wb") as buffer:
                shutil.copyfileobj(resume.file, buffer)
            temp_files.append(resume_path)
            text = extract_text_from_pdf(resume_path)
            label_data = json.load(label.file)
            if label_data.get('text') != text:
                raise ValueError(f"Text mismatch for {resume.filename}")
            training_data.append(label_data)
        
        train_parsing_model(training_data)
        
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        return {"message": "Model trained successfully"}
    except Exception as e:
        logging.error(f"Error in train_parsing_model: {str(e)}")
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)