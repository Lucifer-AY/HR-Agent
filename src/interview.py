import google.generativeai as genai
import pyaudio
import wave
import speech_recognition as sr
import logging
import time
import os
import json
import re
import random
from textblob import TextBlob
from datetime import datetime
import winspeech
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
log_dir = 'C:/HR_Agent_Logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'interview.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s - [File: %(filename)s, Line: %(lineno)d]'
)

# Initialize Gemini client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logging.error("GEMINI_API_KEY not found in environment variables.")
    raise ValueError("GEMINI_API_KEY is required but not set in .env file.")
try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')  # Use a suitable Gemini model
    test_response = model.generate_content("Test")
    logging.info(f"Gemini client initialized and validated successfully. Test response: {test_response.text}")
except Exception as e:
    logging.error(f"Failed to initialize Gemini client: {str(e)}")
    model = None

# Initialize SpeechRecognition
recognizer = sr.Recognizer()
logging.info("SpeechRecognition recognizer initialized.")

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 15

# Predefined question templates by job role
QUESTION_TEMPLATES = {
    "Software Engineer": [
        "What programming languages are you proficient in?",
        "Can you describe a challenging bug you’ve fixed?",
        "How do you ensure code quality in your projects?",
        "Tell me about a project you’ve worked on recently.",
        "What’s your experience with version control systems?"
    ],
    "Data Scientist": [
        "What machine learning models have you used?",
        "How do you handle missing data in a dataset?",
        "Tell me about a data visualization you’ve created.",
        "What’s your experience with Python or R?",
        "Can you explain a statistical method you’ve applied?"
    ],
    "Project Manager": [
        "How do you manage project timelines?",
        "Tell me about a time you resolved a team conflict.",
        "What tools do you use for project tracking?",
        "How do you prioritize tasks in a project?",
        "What’s your experience with Agile methodologies?"
    ],
    "DevOps Engineer": [
        "What tools do you use for CI/CD pipelines?",
        "How do you handle infrastructure as code?",
        "Tell me about a time you improved system reliability.",
        "What’s your experience with containerization?",
        "How do you monitor production systems?"
    ],
    "Cloud Engineer": [
        "What cloud platforms are you experienced with?",
        "How do you manage cloud security?",
        "Tell me about a cloud migration you’ve performed.",
        "What’s your experience with Infrastructure as Code?",
        "How do you optimize cloud costs?"
    ]
}

def record_audio(output_file, timeout=20):
    logging.debug(f"Starting audio recording to {output_file}")
    audio = None
    stream = None
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        logging.info(f"Recording audio to {output_file} for {RECORD_SECONDS} seconds...")
        frames = []

        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        wf = wave.open(output_file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        time.sleep(1)
        file_size = os.path.getsize(output_file)
        logging.info(f"Audio recorded and saved to {output_file}, size: {file_size} bytes")
        if file_size < 1000:
            logging.warning("Audio file is nearly empty, check microphone.")
        return output_file
    except Exception as e:
        logging.error(f"Error recording audio: {str(e)}")
        if stream:
            stream.stop_stream()
            stream.close()
        if audio:
            audio.terminate()
        return None

def transcribe_audio(audio_file, timeout=10):
    logging.debug(f"Starting real-time transcription of {audio_file}")
    if not audio_file or not os.path.exists(audio_file):
        logging.error(f"No valid audio file to transcribe: {audio_file}")
        return "No audio recorded."
    try:
        absolute_path = os.path.abspath(audio_file)
        logging.info(f"Transcribing file at: {absolute_path}")
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"File not found immediately before transcription: {absolute_path}")
        file_size = os.path.getsize(absolute_path)
        logging.debug(f"File size before transcription: {file_size} bytes")
        
        with sr.AudioFile(absolute_path) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)
        if not transcription:
            transcription = "No speech detected."
        logging.info(f"Transcribed in real-time: {transcription}")
        return transcription
    except sr.UnknownValueError:
        logging.warning("Google Speech Recognition could not understand the audio.")
        return "No speech detected."
    except sr.RequestError as e:
        logging.error(f"Google Speech Recognition request failed: {str(e)}")
        return f"Transcription failed: {str(e)}"
    except Exception as e:
        logging.error(f"Real-time transcription error: {str(e)}")
        return f"Transcription failed: {str(e)}"

def speak_question(question, timeout=10):
    logging.debug(f"Attempting to speak question: {question}")
    try:
        winspeech.say_wait(question)
        logging.info(f"Successfully spoke question: {question}")
        return True
    except Exception as e:
        logging.error(f"TTS error: {str(e)}")
        return False

def generate_question(job_role, previous_response=None, previous_score=None):
    logging.debug(f"Generating question for {job_role}")
    if not previous_response:
        questions = QUESTION_TEMPLATES.get(job_role, ["Tell me about your experience with this role."])
        question = random.choice(questions)
    else:
        difficulty = "hard" if previous_score and previous_score > 7 else "medium"
        prompt = (f"Based on the response '{previous_response}' for a {job_role}, "
                  f"generate a {difficulty}-difficulty follow-up question.")
        if model:
            try:
                response = model.generate_content(prompt)
                question = response.text.strip()
                logging.info(f"Generated question: {question}")
            except Exception as e:
                logging.error(f"Error generating question with Gemini: {str(e)}")
                question = f"Tell me more about your experience with {job_role}."
        else:
            question = f"Tell me more about your experience with {job_role}."
    return question

def analyze_response(response):
    logging.debug(f"Analyzing response: {response}")
    if not response.strip() or "failed" in response.lower() or "no audio" in response.lower():
        logging.warning("Empty or failed response received.")
        return 0.0, 0, "No valid response provided."
    if model:
        try:
            sentiment = TextBlob(response).sentiment.polarity
            eval_prompt = (f"Evaluate the response '{response}' for clarity, relevance, and technical accuracy "
                           f"on a scale of 0 to 10. Provide a brief explanation.")
            eval_response = model.generate_content(eval_prompt)
            score_text = eval_response.text.strip()
            score = int(re.search(r'\d+', score_text).group()) if re.search(r'\d+', score_text) else 5
            logging.info(f"Response analysis - Sentiment: {sentiment:.2f}, Score: {score}, Explanation: {score_text}")
            return sentiment, score, score_text
        except Exception as e:
            logging.error(f"Error analyzing response with Gemini: {str(e)}")
            return 0.0, 5, f"Analysis failed due to Gemini error: {str(e)}"
    else:
        sentiment = TextBlob(response).sentiment.polarity
        return sentiment, 5, "Default score due to Gemini unavailability."

def generate_feedback(interview_data):
    logging.debug("Generating feedback")
    total_score = sum(r["score"] for r in interview_data["responses"])
    avg_score = total_score / len(interview_data["responses"]) if interview_data["responses"] else 0
    
    if not model:
        logging.warning("Gemini model unavailable, using fallback feedback.")
        return f"Fallback feedback: Based on your responses, your average score is {avg_score:.2f}. Please review your answers for more detailed insights."
    
    feedback_prompt = (
        f"Provide feedback for a {interview_data['job_role']} interview based on these responses:\n"
        f"{json.dumps(interview_data['responses'], indent=2)}\n"
        f"Average score: {avg_score:.2f}. Give strengths, weaknesses, and suggestions."
    )
    try:
        logging.debug(f"Sending feedback prompt to Gemini: {feedback_prompt}")
        response = model.generate_content(feedback_prompt)
        feedback = response.text.strip()
        logging.info(f"Generated feedback: {feedback}")
        return feedback
    except Exception as e:
        logging.error(f"Error generating feedback with Gemini: {str(e)}")
        return f"Feedback unavailable due to Gemini error: {str(e)}. Average score: {avg_score:.2f}"

def conduct_live_interview(job_role, num_questions=3, output_dir="C:/HR_Agent_Interviews"):
    os.makedirs(output_dir, exist_ok=True)
    interview_data = {"job_role": job_role, "timestamp": datetime.now().isoformat(), "questions": [], "responses": []}
    audio_files = []

    logging.info(f"Starting live interview for {job_role} with {num_questions} questions")

    for i in range(num_questions):
        logging.debug(f"Processing question {i + 1}")
        try:
            previous_response = interview_data["responses"][-1]["response"] if i > 0 else None
            previous_score = interview_data["responses"][-1]["score"] if i > 0 else None
            question = generate_question(job_role, previous_response, previous_score)
            interview_data["questions"].append(question)

            logging.info(f"Question {i + 1}: {question}")
            if not speak_question(f"Question {i + 1}: {question}", timeout=10):
                logging.warning(f"Fallback to text for Question {i + 1}: {question}")

            audio_file = os.path.abspath(os.path.join(output_dir, f"response_{i + 1}_{job_role}_{time.strftime('%Y%m%d_%H%M%S')}.wav"))
            audio_result = record_audio(audio_file, timeout=20)
            audio_files.append(audio_result)
            logging.info(f"Recorded audio for question {i + 1}")

            if audio_result and os.path.exists(audio_result):
                response = transcribe_audio(audio_result, timeout=10)
            else:
                response = "No audio recorded."
                logging.warning(f"No audio file for response {i + 1}")

            sentiment, score, explanation = analyze_response(response)
            interview_data["responses"].append({
                "response": response,
                "sentiment": sentiment,
                "score": score,
                "explanation": explanation,
                "audio_file": audio_result
            })
            logging.info(f"Processed response {i + 1} in real-time")

        except Exception as e:
            logging.error(f"Error during question {i + 1}: {str(e)}")
            interview_data["responses"].append({
                "response": f"Processing failed: {str(e)}",
                "sentiment": 0.0,
                "score": 0,
                "explanation": "Error occurred.",
                "audio_file": None
            })
            audio_files.append(None)

    try:
        feedback = generate_feedback(interview_data)
        interview_data["feedback"] = feedback
    except Exception as e:
        logging.error(f"Error generating feedback in conduct_live_interview: {str(e)}")
        interview_data["feedback"] = f"Feedback generation failed: {str(e)}"

    output_file = os.path.join(output_dir, f"interview_{job_role}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, "w") as f:
        json.dump(interview_data, f, indent=4)
    logging.info(f"Interview completed and saved to {output_file}")
    logging.debug(f"Returning interview_data: {json.dumps(interview_data, default=str)}")

    return interview_data, audio_files

if __name__ == "__main__":
    job_roles = ["Software Engineer", "Data Scientist", "Project Manager", "DevOps Engineer", "Cloud Engineer"]
    selected_role = "Cloud Engineer"
    interview_data, audio_files = conduct_live_interview(selected_role)