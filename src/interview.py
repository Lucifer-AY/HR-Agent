import openai
import pyaudio
import wave
import whisper
import logging
import time
import os
import json
import re
import random
from textblob import TextBlob
from datetime import datetime
import threading
import winspeech

# Configure logging
log_dir = 'data/logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'interview.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s - [File: %(filename)s, Line: %(lineno)d]'
)

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.error("OPENAI_API_KEY not found in environment variables.")
    raise ValueError("OPENAI_API_KEY is required but not set in .env file.")
try:
    openai_client = openai.OpenAI(api_key=api_key)
    # Test API key with a small request
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=5
    )
    logging.info("OpenAI client initialized and validated successfully.")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client or quota exceeded: {str(e)}")
    openai_client = None  # Fallback to no OpenAI if quota is exceeded

# Load Whisper model
logging.debug("Loading Whisper model 'tiny' for faster real-time transcription...")
try:
    whisper_model = whisper.load_model("tiny")
    logging.info("Whisper model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Whisper model: {str(e)}")
    raise

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
        stop_event = threading.Event()

        def record():
            try:
                for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                    if stop_event.is_set():
                        logging.debug("Recording stopped by stop_event")
                        break
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
            except Exception as e:
                logging.error(f"Recording thread error: {str(e)}")

        record_thread = threading.Thread(target=record)
        record_thread.start()
        record_thread.join(timeout)

        if record_thread.is_alive():
            logging.error("Recording timed out, forcing stop...")
            stop_event.set()
            raise TimeoutError("Audio recording exceeded timeout.")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        wf = wave.open(output_file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        time.sleep(0.5)  # Wait for file to be fully written
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
        logging.info(f"Transcribing file at: {os.path.abspath(audio_file)}")
        transcription = whisper_model.transcribe(audio_file, fp16=False)["text"].strip()
        if not transcription:
            transcription = "No speech detected."
        logging.info(f"Transcribed in real-time: {transcription}")
        return transcription
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
        if openai_client:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant generating interview questions."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=100,
                    temperature=0.7
                )
                question = response.choices[0].message.content.strip()
                logging.info(f"Generated question: {question}")
            except Exception as e:
                logging.error(f"Error generating question: {str(e)}")
                question = f"Tell me more about your experience with {job_role}."
        else:
            question = f"Tell me more about your experience with {job_role}."
    return question

def analyze_response(response):
    logging.debug(f"Analyzing response: {response}")
    if not response.strip() or "failed" in response.lower() or "no audio" in response.lower():
        logging.warning("Empty or failed response received.")
        return 0.0, 0, "No valid response provided."
    if openai_client:
        try:
            sentiment = TextBlob(response).sentiment.polarity
            eval_prompt = (f"Evaluate the response '{response}' for clarity, relevance, and technical accuracy "
                           f"on a scale of 0 to 10. Provide a brief explanation.")
            eval_response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an evaluator scoring interview responses."},
                    {"role": "user", "content": eval_prompt}
                ],
                max_tokens=50
            )
            score_text = eval_response.choices[0].message.content.strip()
            score = int(re.search(r'\d+', score_text).group()) if re.search(r'\d+', score_text) else 5
            logging.info(f"Response analysis - Sentiment: {sentiment:.2f}, Score: {score}, Explanation: {score_text}")
            return sentiment, score, score_text
        except Exception as e:
            logging.error(f"Error analyzing response: {str(e)}")
            return 0.0, 5, f"Analysis failed due to quota or error: {str(e)}"
    else:
        sentiment = TextBlob(response).sentiment.polarity
        return sentiment, 5, "Default score due to OpenAI unavailability."

def generate_feedback(interview_data):
    logging.debug("Generating feedback")
    total_score = sum(r["score"] for r in interview_data["responses"])
    avg_score = total_score / len(interview_data["responses"]) if interview_data["responses"] else 0
    feedback_prompt = (
        f"Provide feedback for a {interview_data['job_role']} interview based on these responses:\n"
        f"{json.dumps(interview_data['responses'], indent=2)}\n"
        f"Average score: {avg_score:.2f}. Give strengths, weaknesses, and suggestions."
    )
    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant providing interview feedback."},
                    {"role": "user", "content": feedback_prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            feedback = response.choices[0].message.content.strip()
            logging.info(f"Generated feedback: {feedback}")
            return feedback
        except Exception as e:
            logging.error(f"Error generating feedback: {str(e)}")
            return f"Feedback unavailable due to quota or error: {str(e)}. Average score: {avg_score:.2f}"
    else:
        return f"Feedback unavailable due to OpenAI unavailability. Average score: {avg_score:.2f}"

def conduct_live_interview(job_role, num_questions=3, output_dir="data/interviews"):
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
        interview_data["feedback"] = f"Feedback generation failed: {str(e)}"

    output_file = os.path.join(output_dir, f"interview_{job_role}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, "w") as f:
        json.dump(interview_data, f, indent=4)
    logging.info(f"Interview completed and saved to {output_file}")

    return interview_data, audio_files

if __name__ == "__main__":
    job_roles = ["Software Engineer", "Data Scientist", "Project Manager", "DevOps Engineer"]
    selected_role = "Software Engineer"
    interview_data, audio_files = conduct_live_interview(selected_role)