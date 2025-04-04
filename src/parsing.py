import os
import tempfile
from io import BytesIO
import re
from pdfminer.high_level import extract_text
from pyresparser import ResumeParser
import spacy
from types import SimpleNamespace

# Hardcode the skills.csv path
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
SKILLS_CSV_PATH = os.path.join(PROJECT_ROOT, 'env', 'Lib', 'site-packages', 'pyresparser', 'data', 'skills.csv')

def extract_text_from_pdf(pdf_path):
    """Extract raw text from a PDF file."""
    return extract_text(pdf_path).strip()

def parse_resume(resume_file):
    """
    Parse a single resume file using pyresparser, bypassing config.cfg.
    
    Args:
        resume_file: Uploaded file object (PDF) from Streamlit or file path.
    
    Returns:
        dict: Resume data with skills, experience, education, certifications, and email.
    """
    resume_data = {
        "skills": [],
        "experience": [],
        "education": [],
        "certifications": [],
        "email": ""
    }
    
    # Handle both file objects and paths
    if isinstance(resume_file, str):
        temp_path = resume_file
        delete_temp = False
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            resume_file.seek(0)
            temp_file.write(resume_file.read())
            temp_path = temp_file.name
        delete_temp = True
    
    try:
        # Ensure skills.csv exists
        if not os.path.exists(SKILLS_CSV_PATH):
            raise FileNotFoundError(f"skills.csv not found at {SKILLS_CSV_PATH}")

        # Load skills database
        with open(SKILLS_CSV_PATH, 'r', encoding='utf-8') as f:
            skills_db = set(line.strip().lower() for line in f if line.strip())

        # Mock config object to bypass config.cfg
        mock_config = SimpleNamespace()
        mock_config.nlp_dir = os.path.dirname(SKILLS_CSV_PATH)
        mock_config.skills_csv_path = SKILLS_CSV_PATH

        # Initialize pyresparser with mocked config
        parser = ResumeParser(temp_path)
        if hasattr(parser, 'config'):
            parser.config = mock_config
        parsed_data = parser.get_extracted_data()

        # Load spaCy model for enhanced skill extraction
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(parsed_data.get("text", ""))
        extracted_skills = [token.text for token in doc if token.text.lower() in skills_db]

        resume_data["skills"] = list(set(parsed_data.get("skills", []) + extracted_skills)) or []
        resume_data["experience"] = parsed_data.get("designation", []) or []
        resume_data["education"] = parsed_data.get("degree", []) or []
        resume_data["certifications"] = []  # pyresparser doesn't extract this natively
        resume_data["email"] = parsed_data.get("email", "")

        print(f"Parsed resume with pyresparser: {resume_data}")  # For debugging
    
    except Exception as e:
        print(f"Error parsing resume with pyresparser: {str(e)}")  # For debugging
        
        # Fallback parsing
        text = extract_text(temp_path).strip()
        if text:
            lines = text.split("\n")
            current_section = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if any(kw in line.lower() for kw in ["skills", "technical skills", "proficiencies"]):
                    current_section = "skills"
                    skills_part = line.split(":", 1)[-1].strip() if ":" in line else line
                    skills = [s.strip() for s in re.split(r'[,\s;•-]+', skills_part) if s.strip() and len(s) > 2 and s.lower() not in ["and", "the", "date", "place", "from", "using", "with"]]
                    resume_data["skills"].extend(skills)
                elif any(kw in line.lower() for kw in ["experience", "work history", "employment", "professional experience"]):
                    current_section = "experience"
                elif any(kw in line.lower() for kw in ["education", "degree", "academic", "qualifications"]):
                    current_section = "education"
                elif any(kw in line.lower() for kw in ["certifications", "certified", "license", "credentials"]):
                    current_section = "certifications"
                
                elif current_section == "skills":
                    skills = [s.strip() for s in re.split(r'[,\s;•-]+', line) if s.strip() and len(s) > 2 and s.lower() not in ["and", "the", "date", "place", "from", "using", "with"]]
                    resume_data["skills"].extend(skills)
                elif current_section == "experience" and re.search(r"\b(19|20)\d{2}\b|\bpresent\b", line, re.IGNORECASE):
                    resume_data["experience"].append(line)
                elif current_section == "education" and any(kw in line.lower() for kw in ["bachelor", "master", "phd", "degree", "diploma"]):
                    resume_data["education"].append(line)
                elif current_section == "certifications":
                    resume_data["certifications"].append(line)
                
                email_matches = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}", line)
                if email_matches and not resume_data["email"]:
                    resume_data["email"] = email_matches[0]
        
        resume_data["skills"] = list(set(s.lower() for s in resume_data["skills"] if s.isalpha()))
        resume_data["experience"] = list(set(resume_data["experience"]))
        resume_data["education"] = list(set(resume_data["education"]))
        resume_data["certifications"] = list(set(resume_data["certifications"]))
        
        print(f"Fallback parsed resume: {resume_data}")  # For debugging
    
    finally:
        if delete_temp and os.path.exists(temp_path):
            os.remove(temp_path)
    
    return resume_data

def train_parsing_model(training_data):
    """Placeholder for training a parsing model (not implemented)."""
    # This could integrate with a machine learning model (e.g., spaCy NER) in the future
    print(f"Training model with {len(training_data)} samples (placeholder)")
    return None

if __name__ == "__main__":
    with open("C:/path/to/CV (1).pdf", "rb") as f:
        resume_data = parse_resume(f)
        print(resume_data)