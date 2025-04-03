from typing import List, Dict
import re
import logging
from datetime import datetime

logging.basicConfig(filename='data/logs/ranking.log', level=logging.INFO)

def compute_skill_similarity(resume_skills: List[str], required_skills: List[str]) -> float:
    """Compute similarity score between resume skills and required skills."""
    resume_skills = set(skill.lower().strip() for skill in resume_skills if skill)
    required_skills = set(skill.lower().strip() for skill in required_skills if skill)
    if not required_skills:
        return 0.0
    
    matches = resume_skills.intersection(required_skills)
    partial_matches = sum(1 for r_skill in resume_skills for j_skill in required_skills 
                         if r_skill in j_skill or j_skill in r_skill) - len(matches)
    
    score = len(matches) * 5 + partial_matches * 2
    max_possible = len(required_skills) * 5
    return min(score / max_possible, 1.0) * 50 if max_possible > 0 else 0

def compute_experience_score(resume_experience: List[str], required_exp: float) -> float:
    """Compute experience score with robust handling."""
    total_years = 0
    current_year = datetime.now().year
    
    for exp in resume_experience:
        match = re.search(r"(\d+\.?\d*)\s*years?", exp, re.IGNORECASE)
        if match:
            total_years += float(match.group(1))
        else:
            date_match = re.findall(r"(\d{4})\s*[-–]\s*(present|current|\d{4})", exp.lower(), re.IGNORECASE)
            if date_match:
                for start, end in date_match:
                    end_year = current_year if end.lower() in ["present", "current"] else int(end)
                    years = max(end_year - int(start), 0)
                    total_years += years
    
    if required_exp <= 0:
        return 30 if total_years > 0 else 0
    return min(total_years / required_exp, 1.0) * 30 if total_years > 0 else 0

def compute_education_similarity(resume_edu: List[str], required_edu: List[str]) -> float:
    """Compute similarity score for education."""
    resume_edu = set(edu.lower().strip() for edu in resume_edu if edu)
    required_edu = set(edu.lower().strip() for edu in required_edu if edu)
    if not required_edu:
        return 0.0
    
    matches = 0
    degree_levels = {
        "associate": 1, "diploma": 1,
        "bachelor": 2, "bs": 2, "ba": 2,
        "master": 3, "ms": 3, "mba": 3,
        "phd": 4, "doctorate": 4
    }
    
    for r_edu in resume_edu:
        r_level = max((degree_levels.get(kw, 0) for kw in degree_levels if kw in r_edu), default=0)
        for j_edu in required_edu:
            j_level = max((degree_levels.get(kw, 0) for kw in degree_levels if kw in j_edu), default=0)
            if r_level >= j_level and j_level > 0:
                matches += 1
                break
    
    score = matches * 10
    max_possible = len(required_edu) * 10
    return min(score / max_possible, 1.0) * 15 if max_possible > 0 else 0

def compute_certification_similarity(resume_certs: List[str], required_certs: List[str]) -> float:
    """Compute similarity score for certifications."""
    resume_certs = set(cert.lower().strip() for cert in resume_certs if cert)
    required_certs = set(cert.lower().strip() for cert in required_certs if cert)
    if not required_certs:
        return 0.0
    
    matches = 0
    for r_cert in resume_certs:
        for j_cert in required_certs:
            if r_cert in j_cert or j_cert in r_cert or \
               any(kw in r_cert for kw in ["certified", "certification", "license", "credential"]):
                matches += 1
                break
    
    score = matches * 5
    max_possible = len(required_certs) * 5
    return min(score / max_possible, 1.0) * 5 if max_possible > 0 else 0

def compute_ats_score(resume: Dict, job_data: Dict) -> float:
    """Compute ATS compatibility score (out of 20)."""
    ats_score = 0
    resume_text = " ".join(resume["skills"] + resume["experience"] + resume["education"] + resume["certifications"]).lower()
    job_text = " ".join(job_data["required_skills"] + job_data["required_education"] + job_data["required_certifications"]).lower()
    
    # Keyword matches
    job_keywords = set(re.split(r'[,\s;]+', job_text))
    resume_keywords = set(re.split(r'[,\s;]+', resume_text))
    keyword_matches = len(job_keywords.intersection(resume_keywords))
    ats_score += min(keyword_matches / len(job_keywords), 1.0) * 10 if job_keywords else 0
    
    # Section presence
    if resume["skills"]:
        ats_score += 2
    if resume["experience"]:
        ats_score += 2
    if resume["education"]:
        ats_score += 2
    if resume["certifications"]:
        ats_score += 2
    
    # Formatting (content length as proxy)
    total_length = len(resume_text)
    ats_score += min(total_length / 200, 4)  # Up to 4 points
    
    return min(ats_score, 20)  # Cap at 20

def rank_candidates(resume_data_list: List[Dict], job_data: Dict) -> List[Dict]:
    """Rank resumes with ATS scoring included."""
    scored_candidates = []
    
    for resume in resume_data_list:
        resume = {
            "skills": resume.get("skills", []),
            "experience": resume.get("experience", []),
            "education": resume.get("education", []),
            "certifications": resume.get("certifications", [])
        }
        
        skills_score = compute_skill_similarity(resume["skills"], job_data.get("required_skills", []))
        exp_score = compute_experience_score(resume["experience"], job_data.get("required_experience", 0.0))
        edu_score = compute_education_similarity(resume["education"], job_data.get("required_education", []))
        cert_score = compute_certification_similarity(resume["certifications"], job_data.get("required_certifications", []))
        ats_score = compute_ats_score(resume, job_data)
        
        total_score = skills_score + exp_score + edu_score + cert_score + ats_score
        
        logging.info(f"Resume scoring: Skills={skills_score:.1f}, Experience={exp_score:.1f}, Education={edu_score:.1f}, Certs={cert_score:.1f}, ATS={ats_score:.1f}, Total={total_score:.1f}")
        
        scored_candidates.append({
            "score": round(total_score, 1),
            "ats_score": round(ats_score, 1),
            "resume": resume
        })
    
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    ranked_candidates = []
    for rank, candidate in enumerate(scored_candidates, start=1):
        candidate["rank"] = rank
        ranked_candidates.append(candidate)
    
    logging.info(f"Ranked {len(ranked_candidates)} candidates: {[f'Rank {c['rank']}: {c['score']} (ATS: {c['ats_score']})' for c in ranked_candidates]}")
    return ranked_candidates