from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet
import numpy as np
import logging

# Configure logging
logging.basicConfig(filename='data/logs/matching.log', level=logging.INFO)

# Load SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def expand_keywords(keywords):
    """Expand keywords using WordNet synonyms and hypernyms."""
    expanded = set(keywords)
    for keyword in keywords:
        for syn in wordnet.synsets(keyword):
            # Add synonyms
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace('_', ' '))
            # Add hypernyms (broader terms)
            for hypernym in syn.hypernyms():
                for lemma in hypernym.lemmas():
                    expanded.add(lemma.name().replace('_', ' '))
    logging.info(f"Expanded keywords: {keywords} -> {expanded}")
    return list(expanded)

def compute_similarity(resume_data, job_data):
    """Compute multi-faceted similarity score."""
    # Skill similarity with SBERT
    resume_skills = expand_keywords(resume_data.get('skills', []))
    job_skills = expand_keywords(job_data.get('required_skills', []))
    resume_vec = sbert_model.encode(" ".join(resume_skills))
    job_vec = sbert_model.encode(" ".join(job_skills))
    skill_similarity = cosine_similarity([resume_vec], [job_vec])[0][0]

    # Jaccard similarity for overlap
    resume_set = set(resume_skills)
    job_set = set(job_skills)
    jaccard = len(resume_set.intersection(job_set)) / len(resume_set.union(job_set)) if resume_set.union(job_set) else 0

    # Experience matching
    resume_exp = max([float(re.sub(r'[^\d.]', '', exp)) for exp in resume_data.get('experience', ['0'])], default=0)
    job_exp = job_data.get('required_experience', 0)
    exp_match = min(resume_exp / job_exp, 1.0) if job_exp > 0 else 1.0

    # Combined weighted score
    weights = {'skill_similarity': 0.5, 'jaccard': 0.3, 'exp_match': 0.2}
    match_score = (weights['skill_similarity'] * skill_similarity + 
                   weights['jaccard'] * jaccard + 
                   weights['exp_match'] * exp_match)
    logging.info(f"Similarity score: {match_score:.2f} (Skills: {skill_similarity:.2f}, Jaccard: {jaccard:.2f}, Exp: {exp_match:.2f})")
    return match_score

# Example usage
if __name__ == "__main__":
    resume_data = {"skills": ["python", "machine learning"], "experience": ["5 years"]}
    job_data = {"required_skills": ["python", "data science"], "required_experience": 3}
    score = compute_similarity(resume_data, job_data)
    print(f"Match Score: {score:.2f}")