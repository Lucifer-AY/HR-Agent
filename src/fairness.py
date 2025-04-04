import pandas as pd
import xgboost as xgb
import shap
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
import logging
import matplotlib.pyplot as plt
import os
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# Configure logging
log_dir = 'C:/HR_Agent_Logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'fairness.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - [File: %(filename)s, Line: %(lineno)d]'
)

# Load BERT for bias reduction in text features
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def preprocess_with_bert(text_list):
    """Use BERT to preprocess text data and reduce bias by generating embeddings."""
    embeddings = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        # Use mean of last hidden state as embedding
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)
    # Average embeddings across all samples for a fixed-size feature
    return np.mean(embeddings, axis=0)

def audit_fairness(resume_data_list, ranked_candidates, job_data):
    """Audit fairness of candidate ranking using AIF360 and explain with SHAP."""
    try:
        # Prepare data for fairness analysis
        data = pd.DataFrame([{
            'skill_similarity': sum(1 for skill in cand['resume']['skills'] if skill in job_data['required_skills']) / max(len(job_data['required_skills']), 1),
            'experience_match': len(cand['resume']['experience']) / max(job_data['required_experience'], 1),
            'education_match': 1 if any(edu in cand['resume']['education'] for edu in job_data['required_education']) else 0,
            'gender': 0 if cand['resume'].get('gender', 'Male') == 'Male' else 1,  # Assume Male=0, Female=1
            'age': cand['resume'].get('age', 30),  # Default age 30 if not provided
            'label': cand['score']
        } for cand in ranked_candidates])

        # Preprocess skills with BERT to reduce bias
        skill_texts = [' '.join(cand['resume']['skills']) for cand in ranked_candidates]
        bert_skill_embedding = preprocess_with_bert(skill_texts)
        data['skill_similarity'] = data['skill_similarity'] * np.mean(bert_skill_embedding)  # Adjust with BERT embedding factor

        # Train XGBoost model
        X = data.drop(['label', 'gender', 'age'], axis=1)
        y = data['label']
        model = xgb.XGBRegressor()
        model.fit(X, y)

        # Convert to binary labels for fairness metrics (e.g., score > median is positive)
        median_score = data['label'].median()
        dataset = BinaryLabelDataset(
            df=data.drop('label', axis=1).assign(label=(data['label'] > median_score).astype(int)),
            label_names=['label'],
            protected_attribute_names=['gender', 'age']
        )

        # Define privileged/unprivileged groups
        privileged_groups = [{'gender': 0}, {'age': lambda x: x <= 35}]
        unprivileged_groups = [{'gender': 1}, {'age': lambda x: x > 35}]
        metric = BinaryLabelDatasetMetric(dataset, privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)

        # Log fairness metrics
        disparate_impact = metric.disparate_impact()
        stat_parity_diff = metric.statistical_parity_difference()
        logging.info(f"Disparate Impact (Gender): {disparate_impact:.2f}")
        logging.info(f"Statistical Parity Difference (Gender): {stat_parity_diff:.2f}")

        # Explainability with SHAP
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(os.path.join(log_dir, 'shap_summary.png'))
        plt.close()
        logging.info("SHAP summary plot saved to C:/HR_Agent_Logs/shap_summary.png")

        return {
            'disparate_impact': disparate_impact,
            'statistical_parity_difference': stat_parity_diff
        }

    except Exception as e:
        logging.error(f"Error in fairness audit: {str(e)}")
        return {'disparate_impact': None, 'statistical_parity_difference': None}

if __name__ == "__main__":
    # Example synthetic data for standalone testing
    synthetic_data = pd.DataFrame({
        'skill_similarity': [0.8, 0.6, 0.9, 0.4, 0.7],
        'experience_match': [1.0, 0.8, 1.0, 0.5, 0.9],
        'education_match': [1, 0, 1, 0, 1],
        'gender': [0, 1, 0, 1, 0],
        'age': [25, 35, 28, 40, 30],
        'label': [9, 7, 10, 5, 8]
    })
    ranked_candidates = [{'resume': {'skills': ['Python'], 'experience': ['2 years'], 'education': ['BS'], 'gender': g, 'age': a}, 'score': s} 
                         for g, a, s in zip(synthetic_data['gender'], synthetic_data['age'], synthetic_data['label'])]
    job_data = {'required_skills': ['Python'], 'required_experience': 2, 'required_education': ['BS']}
    
    fairness_metrics = audit_fairness([], ranked_candidates, job_data)
    print(f"Disparate Impact: {fairness_metrics['disparate_impact']:.2f}")
    print(f"Statistical Parity Difference: {fairness_metrics['statistical_parity_difference']:.2f}")