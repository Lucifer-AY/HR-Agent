import unittest
from src.ranking import rank_candidates

class TestRanking(unittest.TestCase):
    def test_ranking(self):
        resumes = [{"skills": ["python"], "experience": ["5 years"], "education": ["BS"], "certifications": ["AWS"]}]
        job_data = {"required_skills": ["python"], "required_experience": 3, "required_education": ["BS"], "required_certifications": ["AWS"]}
        ranked = rank_candidates(resumes, job_data)
        self.assertEqual(len(ranked), 1)

if __name__ == '__main__':
    unittest.main()
    