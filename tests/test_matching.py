import unittest
from src.matching import compute_similarity

class TestMatching(unittest.TestCase):
    def test_similarity(self):
        resume_data = {"skills": ["python"], "experience": ["5 years"]}
        job_data = {"required_skills": ["python"], "required_experience": 3}
        score = compute_similarity(resume_data, job_data)
        self.assertGreater(score, 0)

if __name__ == '__main__':
    unittest.main()