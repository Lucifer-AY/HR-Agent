import unittest
from src.parsing import parse_resume

class TestParsing(unittest.TestCase):
    def test_parse_pdf(self):
        resume_data = parse_resume("data/resumes/sample.pdf")
        self.assertIn("skills", resume_data)

if __name__ == '__main__':
    unittest.main()