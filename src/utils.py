import logging

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(filename='data/logs/app.log', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')