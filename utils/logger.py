import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

def log_request(data):
    logging.info(f"Received request: {data}")

def log_error(err):
    logging.error(f"Error: {err}")

def log_success(msg):
    logging.info(f"Success: {msg}")
