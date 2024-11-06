import json
from datetime import datetime
import logging
from typing import Dict, Any
import os

def setup_logging():
    """Set up logging configuration"""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename='logs/utils.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def validate_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate input data"""
    required_fields = ['student_id', 'query']
    
    for field in required_fields:
        if field not in data:
            return {
                'valid': False,
                'message': f"Missing required field: {field}"
            }
        
        if not data[field]:
            return {
                'valid': False,
                'message': f"Empty value for required field: {field}"
            }
    
    return {'valid': True, 'message': "Input validation successful"}

def log_interaction(student_id: str, query: str, response: str, language: str):
    """Log interaction with timestamp and metadata"""
    try:
        log_data = {
            "student_id": student_id,
            "query": query,
            "response": response,
            "language": language,
            "timestamp": datetime.now().isoformat(),
            "session_id": generate_session_id(student_id)
        }
        
        log_file_path = f"logs/interactions_{datetime.now().strftime('%Y%m%d')}.json"
        
        with open(log_file_path, "a") as log_file:
            json.dump(log_data, log_file)
            log_file.write("\n")
            
        logging.info(f"Interaction logged for student {student_id}")
        
    except Exception as e:
        logging.error(f"Failed to log interaction: {str(e)}")

def generate_session_id(student_id: str) -> str:
    """Generate a unique session ID"""
    return f"{student_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

# Initialize logging
setup_logging()