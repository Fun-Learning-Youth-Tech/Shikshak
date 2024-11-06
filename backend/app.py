from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import TeacherBotInference
from utils import log_interaction, validate_input
import logging

app = Flask(__name__)
CORS(app)

# Initialize the inference engine
inference_engine = TeacherBotInference()

logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        
        # Validate input
        validation_result = validate_input(data)
        if not validation_result['valid']:
            return jsonify({"error": validation_result['message']}), 400
        
        student_id = data['student_id']
        query = data['query']
        language = data.get('language', 'EN')
        
        # Get response from the model
        response = inference_engine.get_response(query, language)
        
        # Log the interaction
        log_interaction(student_id, query, response, language)
        
        return jsonify({
            "response": response,
            "language": language,
            "status": "success"
        })
        
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)