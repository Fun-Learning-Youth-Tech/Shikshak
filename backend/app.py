from flask import Flask, request, jsonify
from inference import get_inference
from utils import log_interaction

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    student_id = data.get('student_id')
    query = data.get('query')
    
    if not student_id or not query:
        return jsonify({"error": "Missing student_id or query"}), 400
    
    # Get inference from the model (could be local or Cerebras API)
    response = get_inference(query)

    # Log the interaction
    log_interaction(student_id, query, response)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
