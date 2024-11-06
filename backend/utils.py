import json
from datetime import datetime

def log_interaction(student_id, query, response):
    log_data = {
        "student_id": student_id,
        "query": query,
        "response": response,
        "timestamp": datetime.now().isoformat()
    }
    try:
        with open("interaction_log.json", "a") as log_file:
            json.dump(log_data, log_file)
            log_file.write("\n")
    except Exception as e:
        print(f"Error writing to log file: {e}")

def run_cli():
    print("Welcome to the Educational Assistant!")
    student_id = input("Enter student ID: ")
    while True:
        query = input("Ask your question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        response = get_inference(query)  # Call your model or RAG pipeline here
        print(f"Response: {response}")
        log_interaction(student_id, query, response)

run_cli()
