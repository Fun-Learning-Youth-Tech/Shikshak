import requests
import os

# Get API Key from environment variable for security
api_key = os.getenv('CEREBRAS_API_KEY', 'default_api_key_here')  # Default for development
url = "https://api.cerebras.ai/inference"
headers = {"Authorization": f"Bearer {api_key}"}

def get_inference(prompt):
    data = {"input": prompt}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}  # Return error message if request fails

# Example usage
result = get_inference("What is the capital of India?")
print(result)
