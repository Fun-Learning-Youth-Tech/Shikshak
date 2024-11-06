import requests
import os

# Get API Key from environment variable for security
api_key = os.getenv('csk-dm398my8393c94238598fpwddpkcfffnr3me5pfkj3tp4hpn', 'csk-255c4k9cppdxcfck6evt6pphnv4k38k9fm8c92tjv5fnjcd8')  # Default for development
url = "https://api.cerebras.ai/v1/models/llama3.1-8b"
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
