import requests

# Replace with actual endpoint and API key
url = "https://api.cerebras.ai/inference"
api_key = "your_api_key_here"
headers = {"Authorization": f"Bearer {api_key}"}

def get_inference(prompt):
    data = {"input": prompt}
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Example usage
result = get_inference("What is the capital of India?")
print(result)
