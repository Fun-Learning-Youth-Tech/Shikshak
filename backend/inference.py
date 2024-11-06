import os
import logging
import requests
from typing import Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from cerebras.cloud.sdk import Cerebras  # Assuming you have this SDK available

class CerebrasInference:
    def __init__(self, api_key: str, base_url: str = "https://api.cerebras.ai/v1/models/llama3.1-8b"):
        """
        Initialize Cerebras inference engine.
        Args:
            api_key (str): csk-dm398my8393c94238598fpwddpkcfffnr3me5pfkj3tp4hpn
            base_url (str): Base URL for Cerebras API (default is v1 endpoint)
        """
        self.api_key = dm398my8393c94238598fpwddpkcfffnr3me5pfkj3tp4hpn
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        logging.basicConfig(
            filename='logs/cerebras_inference.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Initialize Cerebras client
        self.client = Cerebras(api_key=self.api_key)
        
        # List available models (optional step)
        self._list_models()

    def _list_models(self):
        """List available models in Cerebras account."""
        try:
            models = self.client.models.list()
            logging.info(f"Models available: {models}")
        except Exception as e:
            logging.error(f"Error listing models: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_inference(self, prompt: str, language: str = "EN", temperature: float = 0.7, max_tokens: int = 150) -> Dict[str, Any]:
        """
        Get inference from Cerebras API with retry logic
        """
        try:
            formatted_prompt = self._format_prompt(prompt, language)
            
            payload = {
                "prompt": formatted_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "model": "llama3.1-8b",  # Model identifier
                "stream": False
            }

            # Construct endpoint and send request
            endpoint = f"{self.base_url}/completions"
            logging.info(f"Making request to endpoint: {endpoint}")
            
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            logging.info(f"Successfully got inference for prompt: {prompt[:50]}...")
            return self._process_response(result)

        except requests.exceptions.RequestException as e:
            logging.error(f"Cerebras API request failed: {str(e)}")
            raise

    def _format_prompt(self, prompt: str, language: str) -> str:
        """Format the prompt with language and educational context"""
        language_tokens = {
            "EN": "Answer in English:",
            "ES": "Responde en español:",
            "FR": "Répondez en français:",
            "DE": "Antworten Sie auf Deutsch:",
            "ZH": "用中文回答："
        }
        
        language_prefix = language_tokens.get(language, language_tokens["EN"])
        educational_context = "As an educational AI assistant, provide a clear and detailed explanation."
        
        return f"{educational_context}\n{language_prefix}\n{prompt}"

    def _process_response(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """Process and format the API response"""
        try:
            return {
                "response": api_response.get("choices", [{}])[0].get("text", ""),
                "usage": api_response.get("usage", {}),
                "model": api_response.get("model", ""),
                "status": "success"
            }
        except Exception as e:
            logging.error(f"Error processing API response: {str(e)}")
            return {
                "response": "Sorry, I couldn't process the response properly.",
                "status": "error",
                "error": str(e)
            }
