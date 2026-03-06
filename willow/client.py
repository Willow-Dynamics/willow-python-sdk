import base64
import json
import requests
from io import BytesIO
from typing import Union

from .types import WillowModel
from .parsers import parse_int8_model, parse_json_model

class WillowClient:
    """
    The official Willow API Client for Model Provisioning.
    Handles secure retrieval of action recognition models for local or edge execution.
    """
    
    # Defaults to the Master AWS Oracle Gateway
    DEFAULT_API_URL = "https://55zydxbe05.execute-api.us-east-2.amazonaws.com"
    
    def __init__(self, api_key: str, base_url: str = DEFAULT_API_URL):
        """
        Initializes the client.
        :param api_key: Your Willow API Key.
        :param base_url: The base URL of the Willow API.
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }

    def _fetch_model_payload(self, model_id: str, format_type: str) -> Union[bytes, dict]:
        """Internal method to handle the HTTP request and Base64 decoding."""
        url = f"{self.base_url}/export"
        
        # Legacy architecture routing abstraction:
        # 'shopifyCustomerId' acts as the API identifier credential for the Oracle backend.
        params = {
            "analysisId": model_id,
            "shopifyCustomerId": self.api_key, 
            "format": format_type
        }
        
        response = requests.get(url, params=params, headers=self.headers)
        
        if response.status_code != 200:
            raise ConnectionError(f"Failed to fetch model {model_id}: {response.status_code} - {response.text}")

        if format_type == "int8":
            # The Willow Backend returns the binary payload as a Base64 string
            return base64.b64decode(response.text)
        else:
            # Standard JSON payload
            return response.json()

    def get_model(self, model_id: str) -> WillowModel:
        """
        ON-DEMAND EPHEMERAL RAM: 
        Fetches the highly-optimized .int8 model directly into an ephemeral memory buffer.
        The model is parsed and ready for inference, never touching the physical disk.
        """
        raw_bytes = self._fetch_model_payload(model_id, format_type="int8")
        memory_buffer = BytesIO(raw_bytes)
        return parse_int8_model(memory_buffer)

    def download_model(self, model_id: str, dest_path: str, format_type: str = "int8") -> str:
        """
        STORE LOCALLY: 
        Fetches the model via the API and writes it to the local disk for offline usage.
        :param dest_path: The file path to save the model to (e.g., './models/reload.int8').
        :param format_type: 'int8' (recommended) or 'json'.
        """
        data = self._fetch_model_payload(model_id, format_type)
        
        mode = 'wb' if format_type == "int8" else 'w'
        with open(dest_path, mode) as f:
            if format_type == "int8":
                f.write(data)
            else:
                json.dump(data, f, indent=2)
                
        return dest_path