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
    Handles secure retrieval of action recognition models via the Willow AWS Gateway.
    
    This client is strictly for fetching trained models. It does not handle
    video uploads or cloud processing pipelines.
    """
    
    def __init__(self, api_url: str, api_key: str, customer_id: str):
        """
        Initializes the client.
        
        :param api_url: The endpoint URL for the Willow API Gateway.
        :param api_key: Your secure Gateway Access Token (x-api-key or Bearer token).
        :param customer_id: Your unique Partner/Tenant ID (scopes data access).
        """
        if not api_url or not api_key or not customer_id:
            raise ValueError("WillowClient requires api_url, api_key, AND customer_id.")
            
        self.api_url = api_url.rstrip('/')
        self.customer_id = customer_id
        
        # Standardize headers for the new API Gateway
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "x-api-key": api_key,  # Redundant fallback for AWS Gateway compatibility
            "Accept": "application/json"
        }

    def _fetch_model_payload(self, model_id: str, format_type: str) -> Union[bytes, dict]:
        """
        Internal method to execute the network request.
        Handles status codes and format decoding.
        """
        url = f"{self.api_url}/export"
        
        # New Gateway Contract: Uses 'customerId' explicitly
        params = {
            "analysisId": model_id,
            "customerId": self.customer_id, 
            "format": format_type
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            
            if response.status_code == 401 or response.status_code == 403:
                raise ConnectionError(f"Authentication Failed (401/403). Check your API Key and Customer ID.")
            
            if response.status_code != 200:
                raise ConnectionError(f"Failed to fetch model {model_id}: {response.status_code} - {response.text}")

            if format_type == "int8":
                # The backend returns binary payload as Base64 string for safe transport
                return base64.b64decode(response.text)
            else:
                # Standard JSON payload
                return response.json()
                
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Network error connecting to Willow API: {e}")

    def get_model(self, model_id: str) -> WillowModel:
        """
        ON-DEMAND EPHEMERAL RAM (DRM): 
        Fetches the highly-optimized .int8 model directly into an ephemeral memory buffer.
        The model is parsed and ready for inference, never touching the physical disk.
        
        Use this for secure cloud pipelines or closed-source edge deployments.
        """
        raw_bytes = self._fetch_model_payload(model_id, format_type="int8")
        memory_buffer = BytesIO(raw_bytes)
        return parse_int8_model(memory_buffer)

    def download_model(self, model_id: str, dest_path: str, format_type: str = "int8") -> str:
        """
        STORE LOCALLY: 
        Fetches the model via the API and writes it to the local disk.
        
        Use this for offline edge devices or air-gapped environments.
        :param dest_path: File path (e.g., './models/reload.int8')
        """
        data = self._fetch_model_payload(model_id, format_type)
        
        mode = 'wb' if format_type == "int8" else 'w'
        with open(dest_path, mode) as f:
            if format_type == "int8":
                f.write(data)
            else:
                json.dump(data, f, indent=2)
                
        return dest_path