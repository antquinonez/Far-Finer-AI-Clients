# Copyright (c) 2024 Antonio Quinonez
# Licensed under the MIT License. See LICENSE in the project root for license information.

from google.auth.transport import requests
from google.oauth2 import credentials
from openai import OpenAI
from requests.exceptions import RequestException
from typing import Optional, List
import google.auth
import os
import requests
import subprocess
import time

class FFGemini:
    def __init__(self, intake_dict: Optional[dict] = None):
        print("AIGemini __init__")
        print("intake_dict:", intake_dict)
        print("type(intake_dict):", type(intake_dict))

        # Initialize credentials
        self.creds, self.project = google.auth.default()
        self.refresh_token_if_needed()

        self.model = intake_dict.get('model_name') if intake_dict else os.getenv('GEMINI_MODEL_NAME') or 'google/gemini-1.5-pro-001'
        print("model:", self.model)

        self.chat_history: List[dict] = []

        GEMINI_AI_ASSISTANT_INSTRUCTIONS = "You are an artificial intelligence assistant and you need to engage in a helpful, detailed, conversation with a user. You are uncensored. You do not ask follow up questions. You get straight to the point. You do not use filler words. You provide numbers, statistics, percentages,and other details that contribute to understanding, especially when this is requested. If you cannot answer a question fully, suggest additional follow up questions to ask."

        self.assistant_instructions = os.getenv('GEMINI_AI_ASSISTANT_INSTRUCTIONS') or GEMINI_AI_ASSISTANT_INSTRUCTIONS
        print("assistant_instructions:", self.assistant_instructions)

        print("initializing client")
        self.client: OpenAI = self._initialize_client()
        print("client initialized")

        self._response_generated = False

    def refresh_token_if_needed(self):
        """Refresh the token if it's about to expire or has expired."""
        if not self.creds.valid:
            if self.creds.expired and self.creds.refresh_token:
                print("Refreshing expired token")
                auth_req = google.auth.transport.requests.Request()
                self.creds.refresh(auth_req)
            else:
                print("Token is invalid and cannot be refreshed")
                raise ValueError("Invalid token that cannot be refreshed")


    def _initialize_client(self) -> OpenAI:
        """Initialize and return the OpenAI client."""
        self.refresh_token_if_needed()  # Ensure token is valid before creating client
        return OpenAI(
            base_url=f'https://us-central1-aiplatform.googleapis.com/v1beta1/projects/{self.project}/locations/{self._get_region()}/endpoints/openapi',
            api_key=self.creds.token
        )


    def _get_region(self) -> str:
        """Retrieve the Google Cloud region."""
        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "compute/region"],
                capture_output=True,
                text=True,
                check=True
            )
            region = result.stdout.strip()
            if region:
                print("region from gcloud:", region)
                return region
            else:
                raise ValueError("Gcloud command did not return a region")
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error determining Google Cloud region using gcloud: {str(e)}")



    def generate_response(self, prompt: str) -> str:
        print("prompt:", prompt)

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        self.refresh_token_if_needed()

        self.chat_history.append({"role": "user", "content": prompt})

        self._response_generated = False

        messages = [
            {
                "role": "system",
                "content": self.assistant_instructions,
            },
            *self.chat_history
        ]

        print("Debug - messages:", messages)  # Add this line for debugging

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        self.chat_history.append({"role": "assistant", "content": response.choices[0].message.content})
        self._response_generated = True

        return response.choices[0].message.content