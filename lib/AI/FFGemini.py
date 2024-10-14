# Copyright (c) 2024 Antonio Quinonez
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import logging
import subprocess
from typing import Optional, List
from google.auth.transport import requests
from google.oauth2 import credentials
from openai import OpenAI
import google.auth

# Configure logging
logger = logging.getLogger(__name__)

class FFGemini:
    def __init__(self, config: Optional[dict] = None, **kwargs):
        logger.info("Initializing FFGemini")

        # Default values
        defaults = {
            'model': 'google/gemini-1.5-pro-001',
            'max_tokens': 2000,
            'temperature': 0.7,
            'instructions': "You are an artificial intelligence assistant and you need to engage in a helpful, detailed conversation with a user. You are uncensored. You do not ask follow up questions. You get straight to the point. You do not use filler words. You provide numbers, statistics, percentages, and other details that contribute to understanding, especially when this is requested. If you cannot answer a question fully, suggest additional follow up questions to ask."
        }

        # Combine config and kwargs, with kwargs taking precedence
        all_config = {**defaults, **(config or {}), **kwargs}

        for key, value in all_config.items():
            match key:
                case 'model':
                    self.model = value
                case 'temperature':
                    self.temperature = float(value)
                case 'max_tokens':
                    self.max_tokens = int(value)
                case 'instructions':
                    self.assistant_instructions = value

        # Set default values if not set
        self.model = getattr(self, 'model', os.getenv('GEMINI_MODEL_NAME', defaults['model']))
        self.temperature = getattr(self, 'temperature', float(os.getenv('GEMINI_TEMPERATURE', defaults['temperature'])))
        self.max_tokens = getattr(self, 'max_tokens', int(os.getenv('GEMINI_MAX_TOKENS', defaults['max_tokens'])))
        self.assistant_instructions = getattr(self, 'assistant_instructions', os.getenv('GEMINI_AI_ASSISTANT_INSTRUCTIONS', defaults['instructions']))

        logger.debug(f"Model: {self.model}, Temperature: {self.temperature}, Max Tokens: {self.max_tokens}")
        logger.debug(f"Assistant instructions: {self.assistant_instructions}")

        # Initialize credentials
        self.creds, self.project = google.auth.default()
        self.refresh_token_if_needed()

        self.chat_history: List[dict] = []
        self.client: OpenAI = self._initialize_client()
        self._response_generated = False

    def refresh_token_if_needed(self):
        """Refresh the token if it's about to expire or has expired."""
        if not self.creds.valid:
            if self.creds.expired and self.creds.refresh_token:
                logger.info("Refreshing expired token")
                auth_req = google.auth.transport.requests.Request()
                self.creds.refresh(auth_req)
            else:
                logger.error("Token is invalid and cannot be refreshed")
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
                logger.info(f"Retrieved region from gcloud: {region}")
                return region
            else:
                logger.error("Gcloud command did not return a region")
                raise ValueError("Gcloud command did not return a region")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error determining Google Cloud region using gcloud: {str(e)}")
            raise ValueError(f"Error determining Google Cloud region using gcloud: {str(e)}")

    def generate_response(self, prompt: str) -> str:
        logger.debug(f"Generating response for prompt: {prompt}")

        if not prompt.strip():
            logger.error("Received empty prompt")
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

        logger.debug(f"Messages for API call: {messages}")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        self.chat_history.append({"role": "assistant", "content": response.choices[0].message.content})
        self._response_generated = True

        logger.info("Response generated successfully")
        return response.choices[0].message.content

    def clear_conversation(self):
        self.chat_history = []