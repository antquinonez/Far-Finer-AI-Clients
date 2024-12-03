# Copyright (c) 2024 Antonio Quinonez
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import time
import logging
from typing import Optional
# from openai import OpenAI
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class FFAzureOpenAI:
    def __init__(self, config: Optional[dict] = None, **kwargs):
        logger.info("Initializing AzureOpenAI")

        # DEFAULT VALUES
        defaults = {
            'model': "gpt-4o",
            'max_tokens': 4000,
            'temperature': 0.5,
            'instructions': "Respond accurately to user queries. Never start with a preamble. Immediately address the ask or request. Do not add meta information about your response. If there's nothing to do, answer with ''"
        }

        # Combine config and kwargs, with kwargs taking precedence
        all_config = {**(config or {}), **kwargs}

        for key, value in all_config.items():
            match key:
                case 'api_key':
                    self.api_key = value or os.getenv('AZUREOPENAI_TOKEN')
                case 'model':
                    self.model = value
                case 'temperature':
                    self.temperature = float(value)
                case 'max_tokens':
                    self.max_tokens = int(value)
                case 'system_instructions':
                    self.system_instructions = value

        # Set default values if not set
        self.api_key = getattr(self, 'api_key', os.getenv('AZUREOPENAI_TOKEN'))
        self.model = getattr(self, 'model', os.getenv('AZUREOPENAI_MODEL', defaults['model']))
        self.temperature = getattr(self, 'temperature', float(os.getenv('AZUREOPENAI_TEMPERATURE', defaults['temperature'])))
        self.max_tokens = getattr(self, 'max_tokens', int(os.getenv('AZUREOPENAI_MAX_TOKENS', defaults['max_tokens'])))
        self.system_instructions = getattr(self, 'system_instructions', os.getenv('AZUREOPENAI_ASSISTANT_INSTRUCTIONS', defaults['instructions']))

        logger.debug(f"Model: {self.model}, Temperature: {self.temperature}, Max Tokens: {self.max_tokens}")
        logger.debug(f"System instructions: {self.system_instructions}")

        self.conversation_history = []
        self.client: AzureOpenAI = self._initialize_client()

    def _initialize_client(self) -> AzureOpenAI:
        """Initialize and return the OpenAI client."""
        logger.info("Initializing Azure OpenAI client")
        api_key = self.api_key
        if not api_key:
            logger.error("API key not found")
            raise ValueError("API key not found")
        
        azure_endpoint =  os.getenv('AZUREOPENAI_BASE')
        api_version = os.getenv('AZURE_API_VERSION') or '2024-08-01-preview'
        return AzureOpenAI( api_key=api_key, 
                            azure_endpoint=azure_endpoint,
                            api_version = api_version
        )

    def generate_response(self, prompt: str) -> str:
        logger.debug(f"Generating response for prompt: {prompt}")

        try:
            self.conversation_history.append({"role": "user", "content": prompt})
            
            messages = [
                {
                    "role": "system",
                    "content": self.system_instructions,
                },
                *self.conversation_history
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            assistant_response = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            logger.info("Response generated successfully")
            return assistant_response
        except Exception as e:
            logger.error("Problem with response generation")
            logger.error(f"  -- exception: {str(e)}")
            logger.error(f"  -- model: {self.model}")
            logger.error(f"  -- system: {self.system_instructions}")
            logger.error(f"  -- conversation history: {self.conversation_history}")
            logger.error(f"  -- max_tokens: {self.max_tokens}")
            logger.error(f"  -- temperature: {self.temperature}")
            
            raise RuntimeError(f"Error generating response from Perplexity: {str(e)}")

    def clear_conversation(self):
        logger.info("Clearing conversation history")
        self.conversation_history = []