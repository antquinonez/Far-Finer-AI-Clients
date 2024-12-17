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
        self._defaults = {
            'model': "gpt-4o",
            'infer_o1': False,
            'is_o1': False,
            'max_tokens': 4000,
            'max_completion_tokens': 8000,
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
                case 'infer_o1':
                    self.infer_o1 = bool(value)
                case 'is_o1':
                    self.is_o1 = bool(value)
                case 'temperature':
                    self.temperature = float(value)
                case 'max_tokens':
                    self.max_tokens = int(value)
                case 'max_completion_tokens':
                    self.max_completion_tokens = int(value)
                case 'system_instructions':
                    self.system_instructions = value

        # Set default values if not set
        self.api_key = getattr(self, 'api_key', os.getenv('AZUREOPENAI_TOKEN'))
        self.model = getattr(self, 'model', os.getenv('AZUREOPENAI_MODEL',  self._defaults['model']))
        self.is_o1 = getattr(self, 'is_o1', self._defaults['is_o1'])
        self.infer_o1 = getattr(self, 'infer_o1',  self._defaults['infer_o1'])
        self.temperature = getattr(self, 'temperature', float(os.getenv('AZUREOPENAI_TEMPERATURE',  self._defaults['temperature'])))

        # infer if o1
        if self.infer_o1 == True and self.is_o1 is None:
            if 'o1' in self.model:
                self.is_o1 = True
            else:
                self.is_o1 = False

        # use appropriate token param depending on if o1 or not ================================================================
        if self.is_o1:
            self.max_completion_tokens = getattr(self, 'max_completion_tokens', int(os.getenv('AZUREOPENAI_MAX_COMPLETION_TOKENS', self._defaults['max_completion_tokens'])))
            logger.debug(f"Using o1 type model: {self.model}")
            # temperature not used by o1
            logger.debug(f"Using max_completion_tokens: {self.max_completion_tokens}")
        else:
            self.max_tokens = getattr(self, 'max_tokens', int(os.getenv('AZUREOPENAI_MAX_TOKENS', self._defaults['max_tokens'])))
            
            logger.debug(f"Using model: {self.model}")
            logger.debug(f"Temperature: {self.temperature}") 
            logger.debug(f"Using max_tokens: {self.max_tokens}")
        # ====================================================================================================================

        self.system_instructions = getattr(self, 'system_instructions', os.getenv('AZUREOPENAI_SYSTEM_INSTRUCTIONS', self._defaults['instructions']))

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


    from inspect import signature

    def generate_response(self, prompt: str, model: Optional[str] = None, is_o1: Optional[bool] = None, infer_o1:Optional[bool] = None, prompt_name: Optional[str] = None) -> str:
        logger.debug(f"Generating response for prompt: {prompt}")
        logger.debug("Method args")
        logger.debug(locals())

        method_is_o1 = is_o1
        # are we using the model and is_o1 from init or the one passed with the generate_response method?
        used_model = model if model else self.model

        logger.debug(f"Using model: {used_model}")

        # infer if o1 for method call with model arg
        if model and infer_o1 == True:
            logger.debug(f"Using infer_o1 from method call")
            if 'o1' in model:
                is_o1 = True
            else:
                is_o1 = False

        elif model and infer_o1 == False and is_o1 == True:
            logger.debug(f"Using is_o1 == True from method call") 
        elif model and is_o1 == True:
            logger.debug(f"Method says model is an o1 model")
        elif model and is_o1 == False:
            logger.debug(f"Method says model is not an o1 model")
        elif model == self.model:
            is_o1 = self.is_o1
            logger.debug(f"Using is_o1 from self.is_o1 since models are the same from init and method call:")
            logger.debug(f"Method call arg is_o1: {method_is_o1}| Init is_o1: {self.is_o1}")
            logger.debug(f"Method model: {model}")
        elif model != self.model:
            is_o1 = False
            logger.debug(f"Method model not init model. Using is_o1 == False")
            # max_tokens = self._defaults['max_tokens']
        else:
            is_o1 = self.is_o1
            logger.debug(f"Other condition. Using is_o1 from self.is_o1: {self.is_o1}")

        try:
            self.conversation_history.append({"role": "user", "content": prompt})
            
            messages = [
                {
                    "role": "assistant" if self.is_o1 == True else "system",
                    "content": self.system_instructions,
                },
                *self.conversation_history
            ]

            # DIFFERENT PROMPT COMPLETIONS DEPENDING ON IF o1 OR NOT
            if is_o1 == True:
                response = self.client.chat.completions.create(
                    model=used_model,
                    messages=messages,
                    max_completion_tokens = getattr(self, 'max_completion_tokens', self._defaults['max_completion_tokens'])
                )
            else:
                response = self.client.chat.completions.create(
                    model=used_model,
                    messages=messages,
                    max_tokens= getattr(self, 'max_tokens', self._defaults['max_tokens']),
                    temperature=self.temperature
                )
            # ================================================================

            
            assistant_response = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            logger.info("Response generated successfully")
            return assistant_response
        except Exception as e:
            logger.error("Problem with response generation")
            logger.error(f"  -- exception: {str(e)}")
            logger.error(f"  -- model: {used_model}")
            logger.error(f"  -- system: {self.system_instructions}")
            logger.error(f"  -- conversation history: {self.conversation_history}")
            
            raise RuntimeError(f"Error generating response from Azure OpenAI: {str(e)}")

    def clear_conversation(self):
        logger.info("Clearing conversation history")
        self.conversation_history = []