# Copyright (c) 2024 Antonio Quinonez
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import time
from typing import Optional, List
from anthropic import Anthropic as AnthropicClient
from dotenv import load_dotenv

load_dotenv()

class Anthropic:
    def __init__(self, config: Optional[dict] = None, **kwargs):
        # DEFAULT VALUES

        default_max_model = 'max-tokens-3-5-sonnet-2024-07-15'
        defaults = {
            'model': "claude-3-5-sonnet-20240620",
            'max_tokens': 2000,
            'max_model_max_tokens': 8192,
            'temperature': 0.5,
            'instructions': "Respond accurately to user queries. Never start with a preamble, such as 'The provided JSON data structure has been reordered and formatted as valid'. Immediately address the ask or request. Do not add meta information about your response. If there's nothing to do, answer with ''"
        }

        # Combine config and kwargs, with kwargs taking precedence
        all_config = {**defaults, **(config or {}), **kwargs}

        for key, value in all_config.items():
            match key:
                case 'api_key':
                    self.api_key = value or os.getenv('ANTHROPIC_TOKEN')
                case 'model':
                    self.model = value
                case 'temperature':
                    self.temperature = float(value)
                case 'max_model' | 'use_max_model':
                    if value:
                        self.max_model = all_config.get('max_model', default_max_model) or os.getenv('ANTHROPIC_MAX_MODEL', default_max_model)
                        self.max_tokens = int(all_config.get('max_model_max_tokens'))
                case 'max_tokens':
                    if not hasattr(self, 'max_tokens'):
                        self.max_tokens = int(value)
                case 'system_instructions':
                    self.system_instructions = value

        # Set default values if not set
        self.api_key = getattr(self, 'api_key', os.getenv('ANTHROPIC_TOKEN'))
        self.model = getattr(self, 'model', os.getenv('ANTHROPIC_MODEL', defaults['model']))
        self.temperature = getattr(self, 'temperature', float(os.getenv('ANTHROPIC_TEMPERATURE', defaults['temperature'])))
        self.max_tokens = getattr(self, 'max_tokens', int(os.getenv('ANTHROPIC_MAX_TOKENS', defaults['max_tokens'])))
        self.system_instructions = getattr(self, 'system_instructions', os.getenv('ANTHROPIC_ASSISTANT_INSTRUCTIONS', defaults['instructions']))
        self.max_model = getattr(self, 'max_model', None)

        self.conversation_history = []
        self.client: AnthropicClient = self._initialize_client()       
             
    def _initialize_client(self) -> AnthropicClient:
        api_key = self.api_key
        if not api_key:
            raise ValueError("API key not found")
        
        return AnthropicClient(api_key=api_key)

    def generate_response(self, prompt: str) -> str:
        try:
            self.conversation_history.append({"role": "user", "content": prompt})
            
            if self.max_model:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=self.system_instructions,
                    messages=self.conversation_history,
                    extra_headers={"anthropic-beta": self.max_model}
                )
            else:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=self.system_instructions,
                    messages=self.conversation_history
                )                
            
            assistant_response = response.content[0].text
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response
        except Exception as e:
            print("problem with response generation")
            print("  -- model: ", self.model)
            print("  -- system: ", self.system_instructions)
            # print string representation of clas object
            print("  -- exception: ", e)
            print("  -- conversation history: ", self.conversation_history)
            print("  -- max_model: ", self.max_model)
            print("  -- max_tokens: ", self.max_tokens)
            print("  -- temperature: ", self.temperature)
            # Error(f"Error generating response from Claude: {str(e)}")

            
            raise RuntimeError(f"Error generating response from Claude: {str(e)}")

    def clear_conversation(self):
        self.conversation_history = []
