# Copyright (c) 2024 Antonio Quinonez
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import time
from typing import Optional, List
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

class FFAnthropicCached:
    def __init__(self, config: Optional[dict] = None, **kwargs):
        # DEFAULT VALUES
        default_model = "claude-3-5-sonnet-20240620"
        default_max_tokens = 2000       
        default_temperature = 0.5
        
        default_instructions = "Respond accurately to user queries. Never start with a preamble, such as 'The provided JSON data structure has been reordered and formatted as valid'. Immediately address the ask or request. Do not add meta information about your response. If there's nothing to do, answer with ''"

        # Combine config and kwargs, with kwargs taking precedence
        all_config = {**(config or {}), **kwargs}

        # GET API KEY
        self.api_key = all_config.get('api_key', os.getenv('ANTHROPIC_TOKEN')) if all_config else os.getenv('ANTHROPIC_TOKEN')

        # SET MODEL
        self.model = all_config.get('model', default_model) if all_config else os.getenv('ANTHROPIC_MODEL', default_model)
        
        #SET MAX TOKENS
        self.max_tokens = int(all_config.get('max_tokens', default_max_tokens)) if all_config else int(os.getenv('ANTHROPIC_MAX_TOKENS', default_max_tokens))

        # SET TEMPERATURE
        self.temperature = float(all_config.get('temperature', default_temperature)) if all_config else float(os.getenv('ANTHROPIC_TEMPERATURE', default_temperature))

        self.system_instructions = config.get('system_instructions', default_instructions) if config else os.getenv('ANTHROPIC_ASSISTANT_INSTRUCTIONS', default_instructions)
        self.conversation_history = ConversationHistory()
             
        self.client: Anthropic = self._initialize_client()
    
    def _initialize_client(self) -> Anthropic:
        api_key = self.api_key
        if not api_key:
            raise ValueError("API key not found")
        return Anthropic(api_key=api_key)

    def generate_response(self, prompt: str) -> str:
        try: 
            self.conversation_history.add_turn_user(prompt)

            turns = self.conversation_history.get_turns()
            if not turns:
                raise ValueError("Conversation history is empty")

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=[{
                    "type": "text",
                    "text": self.system_instructions,
                    "cache_control": {"type": "ephemeral"}
                }],
                messages=turns,
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
            )

            assistant_response = response.content[0].text
            self.conversation_history.add_turn_assistant(assistant_response)
            
            return assistant_response
        except Exception as e:
            print("problem with response generation")
            print("  -- model: ", self.model)
            print("  -- system: ", self.system_instructions)
            raise RuntimeError(f"Error generating response from Claude: {str(e)}")

    def clear_conversation(self):
        self.conversation_history = ConversationHistory()

class ConversationHistory:
    def __init__(self):
        self.turns = []

    def add_turn_assistant(self, content):
        self.turns.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": content
                }
            ]
        })

    def add_turn_user(self, content):
        if self.turns and self.turns[-1]["role"] == "user":
            # If the last turn was a user, update its content instead of adding a new turn
            self.turns[-1]["content"][0]["text"] += "\n" + content
        else:
            self.turns.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": content
                    }
                ]
            })

    def get_turns(self):
        result = []
        for turn in self.turns[-100:]:  # Get the last 100 turns
            if turn["role"] == "user":
                result.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": turn["content"][0]["text"]
                        }
                    ]
                })
            else:
                result.append(turn)
        return result