# Copyright (c) 2024 Antonio Quinonez
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import time
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class FFOpenAI:
    """
    A class to interact with OpenAI's API, specifically designed for chat-based models.

    This class provides methods to initialize an OpenAI client, manage assistants,
    and generate responses using the OpenAI API.

    Attributes:
        api_key (str)
        model (str): The name of the OpenAI model to use.
        temperature (float)
        max_tokens (int): The maximum number of tokens to generate in the response.
        system_instructions (str)
        assistant_name (str): The name of the assistant to use or create.
        assistant_id (str): The ID of the assistant being used.
        thread_id (str): The ID of the current conversation thread.
        client (OpenAI): The OpenAI client instance.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the FFOpenAI instance.

        Args:
            config (Optional[dict]): A dictionary of configuration parameters.
            **kwargs: Additional keyword arguments to override config and defaults.

        The initialization process prioritizes values in the following order:
        1. Keyword arguments
        2. Configuration dictionary
        3. Environment variables
        4. Default values
        """
        # DEFAULT VALUES
        defaults = {
            'model': "gpt-3.5-turbo",
            'max_tokens': 1000,
            'temperature': 0.5,
            'assistant_name': "default",
            'system_instructions': "Respond accurately to user queries. Be thorough but not repetitive. Be concise. Never start with a preamble. Immediately address the ask or request. Do not add meta information about your response. If there's nothing to do, answer with 'Not Applicable'."
        }

        # Combine config and kwargs, with kwargs taking precedence
        all_config = {**defaults, **(config or {}), **kwargs}

        # Set attributes based on the combined configuration
        for key, value in all_config.items():
            match key:
                case 'api_key':
                    self.api_key = value or os.getenv('OPENAI_TOKEN')
                case 'model':
                    self.model = value
                case 'temperature':
                    self.temperature = float(value)
                case 'max_tokens':
                    self.max_tokens = int(value)
                case 'system_instructions':
                    self.system_instructions = value
                case 'assistant_name':
                    self.assistant_name = value
                case 'assistant_id':
                    self.assistant_id = value
                case 'thread_id':
                    self.thread_id = value

        # Set default values if not set
        self.api_key = getattr(self, 'api_key', os.getenv('OPENAI_TOKEN'))
        self.model = getattr(self, 'model', os.getenv('OPENAI_MODEL', defaults['model']))
        self.temperature = getattr(self, 'temperature', float(os.getenv('OPENAI_TEMPERATURE', defaults['temperature'])))
        self.max_tokens = getattr(self, 'max_tokens', int(os.getenv('OPENAI_MAX_TOKENS', defaults['max_tokens'])))
        self.system_instructions = getattr(self, 'system_instructions', os.getenv('OPENAI_ASSISTANT_INSTRUCTIONS', defaults['system_instructions']))
        self.assistant_name = getattr(self, 'assistant_name', os.getenv('OPENAI_ASSISTANT_NAME', defaults['assistant_name']))
        self.assistant_id = getattr(self, 'assistant_id', None)
        self.thread_id = getattr(self, 'thread_id', None)

        # Initialize the OpenAI client and get the assistant
        self.client: OpenAI = self._initialize_client()
        self.assistant_id = self._get_assistant(self.assistant_id)

    def _initialize_client(self) -> OpenAI:
        """
        Initialize and return the OpenAI client.

        Returns:
            OpenAI: An instance of the OpenAI client.
        """
        if not self.api_key:
            raise ValueError("API key not found")
        
        return OpenAI(api_key=self.api_key)

    def _get_assistant(self, assistant_id: Optional[str]) -> str:
        """
        Retrieve an existing assistant or create a new one if it doesn't exist.

        Args:
            assistant_id (Optional[str]): The ID of the assistant to retrieve.

        Returns:
            str: The ID of the retrieved or created assistant.
        """
        if assistant_id:
            try:
                assistant = self.client.beta.assistants.retrieve(assistant_id)
                return assistant.id
            except Exception as e:
                print(f"Error retrieving assistant with ID {assistant_id}: {str(e)}")
        
        try:
            assistants = self.client.beta.assistants.list(order="desc")
            for assistant in assistants.data:
                if assistant.name == self.assistant_name:
                    return assistant.id
        except Exception as e:
            print(f"Error listing assistants: {str(e)}")
        
        return self._create_assistant(self.assistant_name)

    def _create_assistant(self, name: str) -> str:
        """
        Create a new OpenAI assistant.

        Args:
            name (str): The name of the assistant to create.

        Returns:
            str: The ID of the created assistant.
        """
        try:
            assistant = self.client.beta.assistants.create(
                name=name,
                instructions=self.system_instructions,
                model=self.model
            )
            return assistant.id
        except Exception as e:
            raise RuntimeError(f"Error creating OpenAI assistant: {str(e)}")

    def _ensure_thread(self) -> None:
        """
        Ensure a valid thread exists, creating a new one if necessary.
        """
        if self.thread_id is None:
            thread = self.client.beta.threads.create()
            self.thread_id = thread.id

    def _run_conversation(self, prompt: str) -> str:
        """
        Run a conversation in the current thread and return the response.

        Args:
            prompt (str): The user's input prompt.

        Returns:
            str: The assistant's response.
        """
        self._ensure_thread()
        
        try:
            # Add the user's message to the thread
            self.client.beta.threads.messages.create(
                thread_id=self.thread_id,
                role="user",
                content=prompt
            )

            # Create and monitor the run
            run = self.client.beta.threads.runs.create(
                thread_id=self.thread_id,
                assistant_id=self.assistant_id
            )

            # Wait for the run to complete
            while run.status in ['queued', 'in_progress']:
                time.sleep(1)
                run = self.client.beta.threads.runs.retrieve(thread_id=self.thread_id, run_id=run.id)

            if run.status != 'completed':
                raise RuntimeError(f"Run failed with status: {run.status}")

            # Retrieve the assistant's response
            messages = self.client.beta.threads.messages.list(thread_id=self.thread_id)
            return messages.data[0].content[0].text.value

        except Exception as e:
            raise RuntimeError(f"Error in OpenAI conversation: {str(e)}")

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response to the given prompt.

        Args:
            prompt (str): The user's input prompt.

        Returns:
            str: The generated response from the OpenAI model.
        """
        return self._run_conversation(prompt)
