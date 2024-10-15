# Copyright (c) 2024 Antonio Quinonez
# Licensed under the MIT License. See LICENSE in the project root for license information.

from lib.AI.FFOpenAIAssistant import FFOpenAIAssistant
import logging
from enum import Enum
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Message(BaseModel):
    user_question: str
    ai_response: str
    confidence: float

class Messages(BaseModel):
    messages: list[Message]

def main():
    logger.info("Starting OpenAI command-line interface")

    # Define the JSON schema for the response

    config = {
        'max_tokens': 1000,
        'temperature': 0.7,
        'assistant_name': 'test_cd',
        'response_format':{
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "schema": Messages.model_json_schema()
            }
        },
        'system_instructions': "Always respond in JSON format according to the provided schema."
    }

    try:
        # Create an instance of the FFOpenAI class
        ai = FFOpenAIAssistant(config)
        logger.info("FFOpenAI initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize FFOpenAI: %s", str(e))
        return

    logger.info("AI initialized. Type 'exit' to quit.")

    # Go ahead and ask two questions at a time; for example, Why is the skly blue? why are apples sweet? 
    while True:
        # Get user input
        user_input = input("You: ")

        # Check if the user wants to exit
        if user_input.lower() == 'exit':
            logger.info("User requested to exit")
            print("Goodbye!")
            break

        # Append a JSON request to the user input
        user_input += " Please respond in JSON format."

        try:
            # Generate a response
            logger.debug("Generating response for user input: %s", user_input)
            response = ai.generate_response(user_input)
            print("Assistant:", response)
            logger.info("Response generated and displayed to user")
        except Exception as e:
            logger.error("An error occurred while generating response: %s", str(e))

if __name__ == "__main__":
    main()