# Copyright (c) 2024 Antonio Quinonez
# Licensed under the MIT License. See LICENSE in the project root for license information.

from lib.AI.FFOpenAIAssistant import FFOpenAIAssistant
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResponseFormat(Enum):
    JSON_OBJECT = {"type": "json_object"}
    TEXT = {"type": "text"}
    AUTO = 'auto'

def main():
    logger.info("Starting OpenAI command-line interface")

    config = {
        'max_tokens': 1000,
        'temperature': 0.7,
        'assistant_name': 'test_1',
        # 'response_format': ResponseFormat.TEXT.value
        # 'response_format': ResponseFormat.JSON_OBJECT.value
        # 'response_format': ResponseFormat.AUTO.value
        'response_format': 'auto'
    }

    try:
        # Create an instance of the FFOpenAI class
        ai = FFOpenAIAssistant(config)
        logger.info("FFOpenAI initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize FFOpenAI: %s", str(e))
        return

    logger.info("AI initialized. Type 'exit' to quit.")

    # We'll inject this into the prompt to get a JSON response when we need it
    reply_in_json = "FORMAT: Reply in JSON. Do not speak to the fact that you're responding in JSON. But do respond to the prompt otherwise."

    while True:
        # Get user input
        user_input = input("You: ")

        # Check if the user wants to exit
        if user_input.lower() == 'exit':
            logger.info("User requested to exit")
            print("Goodbye!")
            break

        # Prepare the prompt based on the response format
        if config['response_format'] == ResponseFormat.JSON_OBJECT.value:
            full_prompt = f'{user_input} {reply_in_json}'
            logger.info("Requesting JSON response")
        else:
            full_prompt = user_input
            logger.info(f"Using {config['response_format']} response format")

        try:
            # Generate a response
            logger.debug(f"Generating response for user input: {full_prompt}")
            response = ai.generate_response(full_prompt)
            print("Assistant:", response)
            logger.info("Response generated and displayed to user")
        except Exception as e:
            logger.error(f"An error occurred while generating response: {str(e)}")

if __name__ == "__main__":
    main()