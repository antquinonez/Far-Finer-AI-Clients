# Copyright (c) 2024 Antonio Quinonez
# Licensed under the MIT License. See LICENSE in the project root for license information.

from lib.AI.FFOpenAI import FFOpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting OpenAI command-line interface")

    config = {
        'max_tokens': 1000,  # or any other value
        'temperature': 0.7,  # or any other value
        'assistant_name': "Borges",
    }

    try:
        # Create an instance of the FFOpenAI class
        ai = FFOpenAI(config)
        logger.info("FFOpenAI initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize FFOpenAI: %s", str(e))
        return

    logger.info("AI initialized. Type 'exit' to quit.")

    while True:
        # Get user input
        user_input = input("You: ")

        # Check if the user wants to exit
        if user_input.lower() == 'exit':
            logger.info("User requested to exit")
            print("Goodbye!")
            break

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