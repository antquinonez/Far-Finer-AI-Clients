# Copyright (c) 2024 Antonio Quinonez
# Licensed under the MIT License. See LICENSE in the project root for license information.

from lib.AI.FFGemini import FFGemini
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Gemini command-line interface")
    
    try:
        # Create an instance of the AI class
        ai = FFGemini()
        logger.info("FFGemini initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize FFGemini: %s", str(e))
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
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()