# Copyright (c) 2024 Antonio Quinonez
# Licensed under the MIT License. See LICENSE in the project root for license information.

from lib.Anthropic import Anthropic

def main():

    config = {
        'max_tokens': 2000,  # or any other value
        'temperature': 0.7,  # or any other value
    }


    # Create an instance of the ChatGPT class
    ai = Anthropic(config, use_max_model=True)

    print("AI initialized. Type 'exit' to quit.")

    while True:
        # Get user input
        user_input = input("You: ")

        # Check if the user wants to exit
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        try:
            # Generate a response
            response = ai.generate_response(user_input)
            print("Assistant:", response)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
    