# Far-Finer-AI-Clients
Python libraries to simplify working with artificial intelligence APIs such as OpenAI's Assistant API and Anthropic's API with caching support & max models. Includes chat history management and very robust config handling.

## Features
(for Anthropic)
* Caching of system instructions 
* Handling of Max models

For all:
* Chat history management
* Consistent config file handling

## Status
This is very alpha software. Use at your own risk. Watch those tokens!

## Installation
Copy the files in the lib directory to your own lib directory.

## Configuration
Create a .env file in the root directory with the following variables -- DO NOT ADD .env TO GIT!!!:

Add the following variables to the .env file: Assign your token values to the following variables:
* OPENAI_TOKEN
* ANTHROPIC_TOKEN
* PERPLEXITY_TOKEN

Additional config variables to simplify config handling (If you put these in the .env file, don't worry about adding them as config param):
* ANTHROPIC_MODEL
* ANTHROPIC_MAX_MODEL
* ANTHROPIC_MAX_TOKENS
* ANTHROPIC_TEMPERATURE
* ANTHROPIC_ASSISTANT_INSTRUCTIONS

I use defaults in the code to make it easier to use in case you forget to set them. *Watch your tokens*

## Try it out
* Run the appropriate try_ scripts in the root directory and interact with the AI.
* Pass a 'config' dict as a parameter to the AI class to override the defaults.

Some config options:
* api_key
* model
* temperature
* max_tokens

Example:
```python
config = {
    "api_key": "YOUR_API_KEY",
    "model": "gpt-3.5-turbo",
    "temperature": 0.5,
    "max_tokens": 100
}

ai = Anthropic(config, use_max_model=True)
## AIs Implemented

* Anthropic -- Separate ones for the
- Basic & Max models -- Anthropic.py
- Caching of system instructions -- AnthropicCached.py

### Coming soon:
* OpenAI -- Assistant API
* Perplexity
* Gemini
* Dictionaries to hold prompts and responses.
