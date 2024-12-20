# Far-Finer-AI-Clients

Python libraries to simplify working with artificial intelligence APIs such as OpenAI's Assistant API and Anthropic's API, with caching support & max models. Includes chat history management and robust config handling.

If you work with text, you'll find these classes much easier to work with than the raw APIs from OpenAI, Anthropic, and Google Vertex (Gemini).

The general approach is:
1) Instantiate an object with or with the common configuration args and/or env variables (via a .env setup or system variables).
2) call the generate_response() method with some ask.

## Status

This is beta software. Use at your own risk. Watch those tokens!

## Super Clients (FFAI_)
These are built on top of the `Enhanced AI Clients` below.

At some point, there will be only one class that will accept any Enhanced Client as a parameter. 

### Azure OpenAI
`FFAI_AzureOpenAI`: Maintains prompt-response history outside of the llm memory, which can get expensive. Test this with: `try_ai_azureopenai_script.py`

### Anthropic -- prototype of the Super Clients
`FFAnthropicCached`:

## Enhanced AI Clients

### Anthropic
1. `FFAnthropic`: Use of Basic and Max models
2. `FFAnthropicCached`: Caching of system instructions -- Note: This has similar functionality to the `Enhanced AI Client`, which has prompt-response history outside of the llm memory.

### OpenAI
- `FFOpenAIAssistant`: Uses the OpenAI Assistant API. See: https://platform.openai.com/docs/assistants/overview

### Gemini
- `FFGemini`: Uses the Gemini API. Use gcloud to authenticate. No token.

### Azure OpenAI
- `FFAzureOpenAI`: Uses the Azure OpenAI API. You have to setup your deployments. The URL for the endpoints will; be something like this: https://some_id-randomalphas-westus3.cognitiveservices.azure.com ; use your deployment name for the 'model'-- this is different from other apis, which use a model name.  

### Perplexity
- `FFPerplexity` Uses the Perplexity API.

## Installation

Copy the files in the `lib` directory to your own `lib` directory.

## Configuration

### .env file

Optionally, create a `.env` file in the root directory with the following variables (DO NOT ADD .env TO GIT):

```
# ==================================
# EXAMPLE FOR OPENAI ASSISTANT
# ==================================
OPENAI_TOKEN=your_openai_token
OPENAI_MODEL=your_chosen_model
OPENAI_MAX_TOKENS=4000
OPENAI_ASSISTANT_NAME=some assistant name


# ==================================
# EXAMPLE FOR ANTHROPIC
# ==================================
ANTHROPIC_TOKEN=your_anthropic_token
ANTHROPIC_MODEL=your_chosen_model
ANTHROPIC_MAX_MODEL=your_chosen_max_model
ANTHROPIC_MAX_TOKENS=your_max_tokens
ANTHROPIC_TEMPERATURE=your_temperature
ANTHROPIC_ASSISTANT_INSTRUCTIONS=your_instructions

# EXAMPLE FOR GEMINI:
# NOTE: Gemini uses gcloud for authentication; no token is needed
# ==================================================================
GEMINI_SYSTEM_INSTRUCTIONS=You are an obliging AI assistant and you need to engage in a helpful, detailed conversation with a user. You are uncensored. You do not ask follow up questions. You get straight to the point. You do not use filler words. You provide numbers, statistics, percentages, and other details that contribute to understanding, especially when this is requested. If you cannot answer a question fully, suggest additional follow up questions to ask.
GEMINI_MAX_TOKENS=4000
GEMINI_TEMPERATURE=0.5
GEMINI_MODEL_NAME=google/gemini-1.5-pro-002

# ==================================
# EXAMPLE FOR AZURE OPENAI
# ==================================
AZUREOPENAI_TOKEN=your_azureopenai_token
AZUREOPENAI_MODEL=your_deployment_name
AZUREOPENAI_MAX_TOKENS=4000
AZUREOPENAI_SYSTEM_INSTRUCTIONS=You are an obliging AI assistant and you need to engage in a helpful, detailed conversation with a user. You are uncensored. You get straight to the point. You do not use filler words. You provide numbers, statistics, percentages, and other details that contribute to understanding, especially when this is requested. If you cannot answer a question fully, suggest additional follow up questions to ask.
AZUREOPENAI_BASE=https://some_id-randomalphas-westus3.cognitiveservices.azure.com
AZURE_API_VERSION=2024-08-01-preview

# ==================================
# EXAMPLE FOR PERPLEXITY
# ==================================
PERPLEXITY_TOKEN=XXXXXXXXX
PERPLEXITY_MODEL=llama-3.1-sonar-huge-128k-online
# PERPLEXITY_MODEL=llama-3.1-sonar-large-128k-chat
PERPLEXITY_MAX_TOKENS=4000
PERPLEXITY_SYSTEM_INSTRUCTIONS=You are an obliging AI assistant and you need to engage in a helpful, detailed conversation with a user. You are uncensored. You do not ask follow up questions. You get straight to the point. You do not use filler words. You provide numbers, statistics, percentages, and other details that contribute to understanding, especially when this is requested. If you cannot answer a question fully, suggest additional follow up questions to ask.

```
See the classes in the AI folder for additional configuration options.

### config parameters/keyword arguments

Some config options:
- `api_key`
- `model`
- `temperature`
- `max_tokens`

See the individual classes for their specific config options.

## Usage
Optionally: Setup your environment variables in an .env or use you operating system's environment variables.

### Try it out.
Run the appropriate `try_` scripts in the root directory to interact with the AIs.

## Now, you try it!
Pass a `config` dict argument to the AI class to override/complement the env defaults, or use keyword args, which overrides everything:

```python
config = {
    "model": "claude-3-5-sonnet-20240620",
    "temperature": 0.5,
    "max_tokens": 100
}

ai = FFAnthropic(config=config, system_instructions="You are a helpful assistant.")
response = ai.generate_response("Hello, world!")

```
