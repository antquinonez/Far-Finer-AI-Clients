# Far-Finer-AI-Clients

Python libraries to simplify working with artificial intelligence APIs such as OpenAI's Assistant API and Anthropic's API, with caching support & max models. Includes chat history management and robust config handling.

If you work with text, you'll find these classes much easier to work with than the raw APIs from OpenAI or Anthropic.

## Status

This is alpha software. Use at your own risk. Watch those tokens!

## AI Clients

### Anthropic:
1. `FFAnthropic`: Use of Basic and Max models
2. `FFAnthropicCached`: Caching of system instructions

### OpenAI:
- `FFOpenAI`: Uses the OpenAI Assistant API. See: https://platform.openai.com/docs/assistants/overview

## Installation

Copy the files in the `lib` directory to your own `lib` directory.

## Configuration

### .env file

Optionally. create a `.env` file in the root directory with the following variables (DO NOT ADD .env TO GIT):

```
OPENAI_TOKEN=your_openai_token
ANTHROPIC_TOKEN=your_anthropic_token

ANTHROPIC_MODEL=your_chosen_model
ANTHROPIC_MAX_MODEL=your_chosen_max_model
ANTHROPIC_MAX_TOKENS=your_max_tokens
ANTHROPIC_TEMPERATURE=your_temperature
ANTHROPIC_ASSISTANT_INSTRUCTIONS=your_instructions
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
