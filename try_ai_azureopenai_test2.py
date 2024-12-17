import os
import sys
import logging
# from libs.ARBES_Logging import initialize_logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..')))

from lib.AI.FFAI_AzureOpenAI import FFAI_AzureOpenAI as AI
from lib.AI.FFAzureOpenAI import FFAzureOpenAI

# Create the Azure client
azure_client = FFAzureOpenAI()

# Create the wrapper
ffai = AI(azure_client)

# Generate responses and track history

response = ffai.generate_response("What is taller, a bush or a tree?")
# response = ffai.generate_response("how are you?", prompt_name="greeting")
# response = ffai.generate_response("what is 2 +2?", prompt_name="math")
# response = ffai.generate_response("concatenate these words: cat, dog ", prompt_name="pet")



# ffai.clear_conversation()
# response = ffai.generate_response("what did you say to the math problem?", prompt_name='final query', history=["pet", "math", "greeting"])
# print("RESPONSE:")
# print(response)
# print("============================")

# response = ffai.generate_response("concatenate these words again: cat, dog,shrimp ")

ffai.clear_conversation()
# The history will now include both the 'final query' and its associated history
# response = ffai.generate_response("what did you say to the question?", prompt_name='really final query', history=["final query"])
# print(response)


# prompt_dict = ffai.get_prompt_dict()
# print("=================")
# print(prompt_dict)
# Access history using any of the interface methods

history = ffai.get_interaction_history()
interactions = ffai.get_all_interactions()
# latest = ffai.get_latest_interaction()
# stats = ffai.get_model_usage_stats()
# prompt_dict = ffai.get_prompt_dict()
# formatted_responses = ffai.get_formatted_responses(['math'])

print("--------------------------")
print(history)
print("--------------------------")
print(interactions)

# print(latest)
# print("----------")
# print(stats)
# print("----------")
# print(prompt_dict)
# print("----------")
# print(formatted_responses)
