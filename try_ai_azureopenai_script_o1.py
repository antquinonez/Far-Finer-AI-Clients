import os
import sys
import logging


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..')))

from lib.AI.FFAI_AzureOpenAI import FFAI_AzureOpenAI as AI
from lib.AI.FFAzureOpenAI import FFAzureOpenAI

# Create the Azure client
azure_client = FFAzureOpenAI(model='o1-mini', is_o1=True)
logger.info("Azure client created")

# Create the wrapper
ffai = AI(azure_client)
logger.info("AI wrapper created")

# Generate responses and track history
logger.info("Generating responses...")

response = ffai.generate_response("How do I use AI?")
logger.info(f"Response to 'How do I use AI?': {response}")

response = ffai.generate_response("how are you?", prompt_name="greeting")
logger.info(f"Response to greeting: {response}")

response = ffai.generate_response("what is 2 +2?", prompt_name="math")
logger.info(f"Response to math question: {response}")

response = ffai.generate_response("concatenate these words: cat, dog ", prompt_name="pet")
logger.info(f"Response to concatenation: {response}")

ffai.clear_conversation()
logger.info("Conversation cleared")

logger.info("=============================================================================================================================================================================")
response = ffai.generate_response("what did you say to the math problem?", prompt_name='final query', history=["pet", "math", "greeting"])
logger.info("RESPONSE:")
logger.info(response)
logger.info("============================")

logger.info("=============================================================================================================================================================================")
response = ffai.generate_response("concatenate these words again: cat, dog,shrimp ")
logger.info(f"Response to second concatenation: {response}")

logger.info("=============================================================================================================================================================================")
ffai.clear_conversation()
logger.info("Conversation cleared")
logger.info("=============================================================================================================================================================================")

# The history will now include both the 'final query' and its associated history
logger.info("======================================================================================")
response = ffai.generate_response("what did you say to the question?",
                                  prompt_name='really final query',
                                  history=["final query"],
                                  infer_o1=True)

logger.info(f"Final response: {response}")

logger.info("======================================================================================")
response = ffai.generate_response("Was that a hard question? Why not?", 
                                  prompt_name='really final query',
                                  history=["final query"],
                                  is_o1=True)

logger.info(f"Final response: {response}")

logger.info("======================================================================================")
response = ffai.generate_response("What is the level of difficulty of the question asked?", 
                                  prompt_name='really final query',
                                  history=["final query"],
                                  is_o1=False)


logger.info("======================================================================================")
response = ffai.generate_response("What is the level of difficulty of the question asked?", 
                                  prompt_name='really final query',
                                  model='o1-mini',
                                  history=["final query"],
                                  is_o1=False)


logger.info("======================================================================================")
response = ffai.generate_response("What is the level of difficulty of the question asked?", 
                                  prompt_name='really final query',
                                  model='gpt-4o',
                                  history=["final query"],
                                  is_o1=False)

logger.info("======================================================================================")
response = ffai.generate_response("What is the level of difficulty of the question asked?", 
                                  prompt_name='really final query',
                                  model='gpt-4o-mini',
                                  history=["final query"],
                                  is_o1=False)


logger.info(f"Final response: {response}")

logger.info("=============================================================================================================================================================================")
response = ffai.generate_response("what did you say to the question? Also, how do i spell cat? Respond with a JSON dict.", prompt_name='really final query', history=["final query"])
logger.info(f"Final response: {response}")


# Access history using any of the interface methods
history = ffai.get_interaction_history()
clean_history = ffai.get_clean_interaction_history()
attr_history = ffai.get_prompt_attr_history()


latest = ffai.get_latest_interaction()
stats = ffai.get_model_usage_stats()
prompt_dict = ffai.get_prompt_dict()
formatted_responses = ffai.get_formatted_responses(['math'])

logger.info("=============================================================================================================================================================================")
logger.info("Interaction History:")
logger.info(history)
logger.info("=============================================================================================================================================================================")
logger.info("Clean Interaction History:")
logger.info(clean_history)
logger.info("=============================================================================================================================================================================")
logger.info("Prompt Attr History:")
logger.info(attr_history)
logger.info("=============================================================================================================================================================================")
logger.info("Latest Interaction:")
logger.info(latest)
logger.info("=============================================================================================================================================================================")
logger.info("Model Usage Stats:")
logger.info(stats)
logger.info("=============================================================================================================================================================================")
logger.info("Prompt Dictionary:")
logger.info(prompt_dict)
logger.info("=============================================================================================================================================================================")
logger.info("Formatted Responses:")
logger.info(formatted_responses)

logger.info("Script execution completed")