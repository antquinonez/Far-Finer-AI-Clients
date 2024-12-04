from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
from copy import deepcopy

from .OrderedPromptHistory import OrderedPromptHistory
from .PermanentHistory import PermanentHistory
from .FFAzureOpenAI import FFAzureOpenAI

# Configure logging
logger = logging.getLogger(__name__)

class FFAI_AzureOpenAI:
    def __init__(self, azure_client: FFAzureOpenAI):
        logger.info("Initializing FFAIAzure wrapper")
        self.client = azure_client
        self.permanent_history = PermanentHistory()
        self.ordered_history = OrderedPromptHistory()
        
    def generate_response(self, prompt: str, model: Optional[str] = None, prompt_name: Optional[str] = None) -> str:
        logger.debug(f"Generating response for prompt: {prompt}")
        used_model = model if model else self.client.model
        logger.debug(f"Using model: {used_model}")

        try:
            # Add to permanent history
            self.permanent_history.add_turn_user(prompt)
            
            # Generate response using the wrapped client
            response = self.client.generate_response(prompt=prompt, model=used_model)
            
            # Add response to histories
            self.permanent_history.add_turn_assistant(response)

            self.ordered_history.add_interaction(
                model=used_model,
                prompt=prompt,
                response=response,
                prompt_name=prompt_name
            )
            
            logger.info("Response generated successfully")
            return response
            
        except Exception as e:
            logger.error("Problem with response generation")
            logger.error(f"  -- exception: {str(e)}")
            raise RuntimeError(f"Error generating response: {str(e)}")

    # OrderedPromptHistory interface methods
    def get_interaction_history(self) -> List[Dict[str, Any]]:
        """Get all interactions as a list of dictionaries"""
        interactions = self.ordered_history.get_all_interactions()
        return [i.to_dict() for i in interactions]
    
    def get_last_n_interactions(self, n: int) -> List[Dict[str, Any]]:
        """Get the last n interactions as dictionaries"""
        all_interactions = self.ordered_history.get_all_interactions()
        return [i.to_dict() for i in all_interactions[-n:]]
    
    def get_interaction(self, sequence_number: int) -> Optional[Dict[str, Any]]:
        """Get a specific interaction by sequence number"""
        all_interactions = self.ordered_history.get_all_interactions()
        interaction = next((i for i in all_interactions if i.sequence_number == sequence_number), None)
        return interaction.to_dict() if interaction else None
    
    def get_model_interactions(self, model: str) -> List[Dict[str, Any]]:
        """Get all interactions for a specific model"""
        all_interactions = self.ordered_history.get_all_interactions()
        return [i.to_dict() for i in all_interactions if i.model == model]
    
    def get_interactions_by_prompt_name(self, prompt_name: str) -> List[Dict[str, Any]]:
        """Get all interactions for a specific prompt name"""
        return [i.to_dict() for i in self.ordered_history.get_interactions_by_prompt_name(prompt_name)]
    
    def get_latest_interaction(self) -> Optional[Dict[str, Any]]:
        """Get the most recent interaction"""
        all_interactions = self.ordered_history.get_all_interactions()
        return all_interactions[-1].to_dict() if all_interactions else None
    
    def get_prompt_history(self) -> List[str]:
        """Get all prompts in order"""
        return [i.prompt for i in self.ordered_history.get_all_interactions()]
    
    def get_response_history(self) -> List[str]:
        """Get all responses in order"""
        return [i.response for i in self.ordered_history.get_all_interactions()]
    
    def get_model_usage_stats(self) -> Dict[str, int]:
        """Get statistics on model usage"""
        usage_stats = {}
        for interaction in self.ordered_history.get_all_interactions():
            usage_stats[interaction.model] = usage_stats.get(interaction.model, 0) + 1
        return usage_stats

    def get_prompt_name_usage_stats(self) -> Dict[str, int]:
        """Get statistics on prompt name usage"""
        return self.ordered_history.get_prompt_name_usage_stats()

    def get_prompt_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get the complete history as an ordered dictionary keyed by prompts
        Returns:
            Dict[str, List[Dict[str, Any]]]: OrderedDict where:
                - keys are prompt names (or prompts if no name was provided)
                - values are lists of interaction dictionaries for that prompt
        """
        return self.ordered_history.to_dict()

    def clear_conversation(self):
        """Clear the conversation history in the wrapped client"""
        logger.info("Clearing conversation history in wrapped client (permanent and ordered histories retained)")
        self.client.clear_conversation()