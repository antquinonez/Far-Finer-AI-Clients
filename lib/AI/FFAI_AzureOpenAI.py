from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import time

from .OrderedPromptHistory import OrderedPromptHistory
from .PermanentHistory import PermanentHistory

# Configure logging
logger = logging.getLogger(__name__)

class FFAI_AzureOpenAI:
    def __init__(self, azure_client):
        logger.info("Initializing FFAIAzure wrapper")
        self.client = azure_client
        self.history = []
        self.permanent_history = PermanentHistory()
        self.ordered_history = OrderedPromptHistory()

    def _build_prompt(self, prompt: str, history: Optional[List[str]] = None) -> str:
        if not history:
            logger.debug("No history provided, returning original prompt")
            return prompt
            
        logger.info(f"Building prompt with history references: {history}")
        logger.info(f"Current history size: {len(self.history)}")
        
        # Debug current history
        for idx, entry in enumerate(self.history):
            logger.debug(f"History entry {idx}:")
            logger.debug(f"  Prompt name: {entry.get('prompt_name')}")
            logger.debug(f"  Prompt: {entry.get('prompt')}")
            logger.debug(f"  Response: {entry.get('response')}")

        # Get historical interactions for each prompt name
        history_entries = []
        for prompt_name in history:
            logger.debug(f"Looking for entries with prompt_name: {prompt_name}")
            matching_entries = [
                entry for entry in self.history 
                if entry.get('prompt_name') == prompt_name
            ]
            logger.debug(f"Found {len(matching_entries)} matching entries")
            
            if matching_entries:
                latest = matching_entries[-1]
                history_entries.append({
                    'prompt_name': latest.get('prompt_name'),
                    'prompt': latest['prompt'],
                    'response': latest['response']
                })
                logger.debug(f"Added entry for {prompt_name}: {latest['prompt']} -> {latest['response']}")

        # Format history entries
        formatted_history = []
        for entry in history_entries:
            formatted_entry = (
                f"<interaction prompt_name='{entry['prompt_name']}'>\n"
                f"USER: {entry['prompt']}\n"
                f"SYSTEM: {entry['response']}\n"
                f"</interaction>"
            )
            formatted_history.append(formatted_entry)
        
        # Combine history with current prompt
        if formatted_history:
            final_prompt = (
                "<conversation_history>\n" + 
                "\n".join(formatted_history) + 
                "\n</conversation_history>\n" +
                "===\n" +
                "Based on the conversation history above, please answer: " + prompt
            )
        else:
            final_prompt = prompt
            
        logger.info(f"Final constructed prompt:\n{final_prompt}")
        return final_prompt

    def generate_response(self,
                         prompt: str,
                         model: Optional[str] = None,
                         prompt_name: Optional[str] = None,
                         history: Optional[List[str]] = None) -> str:
        """Generate response using Azure OpenAI"""
        logger.info(f"Generating response for prompt: '{prompt}' with prompt_name: '{prompt_name}'")
        used_model = model if model else self.client.model

        try:
            # Build prompt with history
            final_prompt = self._build_prompt(prompt, history)
            
            # Generate response
            response = self.client.generate_response(
                prompt=final_prompt,
                model=used_model
            )
            
            # Store interaction
            interaction = {
                'prompt': prompt,
                'response': response,
                'prompt_name': prompt_name,
                'timestamp': time.time(),
                'model': used_model
            }
            self.history.append(interaction)

            logger.debug(f"Added new interaction to history: {interaction}")
            
            logger.info("Response generated successfully")

            # ==================================================================================
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
                prompt_name=prompt_name,
                history=history  # Pass the history parameter here
            )
            # ==================================================================================

            return response
            
        except Exception as e:
            logger.error(f"Problem with response generation: {str(e)}")
            logger.error(f"Prompt: {prompt}")
            logger.error(f"History: {history}")
            raise

    def clear_conversation(self):
        """Clear conversation in client but retain history"""
        logger.info(f"Clearing conversation. Current history size: {len(self.history)}")
        self.client.clear_conversation()

    def get_interaction_history(self) -> List[Dict[str, Any]]:
        """Get complete history"""
        return self.history

    def get_latest_interaction_by_prompt_name(self, prompt_name: str) -> Optional[Dict[str, Any]]:
        """Get most recent interaction for a prompt name"""
        matching = [e for e in self.history if e.get('prompt_name') == prompt_name]
        return matching[-1] if matching else None
    
    # ===========================================================================
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


    def get_latest_responses_by_prompt_names(self, prompt_names: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Get the latest prompt and response for each specified prompt name.
        
        Args:
            prompt_names: List of prompt names to retrieve
            
        Returns:
            Dictionary with prompt names as keys and dictionaries containing 
            'prompt' and 'response' as values
        """
        return self.ordered_history.get_latest_responses_by_prompt_names(prompt_names)
    
    def get_formatted_responses(self, prompt_names: List[str]) -> str:
        """
        Get formatted string output of latest prompts and responses.
        
        Args:
            prompt_names: List of prompt names to include
            
        Returns:
            Formatted string in the format:
            <prompt:[prompt text]>[response]</prompt:[prompt text]>
        """
        return self.ordered_history.get_formatted_responses(prompt_names)