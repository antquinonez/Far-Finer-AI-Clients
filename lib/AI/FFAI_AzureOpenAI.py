from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
from copy import deepcopy

from .OrderedPromptHistory import OrderedPromptHistory
from .PermanentHistory import PermanentHistory
from .FFAzureOpenAI import FFAzureOpenAI
import re

# Configure logging
logger = logging.getLogger(__name__)

class FFAI_AzureOpenAI:
    def __init__(self, azure_client: FFAzureOpenAI):
        logger.info("Initializing FFAIAzure wrapper")
        self.client = azure_client
        self.permanent_history = PermanentHistory()
        self.ordered_history = OrderedPromptHistory()

    def _clean_text(self, text):
        cleaned_lines = []
        for line in text.splitlines():
            # Remove extra spaces within each line
            cleaned_line = ' '.join(line.split())
            cleaned_lines.append(cleaned_line)
        
        # Join lines back together with newlines
        return '\n'.join(cleaned_lines)

    def _build_prompt(self, prompt: str, history: Optional[List[str]] = None) -> str:
        if not history:
            return prompt
            
        # Clean the incoming prompt
        cleaned_prompt = re.sub(r'<RAG>[\s\S]*?</RAG>', '', prompt)
        
        # Get history including full conversation chains
        full_history = []
        processed_prompts = set()  # To prevent infinite loops
        
        def collect_history(prompt_names):
            for prompt_name in prompt_names:
                if prompt_name in processed_prompts:
                    continue
                    
                processed_prompts.add(prompt_name)
                latest = self.ordered_history.get_latest_interaction_by_prompt_name(prompt_name)
                
                if latest:
                    # Recursively process this prompt's history if it exists
                    if hasattr(latest, 'history') and latest.history:
                        collect_history(latest.history)
                    
                    # Add this prompt and response
                    full_history.append({
                        'prompt': latest.prompt,
                        'response': latest.response
                    })
        
        collect_history(history)
        
        # Get formatted responses for the full chain
        formatted_responses = []
        for entry in full_history:
            formatted_responses.append(f"<prompt:{entry['prompt']}>{entry['response']}</prompt:{entry['prompt']}>")
        
        rag = "\n".join(formatted_responses)
        
        if rag:
            final_prompt = f"""
            <RAG>
            {rag}
            </RAG>
            ========
            PROMPT
            ========
            {cleaned_prompt}
            """
        else:
            final_prompt = cleaned_prompt
        
        final_prompt = self._clean_text(final_prompt)
        logger.info(f"final prompt: {final_prompt}")
        return final_prompt

    def generate_response(self,
                        prompt: str,
                        model: Optional[str] = None,
                        prompt_name: Optional[str] = None,
                        history: Optional[List[str]] = None ) -> str:
        
        logger.debug(f"Generating response for prompt: {prompt}")
        used_model = model if model else self.client.model
        logger.debug(f"Using model: {used_model}")

        prompt = self._build_prompt(prompt, history)
        logger.debug(f"Using prompt: {prompt}")

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
                prompt_name=prompt_name,
                history=history  # Pass the history parameter here
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