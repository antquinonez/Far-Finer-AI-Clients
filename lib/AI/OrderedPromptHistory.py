from typing import Optional, List, Dict, Any
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import time
from datetime import datetime
import re

@dataclass
class Interaction:
    """Represents a single prompt-response interaction"""
    sequence_number: int
    model: str
    timestamp: float
    prompt_name: Optional[str]
    prompt: str
    response: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_number": self.sequence_number,
            "model": self.model,
            "timestamp": self.timestamp,
            "prompt_name": self.prompt_name,
            "prompt": self.prompt,
            "response": self.response,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat()
        }

class OrderedPromptHistory:
    def __init__(self):
        self.prompt_dict: OrderedDict[str, List[Interaction]] = OrderedDict()
        self._current_sequence = 0
    
    def add_interaction(self, model: str, prompt: str, response: str, prompt_name: Optional[str] = None) -> Interaction:
        """
        Add a new interaction to the history
        
        Args:
            model: The model used for the interaction
            prompt: The prompt text
            response: The response text
            prompt_name: Optional name/key for the prompt. If None, uses prompt text as key
        
        Returns:
            The created Interaction object
        """
        self._current_sequence += 1
        
        # Use prompt text as prompt_name if none provided
        effective_prompt_name = prompt_name if prompt_name is not None else prompt
            
        interaction = Interaction(
            sequence_number=self._current_sequence,
            model=model,
            timestamp=time.time(),
            prompt_name=prompt_name,  # Store original prompt_name (which might be None)
            prompt=prompt,
            response=response
        )
        
        if effective_prompt_name not in self.prompt_dict:
            self.prompt_dict[effective_prompt_name] = []
        self.prompt_dict[effective_prompt_name].append(interaction)
        return interaction

    def get_interactions_by_prompt_name(self, prompt_name: str) -> List[Interaction]:
        """Get all interactions for a specific prompt name"""
        return deepcopy(self.prompt_dict.get(prompt_name, []))
    
    def get_latest_interaction_by_prompt_name(self, prompt_name: str) -> Optional[Interaction]:
        """Get the most recent interaction for a specific prompt name"""
        interactions = self.prompt_dict.get(prompt_name, [])
        return deepcopy(interactions[-1]) if interactions else None
    
    def get_all_prompt_names(self) -> List[str]:
        """Get a list of all prompt names in order of first appearance"""
        return list(self.prompt_dict.keys())
    
    def get_all_interactions(self) -> List[Interaction]:
        """Get all interactions in sequence order"""
        all_interactions = []
        for interactions in self.prompt_dict.values():
            all_interactions.extend(interactions)
        return sorted(deepcopy(all_interactions), key=lambda x: x.sequence_number)
    
    def get_prompt_name_usage_stats(self) -> Dict[str, int]:
        """Get statistics on prompt name usage"""
        return {name: len(interactions) for name, interactions in self.prompt_dict.items()}
    
    def get_interactions_by_model_and_prompt_name(self, model: str, prompt_name: str) -> List[Interaction]:
        """Get all interactions for a specific model and prompt name combination"""
        interactions = self.prompt_dict.get(prompt_name, [])
        return deepcopy([i for i in interactions if i.model == model])
    
    def merge_histories(self, other: 'OrderedPromptHistory') -> None:
        """
        Merge another OrderedPromptHistory into this one
        
        Args:
            other: Another OrderedPromptHistory instance to merge
        """
        for prompt_name, interactions in other.prompt_dict.items():
            if prompt_name not in self.prompt_dict:
                self.prompt_dict[prompt_name] = []
            self.prompt_dict[prompt_name].extend(deepcopy(interactions))
            
        # Resequence all interactions to maintain order
        all_interactions = self.get_all_interactions()
        self._current_sequence = 0
        self.prompt_dict.clear()
        
        for interaction in all_interactions:
            self.add_interaction(
                model=interaction.model,
                prompt=interaction.prompt,
                response=interaction.response,
                prompt_name=interaction.prompt_name
            )
    
    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Convert the entire history to a dictionary organized by prompt names"""
        return {
            prompt_name: [i.to_dict() for i in interactions]
            for prompt_name, interactions in self.prompt_dict.items()
        }
    
    def get_interaction_by_prompt(self, prompt: str) -> Optional[Interaction]:
        """
        Get an interaction by its exact prompt text
        Useful when prompt was used as the prompt_name
        """
        return self.get_latest_interaction_by_prompt_name(prompt)

    def get_latest_responses_by_prompt_names(self, prompt_names: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Get the latest prompt and response for each specified prompt name.
        
        Args:
            prompt_names: List of prompt names to retrieve
            
        Returns:
            Dictionary with prompt names as keys and dictionaries containing 
            'prompt' and 'response' as values
        """
        result = {}
        for prompt_name in prompt_names:
            latest_interaction = self.get_latest_interaction_by_prompt_name(prompt_name)
            if latest_interaction:
                result[prompt_name] = {
                    'prompt': latest_interaction.prompt,
                    'response': latest_interaction.response
                }
        return result
    
    def get_formatted_responses(self, prompt_names: List[str]) -> str:
        """
        Format the latest prompts and responses in the specified format:
        <prompt:[prompt text]>[response]</prompt:[prompt text]>
        
        Args:
            prompt_names: List of prompt names to include in the formatted output
            
        Returns:
            Formatted string containing all prompts and responses
        """
        responses = self.get_latest_responses_by_prompt_names(prompt_names)
        formatted_outputs = []
        
        for prompt_name, content in responses.items():
            formatted_output = f"<prompt:{content['prompt']}>{content['response']}</prompt:{content['prompt']}>"
            
            formatted_output_clean = self._clean_text(formatted_output)       
            formatted_outputs.append(formatted_output_clean)
            # print(f"Formatted output: {formatted_output_clean}")
        
        return '\n'.join(formatted_outputs)

    def _clean_text(self, text):
        cleaned_lines = []
        for line in text.splitlines():
            # Remove extra spaces within each line
            cleaned_line = ' '.join(line.split())
            cleaned_lines.append(cleaned_line)
        
        # Join lines back together with newlines
        return '\n'.join(cleaned_lines)