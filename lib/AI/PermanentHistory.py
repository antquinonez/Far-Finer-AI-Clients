# Copyright (c) 2024 Antonio Quinonez
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import time
import logging
from typing import Optional, List
from anthropic import Anthropic
from dotenv import load_dotenv
from copy import deepcopy

class PermanentHistory:
    def __init__(self):
        self.turns = []
        self.timestamp = time.time()

    def add_turn_assistant(self, content):
        self.turns.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": content
                }
            ],
            "timestamp": time.time()
        })

    def add_turn_user(self, content):
        if self.turns and self.turns[-1]["role"] == "user":
            # If the last turn was a user, update its content instead of adding a new turn
            self.turns[-1]["content"][0]["text"] += "\n" + content
            self.turns[-1]["timestamp"] = time.time()
        else:
            self.turns.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": content
                    }
                ],
                "timestamp": time.time()
            })

    def get_all_turns(self):
        """Returns all turns with their timestamps."""
        return deepcopy(self.turns)  # Return a deep copy to prevent modification

    def get_turns_since(self, timestamp: float):
        """Returns all turns that occurred after the specified timestamp."""
        return [turn for turn in self.turns if turn["timestamp"] > timestamp]