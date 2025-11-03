from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ConversationState:
    session_id: str
    persona_name: str
    turns: List[Dict[str, Any]] = field(default_factory=list)
    max_turns: int = 8

    def add_turn(self, role: str, text: str):
        self.turns.append({"role": role, "text": text})
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def as_messages(self, system_prompt: str) -> List[Dict[str, str]]:
        msgs = [{"role": "system", "content": system_prompt}]
        for t in self.turns:
            msgs.append({"role": t["role"], "content": t["text"]})
        return msgs
