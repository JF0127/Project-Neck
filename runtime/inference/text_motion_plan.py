"""Text-only semantic action plan; no audio or motion timeline in this contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextAction:
    type: str
    anchor: str
    position: int  # Zero-based Unicode character offset of anchor in reply_text.
    intensity: str

    @classmethod
    def from_dict(cls, value: Any, reply_text: str) -> "TextAction":
        if not isinstance(value, dict) or set(value) != {"type", "anchor", "position", "intensity"}:
            raise ValueError("action requires exactly type, anchor, position, intensity")
        action = cls(**value)
        if action.type not in ("nod", "shake"):
            raise ValueError("action type must be nod or shake")
        if action.intensity not in ("low", "medium", "high"):
            raise ValueError("intensity must be low, medium or high")
        if not isinstance(action.anchor, str) or not action.anchor.strip():
            raise ValueError("anchor must be non-empty")
        if isinstance(action.position, bool) or not isinstance(action.position, int):
            raise ValueError("position must be a zero-based integer character offset")
        if action.position < 0 or reply_text[action.position:action.position + len(action.anchor)] != action.anchor:
            # A unique verbatim anchor is more reliable than an LLM counting Unicode characters.
            if reply_text.count(action.anchor) != 1:
                raise ValueError("anchor does not match reply_text at position")
            action = cls(action.type, action.anchor, reply_text.index(action.anchor), action.intensity)
        return action

    def to_dict(self) -> dict:
        return {"type": self.type, "anchor": self.anchor,
                "position": self.position, "intensity": self.intensity}


@dataclass(frozen=True)
class TextMotionPlan:
    reply_text: str
    actions: tuple[TextAction, ...]

    @classmethod
    def from_dict(cls, value: Any, expected_text: str) -> "TextMotionPlan":
        if not isinstance(value, dict) or set(value) != {"reply_text", "actions"}:
            raise ValueError("plan requires exactly reply_text and actions")
        if value["reply_text"] != expected_text:
            raise ValueError("reply_text must match input verbatim")
        raw = value["actions"]
        if not isinstance(raw, list) or len(raw) > 3:
            raise ValueError("actions must be an array of at most 3 items")
        actions = tuple(TextAction.from_dict(item, expected_text) for item in raw)
        previous_end = 0
        for action in actions:
            if action.position < previous_end:
                raise ValueError("actions must be ordered and non-overlapping")
            previous_end = action.position + len(action.anchor)
        return cls(expected_text, actions)

    def to_dict(self) -> dict:
        return {"reply_text": self.reply_text, "actions": [a.to_dict() for a in self.actions]}
