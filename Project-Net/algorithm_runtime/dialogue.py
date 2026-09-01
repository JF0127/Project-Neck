"""Minimal replaceable dialogue policy for Runtime V1."""
from __future__ import annotations

from typing import Protocol


class DialoguePolicy(Protocol):
    def reply(self, user_text: str) -> str: ...


class FixedDialogue:
    def __init__(self, reply_text: str = "I heard you. Thank you for talking with me."):
        self.reply_text = reply_text

    def reply(self, user_text: str) -> str:
        return self.reply_text


class EchoDialogue:
    def reply(self, user_text: str) -> str:
        clean = user_text.strip()
        return f"You said: {clean}" if clean else "I could not hear any speech."
