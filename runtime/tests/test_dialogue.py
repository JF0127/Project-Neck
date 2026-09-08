from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from runtime.dialogue import DeepSeekDialogue, DialogueError, FixedDialogue
from runtime.contracts import DialogueRequest, SessionContext, TurnSummary


class FakeCompletions:
    def __init__(self, results):
        self.results = list(results)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeClient:
    def __init__(self, results):
        self.chat = SimpleNamespace(completions=FakeCompletions(results))


def response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=2,
            total_tokens=12,
        ),
    )


def deepseek(client: FakeClient, **kwargs) -> DeepSeekDialogue:
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-secret-key"}):
        return DeepSeekDialogue(client=client, **kwargs)


class DialogueRequestTests(unittest.TestCase):
    def test_request_uses_at_most_ten_session_turns(self) -> None:
        session = SessionContext("session")
        for number in range(12):
            session.add_turn(
                TurnSummary(
                    turn_id=f"turn_{number}",
                    user_text=f"user {number}",
                    robot_text=f"robot {number}",
                    status="complete",
                )
            )

        request = DialogueRequest.from_session("current", session)
        self.assertEqual(len(request.history), 10)
        self.assertEqual(request.history[0].turn_id, "turn_2")
        self.assertEqual(request.history[-1].turn_id, "turn_11")


class DeepSeekDialogueTests(unittest.TestCase):
    def test_messages_are_built_only_from_request_history(self) -> None:
        client = FakeClient([response("reply one"), response("reply two")])
        dialogue = deepseek(client)

        first_session = SessionContext("first")
        first_session.add_turn(TurnSummary("one", "first user", "first robot", "complete"))
        second_session = SessionContext("second")
        second_session.add_turn(TurnSummary("two", "second user", "second robot", "complete"))

        dialogue.reply(DialogueRequest.from_session("question one", first_session))
        dialogue.reply(DialogueRequest.from_session("question two", second_session))

        first_messages = client.chat.completions.calls[0]["messages"]
        second_messages = client.chat.completions.calls[1]["messages"]
        self.assertEqual(
            first_messages[-3:],
            [
                {"role": "user", "content": "first user"},
                {"role": "assistant", "content": "first robot"},
                {"role": "user", "content": "question one"},
            ],
        )
        self.assertEqual(
            second_messages[-3:],
            [
                {"role": "user", "content": "second user"},
                {"role": "assistant", "content": "second robot"},
                {"role": "user", "content": "question two"},
            ],
        )
        self.assertNotIn("first user", str(second_messages))
        self.assertFalse(hasattr(dialogue, "_history"))

    def test_api_parameters_and_result(self) -> None:
        client = FakeClient([response("  robot reply  ")])
        dialogue = deepseek(client, temperature=0.4)
        result = dialogue.reply(DialogueRequest("hello", ()))

        self.assertEqual(result, "robot reply")
        call = client.chat.completions.calls[0]
        self.assertEqual(call["model"], "deepseek-v4-flash")
        self.assertEqual(call["temperature"], 0.4)
        self.assertEqual(call["timeout"], 30.0)
        self.assertEqual(call["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(dialogue.metadata["usage"]["total_tokens"], 12)

    def test_network_failure_raises_clear_error_without_fallback(self) -> None:
        NetworkError = type("APIConnectionError", (Exception,), {})
        dialogue = deepseek(FakeClient([NetworkError("transport detail")]))

        with self.assertRaisesRegex(DialogueError, "network connection failed"):
            dialogue.reply(DialogueRequest("hello", ()))

    def test_missing_key_fails_at_initialization(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(DialogueError, "DEEPSEEK_API_KEY is required"):
                DeepSeekDialogue(client=FakeClient([]))

    def test_fixed_dialogue_uses_new_request(self) -> None:
        self.assertEqual(
            FixedDialogue("fixed").reply(DialogueRequest("hello", ())),
            "fixed",
        )


if __name__ == "__main__":
    unittest.main()
