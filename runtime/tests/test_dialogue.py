from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import os
import tempfile
import unittest
from unittest.mock import patch

from runtime.__main__ import build_dialogue
from runtime.dialogue import DeepSeekDialogue, DialogueError, EchoDialogue, FixedDialogue
from runtime.experiment_logger import ExperimentLogger


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


def response(text: str, usage: bool = False):
    token_usage = (
        SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12)
        if usage
        else None
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=token_usage,
    )


def deepseek(client: FakeClient, **kwargs) -> DeepSeekDialogue:
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-secret-key"}):
        return DeepSeekDialogue(client=client, **kwargs)


class DeepSeekDialogueTest(unittest.TestCase):
    def test_request_contains_system_and_user_messages(self):
        client = FakeClient([response("你好")])
        dialogue = deepseek(client)
        self.assertEqual(dialogue.reply("  你好  "), "你好")
        call = client.chat.completions.calls[0]
        self.assertEqual([message["role"] for message in call["messages"]], ["system", "user"])
        self.assertEqual(call["messages"][1]["content"], "你好")
        self.assertEqual(call["model"], "deepseek-v4-flash")
        self.assertEqual(call["max_tokens"], 128)
        self.assertEqual(call["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(call["timeout"], 30.0)

    def test_reply_returns_nonempty_robot_text(self):
        dialogue = deepseek(FakeClient([response("  你好  ", usage=True)]))
        self.assertEqual(dialogue.reply("测试"), "你好")
        self.assertEqual(
            dialogue.metadata["usage"],
            {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        )

    def test_success_commits_complete_history_pair(self):
        dialogue = deepseek(FakeClient([response("回答一"), response("回答二")]))
        dialogue.reply("问题一")
        dialogue.reply("问题二")
        self.assertEqual(
            [message["role"] for message in dialogue.history_messages],
            ["system", "user", "assistant", "user", "assistant"],
        )
        second_request = dialogue._client.chat.completions.calls[1]["messages"]
        self.assertEqual(second_request[-3:], [
            {"role": "user", "content": "问题一"},
            {"role": "assistant", "content": "回答一"},
            {"role": "user", "content": "问题二"},
        ])

    def test_history_drops_oldest_complete_pair(self):
        dialogue = deepseek(
            FakeClient([response("答一"), response("答二"), response("答三")]),
            history_turns=2,
        )
        for user in ("问一", "问二", "问三"):
            dialogue.reply(user)
        self.assertEqual(dialogue.history_messages, [
            {"role": "system", "content": dialogue.system_prompt},
            {"role": "user", "content": "问二"},
            {"role": "assistant", "content": "答二"},
            {"role": "user", "content": "问三"},
            {"role": "assistant", "content": "答三"},
        ])

    def test_api_failure_does_not_update_history(self):
        NetworkError = type("APIConnectionError", (Exception,), {})
        dialogue = deepseek(FakeClient([NetworkError("secret transport detail")]))
        with self.assertRaisesRegex(DialogueError, "network connection failed"):
            dialogue.reply("你好")
        self.assertEqual(len(dialogue.history_messages), 1)
        self.assertEqual(dialogue.metadata["status"], "error")

    def test_empty_api_response_fails_without_history_update(self):
        dialogue = deepseek(FakeClient([response("  ")]))
        with self.assertRaisesRegex(DialogueError, "empty robot text"):
            dialogue.reply("你好")
        self.assertEqual(len(dialogue.history_messages), 1)

    def test_empty_user_text_does_not_call_api(self):
        client = FakeClient([])
        dialogue = deepseek(client)
        with self.assertRaisesRegex(DialogueError, "non-empty user text"):
            dialogue.reply("   ")
        self.assertEqual(client.chat.completions.calls, [])

    def test_missing_api_key_fails_at_initialization(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                DialogueError, "DEEPSEEK_API_KEY is required for dialogue=deepseek"
            ):
                DeepSeekDialogue(client=FakeClient([]))

    def test_fixed_and_echo_backends_still_work(self):
        self.assertEqual(FixedDialogue("固定回答").reply("任意输入"), "固定回答")
        self.assertEqual(EchoDialogue().reply(" hello "), "You said: hello")

    def test_fixed_cli_backend_does_not_create_deepseek_or_require_key(self):
        args = SimpleNamespace(
            dialogue="fixed",
            fixed_reply="离线回答",
            deepseek_model="unused",
            deepseek_base_url="unused",
            dialogue_history_turns=5,
            dialogue_timeout=30.0,
        )
        with patch.dict(os.environ, {}, clear=True), patch(
            "runtime.__main__.DeepSeekDialogue"
        ) as deepseek_class:
            dialogue = build_dialogue(args)
        deepseek_class.assert_not_called()
        self.assertEqual(dialogue.reply("你好"), "离线回答")

    def test_api_key_is_absent_from_repr_metadata_and_experiment_log(self):
        secret = "deepseek-test-key-must-not-leak"
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": secret}):
            dialogue = DeepSeekDialogue(client=FakeClient([response("你好", usage=True)]))
        dialogue.reply("测试")
        serialized = repr(dialogue) + json.dumps(dialogue.metadata, ensure_ascii=False)
        self.assertNotIn(secret, serialized)

        with tempfile.TemporaryDirectory() as temporary:
            logger = ExperimentLogger("test.pt", "mock-neck", 30.0, Path(temporary))
            turn_id = logger.start_turn(1)
            metadata = dialogue.metadata
            logger.record_dialogue(
                turn_id,
                user_text="测试",
                robot_text="你好",
                dialogue_backend=metadata["backend"],
                dialogue_model=metadata["model"],
                dialogue_latency_sec=metadata["latency_sec"],
                dialogue_usage=metadata["usage"],
            )
            logged = (logger.session_dir / "turns" / turn_id / "dialogue.json").read_text(
                encoding="utf-8"
            )
            logger.end_session()
        self.assertNotIn(secret, logged)
        document = json.loads(logged)
        self.assertEqual(document["dialogue_backend"], "deepseek")
        self.assertEqual(document["dialogue_model"], "deepseek-v4-flash")
        self.assertEqual(document["dialogue_usage"]["total_tokens"], 12)


if __name__ == "__main__":
    unittest.main()
