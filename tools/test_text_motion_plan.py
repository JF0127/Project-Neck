#!/usr/bin/env python3
"""Independent real DeepSeek text-only semantic motion planning check."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.inference.deepseek_motion import DeepSeekMotionBackend

TEXTS = (
    "是的，我完全同意你的看法。",
    "不，这个方案暂时不可行。",
    "北京今天的天气是晴天，气温二十度。",
    "对，你说得有道理，但我不同意最后的结论。",
)


def main():
    backend = DeepSeekMotionBackend()
    for text in TEXTS:
        print(f"reply_text: {text}", flush=True)
        raw, plan = backend.plan_text(text)
        print(f"DeepSeek raw: {raw}", flush=True)
        print("Motion Plan: " + json.dumps(plan.to_dict(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
