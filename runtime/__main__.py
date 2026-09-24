"""Production entry point for ``python -m runtime``."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

RUNTIME_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = RUNTIME_ROOT / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project-Neck production Runtime: Silero VAD -> Qwen3-ASR "
            "Streaming -> DeepSeek -> Doubao TTS -> Speaking Motion V2"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Runtime YAML config path",
    )
    parser.add_argument(
        "--debug-deepseek-events",
        action="store_true",
        help="print every DeepSeek Responses API event type",
    )
    parser.add_argument("--xiaozhi", action="store_true", help="run XiaoZhi BoardBridge + Global Motion mode")
    parser.add_argument("--ubuntu-v1", action="store_true", help="run Ubuntu-only local mic/streaming TTS/per-turn trajectory mode")
    parser.add_argument("--motor", action="store_true", help="explicitly enable Motor for --ubuntu-v1 (requires manual neutral)")
    parser.add_argument("--dry-run", action="store_true", help="sample XiaoZhi Global Motion without Motor")
    parser.add_argument("--duration", type=float, default=None, help="dry-run duration in seconds (default 60)")
    parser.add_argument("--output", type=Path, default=None, help="save dry-run motion CSV")
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    """Load the shared Runtime YAML; also used by offline audit tools."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read runtime/config.yaml") from exc

    config_path = path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Runtime config root must be a mapping")
    return value, config_path


def main() -> None:
    args = parse_args()
    try:
        if args.xiaozhi and args.ubuntu_v1:
            raise ValueError("--xiaozhi and --ubuntu-v1 are mutually exclusive")
        if args.motor and not args.ubuntu_v1:
            raise ValueError("--motor requires --ubuntu-v1")
        if not args.xiaozhi and (args.dry_run or args.duration is not None or args.output is not None):
            raise ValueError("--dry-run, --duration and --output require --xiaozhi")
        config, config_path = load_config(args.config)
        if args.xiaozhi:
            from .xiaozhi_motion_runtime import XiaoZhiMotionRuntime
            if args.duration is not None and not args.dry_run:
                raise ValueError("--duration requires --dry-run")
            asyncio.run(XiaoZhiMotionRuntime(config, dry_run=args.dry_run).run(
                duration=args.duration, output=args.output))
            return
        if args.ubuntu_v1:
            from .ubuntu_v1_runtime import UbuntuV1Runtime
            asyncio.run(UbuntuV1Runtime(config, config_path, send_to_motor=args.motor).run())
            return
        from .qwen_streaming_runtime import (
            build_production_runtime,
            serve_production_runtime,
        )

        runtime, monitor, host, port = build_production_runtime(
            config,
            config_path,
            debug=args.debug_deepseek_events,
        )
    except Exception as exc:
        raise SystemExit(f"runtime: error: {type(exc).__name__}: {exc}") from None

    try:
        asyncio.run(serve_production_runtime(runtime, monitor, host, port))
    except KeyboardInterrupt:
        print("[runtime] stopped")


if __name__ == "__main__":
    main()
