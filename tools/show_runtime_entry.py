# -*- coding: utf-8 -*-
"""
Project-Neck / Show 统一启动入口。

放置位置：
    /home/jhl/projects/Project-Neck/tools/show_runtime_entry.py

目的：
1. Neck 同学调试完成后，用本文件启动 Ubuntu V1 做最终验证；
2. Show 也只调用本文件，不再自己拼 Project-Neck 启动逻辑；
3. 代理策略集中在这里，避免 Show 和手动终端行为不一致。

标准真机命令：
    cd /home/jhl/projects/Project-Neck
    runtime/.venv/bin/python tools/show_runtime_entry.py --motor

不接 Motor：
    cd /home/jhl/projects/Project-Neck
    runtime/.venv/bin/python tools/show_runtime_entry.py

代理策略：
- auto（默认）：若当前终端已有 Mac proxy，先检查代理端口是否可达；
  可达则原样继承；不可达则清除 HTTP/HTTPS/ALL proxy，改为直连。
- inherit：无条件继承当前终端 proxy。
- direct：无条件清除 HTTP/HTTPS/ALL proxy。

也可通过环境变量覆盖默认策略：
    export PROJECT_NECK_PROXY_POLICY=auto
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse


PROXY_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

REQUIRED_KEYS = (
    "DEEPSEEK_API_KEY",
    "VOLCENGINE_TTS_API_KEY",
)


def _project_root() -> Path:
    # 本文件固定放在 Project-Neck/tools/ 下。
    return Path(__file__).resolve().parents[1]


def _runtime_python(root: Path) -> Path:
    return root / "runtime" / ".venv" / "bin" / "python"


def _first_proxy(env: dict[str, str]) -> str:
    for key in (
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        value = env.get(key, "").strip()
        if value:
            return value
    return ""


def _proxy_endpoint(proxy_url: str) -> tuple[str, int] | None:
    value = proxy_url.strip()
    if not value:
        return None
    parsed = urlparse(value if "://" in value else f"http://{value}")
    host = parsed.hostname
    port = parsed.port
    if not host:
        return None
    if port is None:
        if parsed.scheme in ("https",):
            port = 443
        elif parsed.scheme.startswith("socks"):
            port = 1080
        else:
            port = 80
    return host, int(port)


def _proxy_reachable(proxy_url: str, timeout_sec: float = 1.5) -> bool:
    endpoint = _proxy_endpoint(proxy_url)
    if endpoint is None:
        return False
    host, port = endpoint
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except OSError:
        return False


def _clear_proxy(env: dict[str, str]) -> None:
    for key in PROXY_KEYS:
        env.pop(key, None)


def _prepare_environment(policy: str) -> dict[str, str]:
    env = os.environ.copy()
    proxy = _first_proxy(env)

    if policy == "direct":
        _clear_proxy(env)
        print("[show-entry] proxy_policy=direct -> HTTP/HTTPS/ALL proxy cleared", flush=True)
        return env

    if policy == "inherit":
        if proxy:
            endpoint = _proxy_endpoint(proxy)
            target = f"{endpoint[0]}:{endpoint[1]}" if endpoint else "configured"
            print(f"[show-entry] proxy_policy=inherit -> proxy={target}", flush=True)
        else:
            print("[show-entry] proxy_policy=inherit -> no proxy configured", flush=True)
        return env

    # auto：Mac 代理在线时继续使用；Mac/代理离线时自动回退直连。
    if not proxy:
        print("[show-entry] proxy_policy=auto -> no proxy configured; direct", flush=True)
        return env

    endpoint = _proxy_endpoint(proxy)
    target = f"{endpoint[0]}:{endpoint[1]}" if endpoint else "configured"
    if _proxy_reachable(proxy):
        print(f"[show-entry] proxy_policy=auto -> proxy reachable; inherit {target}", flush=True)
        return env

    _clear_proxy(env)
    print(
        f"[show-entry] proxy_policy=auto -> proxy unreachable ({target}); "
        "HTTP/HTTPS/ALL proxy cleared; direct fallback",
        flush=True,
    )
    return env


def _check_required_environment(env: dict[str, str]) -> None:
    missing = [name for name in REQUIRED_KEYS if not env.get(name)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"缺少必要环境变量：{joined}")

    # 只打印 SET/UNSET，绝不打印 key 内容。
    print("[show-entry] DEEPSEEK_API_KEY=SET", flush=True)
    print("[show-entry] VOLCENGINE_TTS_API_KEY=SET", flush=True)
    speaker = env.get("VOLCENGINE_TTS_SPEAKER", "")
    print(
        "[show-entry] VOLCENGINE_TTS_SPEAKER="
        + ("SET" if speaker else "DEFAULT"),
        flush=True,
    )


def _ensure_runtime_python(root: Path) -> Path:
    python_path = _runtime_python(root)
    if not python_path.is_file():
        raise FileNotFoundError(f"找不到 Runtime Python：{python_path}")
    return python_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Project-Neck Ubuntu V1 / Show 统一启动入口"
    )
    parser.add_argument(
        "--motor",
        action="store_true",
        help="启用真机 Neck；启动前必须已运行 master_stack_test 并执行 NeckPoseSet 0 0 0 0",
    )
    parser.add_argument(
        "--proxy-policy",
        choices=("auto", "inherit", "direct"),
        default=os.environ.get("PROJECT_NECK_PROXY_POLICY", "auto").strip().lower() or "auto",
        help="代理策略；默认 auto",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查入口、环境变量与代理策略，不启动 Runtime",
    )
    args = parser.parse_args()

    root = _project_root()
    runtime_python = _ensure_runtime_python(root)
    env = _prepare_environment(args.proxy_policy)
    _check_required_environment(env)

    print(f"[show-entry] project_root={root}", flush=True)
    print(f"[show-entry] runtime_python={runtime_python}", flush=True)
    print(f"[show-entry] motor={'ON' if args.motor else 'OFF'}", flush=True)

    if args.check_only:
        print("[show-entry] check-only OK", flush=True)
        return 0

    command = [
        str(runtime_python),
        "-m",
        "runtime",
        "--ubuntu-v1",
    ]
    if args.motor:
        command.append("--motor")

    print("[show-entry] exec=" + " ".join(command), flush=True)

    # 用 exec 替换当前进程：Show、手动终端、Neck 同学最终进入的是完全相同的 Runtime 进程。
    os.chdir(root)
    os.execve(str(runtime_python), command, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
