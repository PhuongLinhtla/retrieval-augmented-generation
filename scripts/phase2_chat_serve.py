from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 (chat): run Streamlit app with Ollama LLM for realtime Q&A.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host for Streamlit server.")
    parser.add_argument("--port", type=int, default=8511, help="Port for Streamlit server.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    app_file = project_root / "app.py"

    env = dict(os.environ)
    env.setdefault("LLM_PROVIDER", "ollama")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_file),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
    ]

    print("Running Phase 2 chat server...")
    print("- LLM provider:", env.get("LLM_PROVIDER", "ollama"))
    print("- URL:", f"http://localhost:{args.port}")

    subprocess.run(command, cwd=project_root, env=env, check=False)


if __name__ == "__main__":
    main()
