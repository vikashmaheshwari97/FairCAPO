from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--app",
        default="dashboard/app.py",
        help="Path to the Streamlit dashboard app.",
    )
    parser.add_argument(
        "--port",
        default="8501",
        help="Streamlit server port.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Streamlit in headless mode.",
    )

    args = parser.parse_args()

    app_path = Path(args.app)

    if not app_path.exists():
        raise FileNotFoundError(f"Dashboard app not found: {app_path}")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
    ]

    if args.headless:
        command.extend(["--server.headless", "true"])

    print("Launching HEAL-CAPO dashboard...")
    print("Command:", " ".join(command))

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()