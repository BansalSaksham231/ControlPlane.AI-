"""
ControlPlane.ai launcher.

    python run_app.py            # start the web UI (recommended for the demo)
    python run_app.py --api      # start the FastAPI backend instead
    python run_app.py --both     # start API + UI together
    python run_app.py --demo     # run the end-to-end demo and exit
    python run_app.py --no-open  # don't auto-open the browser

Cross-platform (works in PowerShell / cmd / bash). No shell scripts.

Locally the servers bind to 127.0.0.1 so the printed URL is directly
clickable. On a hosting platform (which sets $PORT) they bind to
0.0.0.0 automatically.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

UI_PORT = os.environ.get("CONTROLPLANE_UI_PORT", "8501")
API_PORT = os.environ.get("CONTROLPLANE_API_PORT", "8000")

# Platforms like Render / Railway / HF Spaces set $PORT — then we must
# listen on all interfaces. Locally, bind to loopback for a clean URL.
BIND_ADDR = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
VIEW_HOST = "localhost"


def _banner(lines: list[str]) -> None:
    width = max(len(l) for l in lines) + 4
    print("\n" + "=" * width, flush=True)
    for line in lines:
        print("  " + line, flush=True)
    print("=" * width + "\n", flush=True)


def _ensure_data() -> None:
    generated = os.path.join(HERE, "data", "generated", "interactions.csv")
    if not os.path.exists(generated):
        print("Generating synthetic datasets (first run, ~2s)…")
        subprocess.run([PY, os.path.join(HERE, "run_generator.py")], check=True, cwd=HERE)


def _open_browser_when_ready(url: str, health_url: str) -> None:
    import urllib.request

    for _ in range(60):
        time.sleep(1)
        try:
            urllib.request.urlopen(health_url, timeout=2)
            webbrowser.open(url)
            return
        except Exception:
            continue


def start_ui(auto_open: bool) -> subprocess.Popen:
    url = f"http://{VIEW_HOST}:{UI_PORT}"
    _banner([f"ControlPlane.ai web UI  ->  {url}", "Open that URL in your browser.  Ctrl+C to stop."])
    if auto_open:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(url, f"http://{VIEW_HOST}:{UI_PORT}/_stcore/health"),
            daemon=True,
        ).start()
    return subprocess.Popen(
        [
            PY, "-m", "streamlit", "run", os.path.join(HERE, "streamlit_app.py"),
            "--server.port", UI_PORT,
            "--server.address", BIND_ADDR,
            "--server.headless", "true",
        ],
        cwd=HERE,
    )


def start_api(auto_open: bool) -> subprocess.Popen:
    docs = f"http://{VIEW_HOST}:{API_PORT}/docs"
    _banner([f"ControlPlane.ai API   ->  http://{VIEW_HOST}:{API_PORT}", f"Swagger docs         ->  {docs}"])
    if auto_open:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(docs, f"http://{VIEW_HOST}:{API_PORT}/health"),
            daemon=True,
        ).start()
    return subprocess.Popen(
        [PY, "-m", "uvicorn", "api.app:app", "--host", BIND_ADDR, "--port", API_PORT],
        cwd=HERE,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ControlPlane.ai launcher")
    parser.add_argument("--api", action="store_true", help="start only the FastAPI backend")
    parser.add_argument("--both", action="store_true", help="start API and UI together")
    parser.add_argument("--demo", action="store_true", help="run the end-to-end demo and exit")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser window")
    args = parser.parse_args()

    if args.demo:
        subprocess.run([PY, os.path.join(HERE, "demo.py"), "--all"], check=True, cwd=HERE)
        return

    _ensure_data()
    auto_open = not args.no_open

    procs: list[subprocess.Popen] = []
    try:
        if args.api:
            procs.append(start_api(auto_open))
        elif args.both:
            procs.append(start_api(auto_open))
            time.sleep(2)
            procs.append(start_ui(auto_open=False))
        else:
            procs.append(start_ui(auto_open))
        for proc in procs:
            proc.wait()
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        for proc in procs:
            proc.terminate()


if __name__ == "__main__":
    main()
