import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
DEFAULT_VENV = PROJECT_ROOT / ".venv"
REQUIREMENTS_TXT = PROJECT_ROOT / "requirements.txt"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "yolo11-obb.pt"
PUBLIC_DIR = PROJECT_ROOT / "public"
OUTPUT_DIR = PROJECT_ROOT / "output_data" / "results_obb"
JOBS_DIR = PROJECT_ROOT / "jobs"
SERVER_PY = PROJECT_ROOT / "server.py"

DEFAULT_REQUIREMENTS = """
fastapi
uvicorn
opencv-python-headless
numpy
rasterio
ultralytics
matplotlib
pandas
""".strip() + "\n"


def run(cmd, env=None, cwd=None):
    print("\n$", " ".join(map(str, cmd)))
    proc = subprocess.Popen(cmd, env=env, cwd=cwd)
    proc.communicate()
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def venv_paths(venv_dir: Path):
    if platform.system().lower().startswith("win"):
        py = venv_dir / "Scripts" / "python.exe"
        pip = venv_dir / "Scripts" / "pip.exe"
    else:
        py = venv_dir / "bin" / "python"
        pip = venv_dir / "bin" / "pip"
    return py, pip


def ensure_requirements_file():
    if not REQUIREMENTS_TXT.exists():
        print(f"[i] requirements.txt not found. Creating a default one at {REQUIREMENTS_TXT}…")
        REQUIREMENTS_TXT.write_text(DEFAULT_REQUIREMENTS)


def ensure_dirs():
    for d in [MODEL_DIR, PUBLIC_DIR, OUTPUT_DIR, JOBS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    # placeholder UI if missing
    index_html = PUBLIC_DIR / "index.html"
    if not index_html.exists():
        print(f"[i] public/index.html missing. Creating a minimal placeholder UI at {index_html}…")
        index_html.write_text("""
<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Server Up</title></head>
<body><h1>Server running</h1><p>Place your UI at <code>public/index.html</code>.</p></body></html>
""")


def ensure_model_notice():
    if not MODEL_PATH.exists():
        print("[!] Model file not found:", MODEL_PATH)
        print("    • Put your model at this path or adjust MODEL_PATH in server.py")


def create_or_rebuild_venv(venv_dir: Path, rebuild: bool):
    if rebuild and venv_dir.exists():
        print(f"[i] Removing existing venv at {venv_dir}…")
        shutil.rmtree(venv_dir, ignore_errors=True)
    if not venv_dir.exists():
        print(f"[i] Creating virtual environment at {venv_dir}…")
        run([sys.executable, "-m", "venv", str(venv_dir)])


def install_requirements(venv_dir: Path, upgrade: bool):
    py, pip = venv_paths(venv_dir)
    # Upgrade pip tooling first
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    # Install project requirements
    args = [str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS_TXT)]
    if upgrade:
        args.insert(5, "--upgrade")  # after 'install'
    run(args)


def run_server(venv_dir: Path, port: int, host: str = "0.0.0.0", reload: bool = True):
    py, _ = venv_paths(venv_dir)
    if not SERVER_PY.exists():
        print(f"[!] server.py not found at {SERVER_PY}. Make sure you've added the backend file.")
        raise SystemExit(1)
    # Run uvicorn against server:app so reload picks changes
    cmd = [str(py), "-m", "uvicorn", "server:app", "--host", host, "--port", str(port)]
    if reload:
        cmd.append("--reload")
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    print(f"[✓] Launching server on http://{host}:{port}/ …")
    run(cmd, env=env, cwd=str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Bootstrap, install, and run the FastAPI server")
    p.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    p.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    p.add_argument("--venv", type=str, default=str(DEFAULT_VENV), help="Path to virtualenv (default: .venv)")
    p.add_argument("--upgrade", action="store_true", help="Upgrade packages if already installed")
    p.add_argument("--rebuild-venv", action="store_true", help="Delete and recreate venv before install")
    p.add_argument("--no-reload", action="store_true", help="Disable uvicorn auto-reload")
    return p.parse_args()


def main():
    args = parse_args()
    venv_dir = Path(args.venv)

    print("[1/5] Ensuring folders & UI…")
    ensure_dirs()

    print("[2/5] Ensuring requirements.txt…")
    ensure_requirements_file()

    print("[3/5] Creating/updating virtual environment…")
    create_or_rebuild_venv(venv_dir, rebuild=args.rebuild_venv)

    print("[4/5] Installing dependencies…")
    install_requirements(venv_dir, upgrade=args.upgrade)

    print("[5/5] Pre-flight checks…")
    ensure_model_notice()

    # Run the server
    run_server(venv_dir, port=args.port, host=args.host, reload=not args.no_reload)


if __name__ == "__main__":
    main()
