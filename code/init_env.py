import subprocess
import sys
import shutil
import os
from pathlib import Path

def run(cmd, cwd=None):
    print(f"→ {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=cwd)

def main():
    root_dir = Path(__file__).resolve().parent.parent  # go up from /code to project root
    print(f"Project root: {root_dir}")

    # 1. Check poetry
    if shutil.which("poetry") is None:
        print("Poetry not found. Installing with pip...")
        run(f"{sys.executable} -m pip install --user poetry")
    else:
        print("Poetry already installed.")

    # 2. Install deps at project root
    print("Running poetry install...")
    run("poetry install", cwd=root_dir)

    # 3. Get poetry’s env path
    env_path = subprocess.check_output(
        "poetry env info --path", shell=True, text=True, cwd=root_dir
    ).strip()
    python_bin = os.path.join(env_path, "bin", "python")

    print(f"\nEnvironment ready: {env_path}")
    print("To use it in this shell, run:")
    print(f"  source {env_path}/bin/activate")
    print("\nOr run Python directly with:")
    print(f"  {python_bin} your_script.py")

if __name__ == "__main__":
    main()
