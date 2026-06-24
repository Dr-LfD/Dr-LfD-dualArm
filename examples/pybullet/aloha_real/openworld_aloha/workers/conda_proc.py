import os
import subprocess
import sys
from pathlib import Path


def launch_conda_worker(env_name: str, worker_script: str) -> subprocess.Popen:
    conda_bin = Path.home() / "miniforge3" / "bin" / "conda"
    if not conda_bin.is_file():
        raise FileNotFoundError(f"Conda executable not found: {conda_bin}")
    worker_script = os.path.abspath(worker_script)
    if not os.path.isfile(worker_script):
        raise FileNotFoundError(f"Worker script does not exist: {worker_script}")
    return subprocess.Popen(
        [str(conda_bin), "run", "--no-capture-output", "-n", env_name, "python", worker_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        bufsize=0,
    )
