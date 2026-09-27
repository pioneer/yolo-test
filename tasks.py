import os
import platform
from pathlib import Path

from invoke import Exit, task

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "yolo26n.pt"
EXPECTED_GPU = "NVIDIA GeForce GTX 1050 Ti"
EXPECTED_CC = (6, 1)


def require_cuda():
    import torch

    if not torch.cuda.is_available():
        raise Exit("ERROR: CUDA is not available to PyTorch.", code=1)
    return torch


@task
def sync(c):
    """Install locked dependencies with uv."""
    with c.cd(str(ROOT)):
        c.run("uv sync", pty=True)


@task(help={"model": f"Ultralytics asset name (default: {MODEL.name})"})
def download(c, model=MODEL.name):
    """Download a YOLO model into the project dir (skips if already present)."""
    from ultralytics.utils.downloads import attempt_download_asset

    path = ROOT / Path(model).name
    if path.is_file():
        print(f"Already present: {path}")
        return
    attempt_download_asset(path)
    if not path.is_file():
        raise Exit(f"ERROR: download failed: {model}", code=1)
    print(f"Downloaded: {path}")


@task
def gpu(c):
    """Print PyTorch/CUDA/GPU info; fail if CUDA is unavailable."""
    import torch

    print(f"Python:               {platform.python_version()}")
    print(f"PyTorch:              {torch.__version__}")
    print(f"CUDA runtime:         {torch.version.cuda}")
    print(f"CUDA available:       {torch.cuda.is_available()}")
    print(f"cuDNN:                {torch.backends.cudnn.version()}")
    print(f"CUDA device count:    {torch.cuda.device_count()}")

    torch = require_cuda()
    props = torch.cuda.get_device_properties(0)
    cc = (props.major, props.minor)
    arch_list = torch.cuda.get_arch_list()
    print(f"GPU name:             {props.name}")
    print(f"Compute capability:   {cc[0]}.{cc[1]}")
    print(f"VRAM:                 {props.total_memory / 1024**3:.2f} GiB")
    print(f"Arch list:            {arch_list}")

    if EXPECTED_GPU not in props.name or cc != EXPECTED_CC:
        print(f"NOTE: expected {EXPECTED_GPU} (compute capability "
              f"{EXPECTED_CC[0]}.{EXPECTED_CC[1]}); continuing anyway.")
    # SASS for sm_XY runs on sm_XZ when Z >= Y (e.g. sm_60 kernels on a 6.1 GPU).
    if not any(f"sm_{cc[0]}{m}" in arch_list for m in range(cc[1] + 1)):
        print(f"WARNING: no sm_{cc[0]}x kernels <= sm_{cc[0]}{cc[1]} in this PyTorch build.")


@task
def smoke(c):
    """Run one real YOLO inference on a blank image on CUDA device 0."""
    import numpy as np

    torch = require_cuda()
    if not MODEL.is_file():
        raise Exit(f"ERROR: model not found: {MODEL} (run: uv run inv download)", code=1)

    from ultralytics import YOLO

    print(f"Model:                {MODEL}")
    print(f"Selected GPU:         cuda:0 ({torch.cuda.get_device_name(0)})")

    image = np.zeros((640, 640, 3), dtype=np.uint8)
    try:
        results = YOLO(str(MODEL)).predict(image, device=0, verbose=False)
    except Exception as e:
        raise Exit(f"ERROR: inference failed: {e}", code=1)
    if len(results) != 1:
        raise Exit(f"ERROR: expected 1 result, got {len(results)}", code=1)

    print("Inference:            OK")
    print(f"CUDA mem allocated:   {torch.cuda.memory_allocated(0) / 1024**2:.1f} MiB")
    print(f"CUDA peak allocated:  {torch.cuda.max_memory_allocated(0) / 1024**2:.1f} MiB")


@task(pre=[gpu, smoke])
def check(c):
    """Run gpu and smoke."""
    print("All checks passed.")


@task
def gui(c):
    """Launch the Ultralytics Streamlit live inference UI (app.py: custom default classes)."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    with c.cd(str(ROOT)):
        # Same command `yolo solutions inference` runs, but with app.py as the script.
        c.run(f'streamlit run app.py --server.headless true model="{MODEL.name}"', env=env, pty=True)


@task
def nvidia(c):
    """Run nvidia-smi."""
    c.run("nvidia-smi", pty=True)


@task
def monitor(c):
    """Watch nvidia-smi every 0.5 s (Ctrl+C to exit)."""
    c.run("watch -n 0.5 nvidia-smi", pty=True, warn=True)
