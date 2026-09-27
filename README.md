# yolo-test

Ultralytics YOLO live inference on a local NVIDIA GPU (tested target: GTX 1050 Ti, CC 6.1),
using the local `yolo26n.pt` model. Python 3.12, PyTorch 2.14.0 with CUDA 12.6 wheels.

Requires: [uv](https://docs.astral.sh/uv/), an NVIDIA driver supporting CUDA 12.6, and `watch` (procps) for `monitor`.

## Normal workflow

```sh
uv sync              # install locked dependencies
uv run inv download  # fetch yolo26n.pt (default; skipped if already present)
uv run inv check     # GPU info + real CUDA inference smoke test
uv run inv gui       # Ultralytics Streamlit live inference UI
```

## Other tasks

```sh
uv run inv download --model yolo26s.pt  # fetch another Ultralytics model
uv run inv gpu      # Python/PyTorch/CUDA/cuDNN/GPU info; fails if CUDA is unavailable
uv run inv smoke    # one YOLO inference on a blank 640x640 image with device=0
uv run inv nvidia   # nvidia-smi
uv run inv monitor  # watch -n 0.5 nvidia-smi (Ctrl+C to exit)
uv run inv --list   # list all tasks
```

`uv run inv sync` is also available and runs `uv sync`.
