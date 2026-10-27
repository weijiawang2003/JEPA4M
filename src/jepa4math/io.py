"""Small reproducibility utilities; no hidden experiment state."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def arrays_hash(arrays):
    h = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        a = np.ascontiguousarray(value)
        h.update(key.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def code_hash():
    h = hashlib.sha256()
    for folder in ("src", "scripts"):
        for p in sorted((ROOT / folder).rglob("*.py")):
            h.update(str(p.relative_to(ROOT)).encode()); h.update(p.read_bytes())
    return h.hexdigest()

def provenance():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            text=True, capture_output=True)
    return {"git_commit": result.stdout.strip() or None,
            "source_sha256": code_hash(), "python": platform.python_version(),
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "torch": str(torch.__version__),
            "numpy": np.__version__, "mps_available": torch.backends.mps.is_available()}

def seed_all(seed):
    np.random.seed(seed); torch.manual_seed(seed)

def sync(device):
    if str(device) == "mps":
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize()

