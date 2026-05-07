"""
Multi-GPU launcher for WorldMirror Gradio on Windows.

Monkey-patches TCPStore to disable libuv BEFORE torchrun starts,
then invokes torchrun programmatically.
"""

import os
import sys

# ── Patch TCPStore *before* any torch.distributed code runs ──
import torch.distributed as _dist
_OrigInit = _dist.TCPStore.__init__

def _patched_init(self, *args, **kwargs):
    kwargs.setdefault("use_libuv", False)
    _OrigInit(self, *args, **kwargs)

_dist.TCPStore.__init__ = _patched_init
print("[launch] TCPStore patched: use_libuv defaults to False")

# ── Now invoke torchrun ──
from torch.distributed.run import main as torchrun_main

if __name__ == "__main__":
    # Replace bare "python" with full interpreter path so torchrun can find it
    for i, arg in enumerate(sys.argv):
        if arg == "python":
            sys.argv[i] = sys.executable
            print(f"[launch] Replaced 'python' with {sys.executable}")
            break
    sys.exit(torchrun_main())
