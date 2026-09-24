"""Register a pixi environment as a Jupyter kernel that activates itself.

Shared by the environments under pixi/.  Run it through each one's
``register-kernel`` task, which sets the working directory to that
environment's manifest directory.

Not ``python -m ipykernel install``: that writes argv from sys.executable,
the bare interpreter, with no activation.  Such a kernel works under a server
started with ``pixi run``, whose PATH it inherits, and dies under VS Code,
which launches the kernelspec directly -- the environment's Library/bin never
reaches PATH, scipy cannot load its delay-loaded BLAS, and the kernel aborts
with 0xc06d007f on ``import skimage``.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from jupyter_client.kernelspec import KernelSpecManager

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--name", required=True, help="kernel name (directory)")
parser.add_argument(
    "--display-name", required=True, help="name shown in pickers"
)
parser.add_argument(
    "--manifest",
    default="pixi.toml",
    help="pixi.toml to activate, relative to the working directory",
)
args = parser.parse_args()

manifest = Path(args.manifest).resolve()
if not manifest.exists():
    sys.exit(f"no manifest at {manifest}")

spec = {
    "argv": [
        "pixi",
        "run",
        "--manifest-path",
        str(manifest),
        "python",
        "-Xfrozen_modules=off",
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ],
    "display_name": args.display_name,
    "language": "python",
    "metadata": {"debugger": True},
}

with tempfile.TemporaryDirectory() as tmp:
    staged = Path(tmp) / args.name
    staged.mkdir()
    (staged / "kernel.json").write_text(
        json.dumps(spec, indent=1), encoding="utf-8"
    )
    dest = KernelSpecManager().install_kernel_spec(
        str(staged), kernel_name=args.name, user=True
    )

print(f"Registered {args.display_name!r} as {args.name!r} in {dest}")
print(f"Activates via: {manifest}")
