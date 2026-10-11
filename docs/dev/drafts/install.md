# Install

## Pixi (recommended)

Follow installation instructions for pixi that can be found here [pixi installation page](https://pixi.prefix.dev/latest/installation/).

Reopen the terminal afterwards, so `pixi` is on the path.

Then clone the repo:

```sh
git clone https://github.com/bnorthan/i2k-2026.git
```

If you want a quicker install:

```sh
cd i2k-2026/pixi/i2k2026_lite
pixi install
pixi run register-kernel
```

If you want the cool interactive microsam labeling:

```sh
cd i2k-2026/pixi/i2k2026_microsam
pixi install
pixi run register-kernel
```

- microsam is several GB.
- Mostly torch and CUDA.

## pip and conda (fallback)

If pixi fails for you use this.

```sh
conda create -n i2k2026 -c conda-forge python=3.12 micro_sam
conda activate i2k2026
pip install "scikit-ops @ git+https://github.com/apposed/scikit-ops.git"
pip install napari-ai-lab "tnia-python[plotting]" albumentations jupyterlab matplotlib scikit-image tifffile
```

## Did our installation work?

```python
import skop
import napari_ai_lab

print('skop         ', skop.__file__)
print('napari_ai_lab', napari_ai_lab.__file__)
```

## Build the environments

Each op runs in its own environment. They are built on the first call that
needs one, so you can skip this and everything still works — the
first call just takes several minutes instead of seconds.

```python
from skop.runner import Runner

runner = Runner()

ENVIRONMENTS = [
    'minimal',       # toy ops
    'cellpose3',     # Cellpose 3
    'pytorch',       # Cellpose 4, micro-sam
    'stardist-tf',   # StarDist
]

for env in ENVIRONMENTS:
    print(f'{env:14} {runner.environment_status(env)}')
```

- `missing` — never built.
- `stale` — built from a different definition than the one installed now.
- `up to date` — nothing to do.

`ensure_environment` builds only what is missing or stale

```python
runner.ensure_environment('cellpose3')
```
