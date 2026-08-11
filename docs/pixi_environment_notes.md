# Pixi environment gotchas

Things that cost an afternoon while building environments under `pixi/`, kept
so they cost minutes the next time. Each entry: what it looked like, what it
was, what to do.

## Conda cellpose installs a second, broken OpenCV

**Symptom.** A global segmenter reports "Dependencies not available" for a
package that is clearly installed. Importing it directly at a Python prompt
works. Importing it *after* napari does not:

```
ImportError: .../cv2/python-3.12/../../../../libgobject-2.0.so.0:
undefined symbol: g_pointer_bit_unlock_and_set
```

**Cause.** conda-forge's `cellpose` is a noarch package, so it installs the
original PyPI `dist-info`, which declares:

```
Requires-Dist: opencv-python-headless
```

pixi's PyPI resolver (uv) reads that and cannot tell that conda's `py-opencv`
already provides `cv2` — the distribution names do not match — so it installs
the PyPI opencv wheel *on top of* the conda one. Two `cv2` implementations then
compete, and whichever loses the import race resolves
`libgobject-2.0.so.0` out of the other's tree, against a `libglib` that does
not export the symbol it wants.

Import order decides which one wins, which is why the failure looks
intermittent and why testing `import cellpose` on its own says everything is
fine.

Conda's opencv is additionally a **qt6** build, so it also puts a second Qt into
an environment running PyQt5.

**Confirm it** by listing both sources:

```sh
ls .pixi/envs/default/conda-meta/ | grep -iE "opencv|glib|cellpose"
ls -d .pixi/envs/default/lib/python3.*/site-packages/opencv*.dist-info
```

Entries from both is the diagnosis.

**Fix.** Take cellpose from `[pypi-dependencies]` rather than `[dependencies]`.
Then opencv comes from PyPI only, there is one `cv2`, and no stray Qt6. Torch
stays on conda, where the CUDA builds are; cellpose finds it there.

Done in `pixi/pytorch_napari/pixi.toml`. **`pixi/microsam_cellposesam_czi`
still takes cellpose from conda** and is presumably carrying the same duplicate
— worth checking if anything there behaves oddly around OpenCV.

**The general shape**, worth remembering beyond this case: a noarch conda
package carries PyPI metadata, and any dependency in it whose *distribution
name* differs from the conda package providing it will be installed twice, once
from each ecosystem. Native libraries then collide. It is not specific to
cellpose or to OpenCV.
