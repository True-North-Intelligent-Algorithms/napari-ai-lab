# pixi environments

- `install/lite`: for users. napari-ai-lab from PyPI, no micro-sam. Smallest download.
- `install/microsam`: for users. The same plus micro-sam, for interactive SAM. Several GB.
- `dev/editable`: for development. Editable sibling checkouts (scikit-ops,
  opspec, ophelper, oprunner, appose) plus pytest and mkdocs:
  `pixi run test`, `pixi run docs`, `pixi run lab`.

The `install/` files stand alone, so a user can download one `pixi.toml` and
run `pixi run napari`. Models are not installed here; scikit-ops builds each
model's environment on first use.
