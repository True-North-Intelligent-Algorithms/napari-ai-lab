"""The Environments panel, as the AI Lab shows it.

The panel itself is napari-skopkit's (``napari_skopkit.EnvironmentsPanel``):
which op environments exist, their state, building and deleting them. What
the AI Lab adds is its own plumbing, through the panel's hooks:

* its runner -- the one the segmenters run on -- so a build started by
  Segment or Train shows here, and one started here shows in the AI Lab's log;
* "Used by" naming the registered segmenters that run in each environment,
  rather than ops.

Without napari-skopkit installed there is no panel; ``EnvironmentsPanel`` is
None and the AI Lab says how to get it.
"""

from __future__ import annotations

import sys

try:
    from napari_skopkit import EnvironmentsPanel as _SkopkitPanel
except ImportError:  # napari-skopkit missing, or older than the panel
    _SkopkitPanel = None


def _users() -> dict[str, list[str]]:
    """Environment name -> the registered segmenters that run in it."""
    from opspec.op import OpSpec

    from ..Segmenters.GlobalSegmenters import GlobalSegmenterBase
    from ..Segmenters.GlobalSegmenters.SkopSegmenter import _get_runner
    from ..Segmenters.InteractiveSegmenters import InteractiveSegmenterBase

    runner = _get_runner()
    found: dict[str, list[str]] = {}
    registries = (
        GlobalSegmenterBase.registry,
        InteractiveSegmenterBase.registry,
    )
    for registry in registries:
        for name, cls in list(registry.items()):
            # Only segmenters that run an op: making one of the others could
            # load a model in this process.
            if isinstance(getattr(cls, "op", None), property):
                try:
                    op = cls().op
                except Exception:  # noqa: BLE001 - informational only
                    op = None
            else:
                # SAM3D's ops are module-level, used on its scikit-ops path.
                module = sys.modules.get(cls.__module__)
                op = getattr(module, "microsam_embed", None)
            if op is None:
                continue
            try:
                if OpSpec.from_op(op).env is None:
                    continue
                env = runner.env_for(op)
            except Exception:  # noqa: BLE001 - informational only
                continue
            found.setdefault(env.name, []).append(name)
    return found


if _SkopkitPanel is None:
    EnvironmentsPanel = None
else:

    class EnvironmentsPanel(_SkopkitPanel):
        """napari-skopkit's panel, on the AI Lab's runner and segmenters."""

        def runner(self):
            from ..Segmenters.GlobalSegmenters.SkopSegmenter import _get_runner

            return _get_runner()

        def users(self) -> dict[str, list[str]]:
            # Blank until an AI Lab has registered its segmenters.
            return _users()

        def listen(self, on_line) -> None:
            # The AI Lab's build news outlives any one runner: close_runner()
            # makes a new one, and this keeps hearing it.
            from ..Segmenters.GlobalSegmenters.SkopSegmenter import (
                add_build_listener,
            )

            add_build_listener(on_line)

        def say(self, line: str) -> None:
            from ..Segmenters.GlobalSegmenters.SkopSegmenter import _report

            _report(line)  # to the terminal and every listener, this one too

        def release_workers(self) -> None:
            from ..Segmenters.GlobalSegmenters.SkopSegmenter import (
                close_runner,
            )

            close_runner()
