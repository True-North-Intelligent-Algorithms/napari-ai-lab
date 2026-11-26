"""pixi's conda package cache: what in it no environment uses any more.

Every environment build downloads its packages into one cache shared by all
pixi environments, and installs them as hard links to the cached files. A
package an environment uses therefore costs its space once. But the cache
never forgets: when a rebuild swaps torch 2.12 for 2.13, the environment
drops 2.12 and the cache keeps it, now as the only copy, for good. pixi can
clear the whole cache (``pixi clean cache --conda``) but not just that.

This finds that. A cached package is *unused* when none of its files has a
second link -- no environment on this drive holds it. A folder whose name
starts with a dot is an unpack that never finished, e.g. from a build that
ran out of disk, and is never used.

Deleting an entry is safe either way: an environment keeps its files, as a
hard link survives the other name's deletion. The worst case is a later
build downloading a package again. That worst case is also the whole cache
where pixi copies rather than links -- an environment on another drive, or
a file system that clones (APFS) -- so there every entry looks unused.

A copy of napari-skopkit's ``_cache``, so the AI Lab needs no napari-skopkit:
a fix in one belongs in the other.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

#: Why an entry can go.
UNUSED, UNFINISHED = "unused", "unfinished"


@dataclass(frozen=True)
class Entry:
    path: Path
    size: int
    reason: str


def conda_pkgs_dir() -> Path | None:
    """Where pixi keeps downloaded conda packages, or None if there is none."""
    for var in ("RATTLER_CACHE_DIR", "PIXI_CACHE_DIR"):
        if os.environ.get(var):
            root = Path(os.environ[var])
            break
    else:
        if sys.platform == "win32":
            base = Path(
                os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")
            )
        elif sys.platform == "darwin":
            base = Path.home() / "Library" / "Caches"
        else:
            base = Path(
                os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
            )
        root = base / "rattler" / "cache"
    pkgs = root / "pkgs"
    return pkgs if pkgs.is_dir() else None


def _files(folder: Path):
    for root, _dirs, files in os.walk(folder):
        for f in files:
            yield os.path.join(root, f)


def _unused_size(folder: Path) -> int | None:
    """The folder's size if no file in it has a second link, else None."""
    total = 0
    for path in _files(folder):
        try:
            st = os.lstat(path)
        except OSError:
            continue
        if st.st_nlink > 1:
            return None
        total += st.st_size
    return total


def _size(folder: Path) -> int:
    total = 0
    for path in _files(folder):
        with contextlib.suppress(OSError):
            total += os.lstat(path).st_size
    return total


def find_prunable(pkgs: Path, stop=lambda: False) -> list[Entry]:
    """Every cache entry in *pkgs* that no environment uses.

    *stop* is polled between entries, so a caller on a thread can give up.
    """
    found = []
    for folder in sorted(pkgs.iterdir()):
        if stop():
            break
        if not folder.is_dir():
            continue
        if folder.name.startswith("."):
            found.append(Entry(folder, _size(folder), UNFINISHED))
            continue
        size = _unused_size(folder)
        if size is not None:
            found.append(Entry(folder, size, UNUSED))
    return found


def delete(entries: list[Entry]) -> tuple[int, list[str]]:
    """Delete *entries* and their lock files. Returns bytes freed and errors."""
    freed, errors = 0, []
    for entry in entries:
        try:
            shutil.rmtree(entry.path)
        except OSError as e:
            errors.append(f"{entry.path.name}: {e}")
            continue
        freed += entry.size
        lock = entry.path.with_name(entry.path.name + ".lock")
        if lock.is_file():
            with contextlib.suppress(OSError):
                lock.unlink()
    return freed, errors
