"""The Environments panel: which model environments exist, and building them.

Each scikit-ops model runs in an environment of its own, built on first use
-- minutes and gigabytes, at whatever moment someone first clicks Segment or
Train. This panel shows each one's state ahead of time, and builds on demand:

* **Ready** -- built from the recipe now installed.
* **Needs rebuild** -- built from an older recipe. The next use rebuilds it,
  as a new release with a changed recipe requires.
* **Not built** -- the next use builds it from scratch.

Listed are the environments the installed op collections bring (every
``opspec.collections`` entry point: scikit-ops, tnia-ops ...): one row each,
however many segmenters share it. Anything else in appose's directory -- old
folders, Fiji's -- is under "Show other environments on disk", with a filter,
and can be deleted from here. "Clean download cache" deletes what pixi's
package cache holds that no environment uses any more, e.g. the torch a
rebuild replaced (see ``utilities.pixi_cache``).

It runs on the segmenters' runner and hears their build news, so a build
started here shows in the AI Lab's log, and one started by Segment or Train
shows here. "Used by" names the registered segmenters.

This is a copy of napari-skopkit's Environments panel (its
``_environments.py``), not an import of it: the AI Lab should not pull in
napari-skopkit and everything it brings. The two differ only in the hooks
under "Hooks" below and in wording; a fix to the rest belongs in both.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
from pathlib import Path

from qtpy.QtCore import QThread, QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import ensure_main_thread

from ..utilities import pixi_cache as _cache

READY, STALE, MISSING, OTHER = "Ready", "Needs rebuild", "Not built", "On disk"
BUILDING, FAILED = "Building…", "Failed"
_STATUS = {"ready": READY, "stale": STALE, "missing": MISSING}

#: "[environment build] Installing conda packages 12/40": the line format the
#: AI Lab's build reporter (SkopSegmenter._report) produces.
_PROGRESS = re.compile(r"\[environment build\] (.*?)(?: (\d+)/(\d+))?$")

#: How many op names "Used by" shows before it counts the rest.
_SHOWN_USERS = 4


def _appose_root() -> Path | None:
    try:
        from appose.util.filepath import appose_envs_dir
    except ImportError:
        try:
            from appose.util.environment import appose_envs_dir
        except ImportError:
            return None
    return Path(appose_envs_dir())


def _size(folder: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(folder):
        for f in files:
            with contextlib.suppress(OSError):
                total += os.lstat(os.path.join(root, f)).st_size
    return total


def _human(nbytes: int) -> str:
    size = float(nbytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _shorten(names: list[str]) -> str:
    if len(names) <= _SHOWN_USERS:
        return ", ".join(names)
    return f"{', '.join(names[:_SHOWN_USERS])} (+{len(names) - _SHOWN_USERS} more)"


def _collections() -> dict[str, str]:
    """Every installed collection of ops: its name, and the package of its ops."""
    import importlib.metadata

    return {
        entry.name: entry.value
        for entry in importlib.metadata.entry_points(
            group="opspec.collections"
        )
    }


def _users() -> dict[str, list[str]]:
    """Environment name -> the registered segmenters that run in it."""
    import sys

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


# -- threads ----------------------------------------------------------------


class _BuildThread(QThread):
    """Builds one environment off the GUI thread."""

    done = Signal(str, bool, str)  # env name, ok, message

    def __init__(self, runner, env):
        super().__init__()
        self.runner = runner
        self.env = env

    def run(self):
        try:
            self.runner.build(self.env)
            self.done.emit(self.env.name, True, "")
        except Exception as e:  # noqa: BLE001 - reported in the log
            self.done.emit(self.env.name, False, str(e))


class _SizeThread(QThread):
    """Adds up folder sizes in the background: a torch stack is 100k files."""

    sized = Signal(str, str)  # env name, size text

    def __init__(self, folders: dict[str, Path]):
        super().__init__()
        self.folders = folders

    def run(self):
        for name, folder in self.folders.items():
            if self.isInterruptionRequested():
                return
            if folder.is_dir():
                self.sized.emit(name, _human(_size(folder)))


class _PruneScanThread(QThread):
    """Finds the cache entries no environment uses: reads every file's links."""

    found = Signal(object)  # list[_cache.Entry]

    def __init__(self, pkgs: Path):
        super().__init__()
        self.pkgs = pkgs

    def run(self):
        entries = _cache.find_prunable(self.pkgs, self.isInterruptionRequested)
        self.found.emit(entries)


class _PruneThread(QThread):
    """Deletes cache entries."""

    done = Signal(int, object)  # bytes freed, list of error strings

    def __init__(self, entries):
        super().__init__()
        self.entries = entries

    def run(self):
        freed, errors = _cache.delete(self.entries)
        self.done.emit(freed, errors)


# -- the panel --------------------------------------------------------------


class EnvironmentsPanel(QWidget):
    """Status of every model environment, and a button to build each."""

    COLUMNS = ("Environment", "Used by", "Status", "Size")

    def __init__(self, viewer=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self._envs: dict[str, object] = {}  # name -> Env, or None if "other"
        self._failed: dict[str, str] = {}
        self._building: dict[str, _BuildThread] = {}
        self._size_thread = None
        self._prune_thread = None
        self._setup_ui()
        self.listen(ensure_main_thread(self._on_build_news))
        self.refresh()

    # Hooks ------------------------------------------------------------------
    # Where this copy differs from napari-skopkit's: the AI Lab's plumbing.

    def runner(self):
        """The segmenters' runner, whose builds this panel starts and hears."""
        from ..Segmenters.GlobalSegmenters.SkopSegmenter import _get_runner

        return _get_runner()

    def collections(self) -> dict[str, str]:
        """Collection name -> the package of its ops, e.g. "skop.ops"."""
        return _collections()

    def users(self) -> dict[str, list[str]]:
        """Environment name -> the registered segmenters that run in it.

        Blank until an AI Lab has registered its segmenters.
        """
        return _users()

    def listen(self, on_line) -> None:
        """Have *on_line* hear every build's news, one line of text at a time.

        The AI Lab's build news outlives any one runner: close_runner() makes
        a new one, and this keeps hearing it. ``on_line`` already hops to the
        GUI thread, so it may be called from any thread.
        """
        from ..Segmenters.GlobalSegmenters.SkopSegmenter import (
            add_build_listener,
        )

        add_build_listener(on_line)

    def say(self, line: str) -> None:
        """Announce *line*: to the terminal and every listener, this one too."""
        from ..Segmenters.GlobalSegmenters.SkopSegmenter import _report

        _report(line)

    def release_workers(self) -> None:
        """Stop the workers that may hold an environment's files open, before
        a delete."""
        from ..Segmenters.GlobalSegmenters.SkopSegmenter import close_runner

        close_runner()

    # UI ---------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Each model runs in an environment of its own, built on first "
            "use: minutes and gigabytes, once. Build them here ahead of "
            "time, rebuild one a new release has changed, or delete old ones."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        layout.addWidget(self.table, 1)

        others = QHBoxLayout()
        self.show_other = QCheckBox("Show other environments on disk")
        self.show_other.toggled.connect(self._apply_filter)
        others.addWidget(self.show_other)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("filter by name")
        self.filter_edit.textChanged.connect(self._apply_filter)
        others.addWidget(self.filter_edit, 1)
        layout.addLayout(others)

        buttons = QHBoxLayout()
        self.build_btn = QPushButton("Build")
        self.build_btn.clicked.connect(self._on_build)
        self.log_btn = QPushButton("Show log")
        self.log_btn.clicked.connect(self._on_show_log)
        self.folder_btn = QPushButton("Open folder")
        self.folder_btn.clicked.connect(self._on_open_folder)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._on_delete)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        for b in (
            self.build_btn,
            self.log_btn,
            self.folder_btn,
            self.delete_btn,
            refresh_btn,
        ):
            buttons.addWidget(b)
        layout.addLayout(buttons)

        self.prune_btn = QPushButton("Clean download cache")
        self.prune_btn.setToolTip(
            "Delete the packages pixi's download cache holds that no "
            "environment uses any more -- e.g. the torch a rebuild replaced. "
            "Environments keep working."
        )
        self.prune_btn.clicked.connect(self._on_prune)
        layout.addWidget(self.prune_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        layout.addWidget(self.log_view, 1)

    def _log(self, line: str) -> None:
        self.log_view.appendPlainText(line)

    # Contents ---------------------------------------------------------------

    def refresh(self):
        """Re-read every environment's state from disk."""
        try:
            runner = self.runner()
        except Exception as e:  # noqa: BLE001 - scikit-ops missing
            self._log(f"No op runner available: {e}")
            return
        try:
            users = self.users()
        except Exception:  # noqa: BLE001 - informational only
            users = {}

        self._envs = {}
        for package in self.collections().values():
            for env in runner.collection_envs(package):
                self._envs.setdefault(env.name, env)
        root = _appose_root()
        if root is not None and root.is_dir():
            for folder in sorted(root.iterdir()):
                if folder.is_dir() and not folder.name.startswith("."):
                    self._envs.setdefault(folder.name, None)

        # Ours first, the ones in use first among them; then the rest.
        def order(name):
            env = self._envs[name]
            return (env is None, name not in users, name)

        names = sorted(self._envs, key=order)
        selected = self._selected_name()
        self.table.setRowCount(len(names))
        for row, name in enumerate(names):
            env = self._envs[name]
            used = users.get(name, [])
            self.table.setItem(row, 0, QTableWidgetItem(name))
            item = QTableWidgetItem(_shorten(used))
            if used:
                item.setToolTip("\n".join(used))
            self.table.setItem(row, 1, item)
            self.table.setItem(row, 2, QTableWidgetItem(self._status_of(name)))
            self.table.setItem(row, 3, QTableWidgetItem(""))
            if env is not None:
                self.table.item(row, 0).setToolTip(str(env.config))
            if name == selected:
                self.table.selectRow(row)
        self._apply_filter()
        self._update_buttons()
        self._start_sizes()

    def _status_of(self, name) -> str:
        if name in self._building:
            return BUILDING
        env = self._envs.get(name)
        if env is None:
            return OTHER
        if name in self._failed:
            return FAILED
        return _STATUS.get(env.status, MISSING)

    def _start_sizes(self):
        if self._size_thread is not None:
            self._size_thread.requestInterruption()
            self._size_thread.wait(2000)
        root = _appose_root()
        if root is None:
            return
        self._size_thread = _SizeThread({n: root / n for n in self._envs})
        self._size_thread.sized.connect(self._on_sized)
        self._size_thread.start()

    def _on_sized(self, name, text):
        row = self._row_of(name)
        if row is not None:
            self.table.item(row, 3).setText(text)

    def _apply_filter(self, *_):
        text = self.filter_edit.text().strip().lower()
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text()
            other = self._envs.get(name) is None
            hidden = (other and not self.show_other.isChecked()) or (
                text and text not in name.lower()
            )
            self.table.setRowHidden(row, bool(hidden))

    def _row_of(self, name):
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == name:
                return row
        return None

    def _selected_name(self):
        rows = self.table.selectionModel().selectedRows()
        return self.table.item(rows[0].row(), 0).text() if rows else None

    def _set_status(self, name, status):
        row = self._row_of(name)
        if row is not None:
            self.table.item(row, 2).setText(status)

    def _update_buttons(self):
        name = self._selected_name()
        env = self._envs.get(name) if name else None
        status = self._status_of(name) if name else None
        self.build_btn.setEnabled(env is not None and status != BUILDING)
        self.build_btn.setText(
            "Rebuild" if status in (READY, STALE, FAILED) else "Build"
        )
        self.log_btn.setEnabled(env is not None)
        root = _appose_root()
        on_disk = (
            name is not None and root is not None and (root / name).is_dir()
        )
        self.folder_btn.setEnabled(on_disk)
        self.delete_btn.setEnabled(on_disk and status != BUILDING)

    # Actions ----------------------------------------------------------------

    def _on_build(self):
        name = self._selected_name()
        env = self._envs.get(name)
        if env is None or name in self._building:
            return
        if env.status == "ready" and not self._confirm(
            "Rebuild",
            f"'{name}' is already built from the current recipe. "
            "Rebuilding re-runs pixi, which installs nothing new. Go ahead?",
        ):
            return
        self._failed.pop(name, None)
        self.say(f"Building the '{name}' environment, from {env.config}")
        thread = _BuildThread(self.runner(), env)
        thread.done.connect(self._on_build_done)
        self._building[name] = thread
        self._set_status(name, BUILDING)
        self._update_buttons()
        thread.start()

    def _on_build_done(self, name, ok, message):
        thread = self._building.pop(name, None)
        if thread is not None:
            thread.wait()
        if ok:
            self.say(f"✅ The '{name}' environment is built and ready")
            self.progress_bar.setValue(100)
        else:
            self._failed[name] = message
            # oprunner's BuildFailed names the environment and the log.
            self.say(f"❌ {message}")
        self.refresh()

    def _on_build_news(self, line):
        """A line from any build: log it, and move the bar if it says how far."""
        self._log(line)
        found = _PROGRESS.match(line.strip())
        if found and found.group(3):
            current, total = int(found.group(2)), int(found.group(3))
            if total:
                self.progress_bar.setValue(int(100 * current / total))

    def _on_show_log(self):
        env = self._envs.get(self._selected_name())
        if env is None:
            return
        if env.log.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(env.log)))
        else:
            self._log(f"No build log for '{env.name}' yet.")

    def _on_open_folder(self):
        name = self._selected_name()
        root = _appose_root()
        if name and root is not None and (root / name).is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(root / name)))

    def _on_delete(self):
        name = self._selected_name()
        root = _appose_root()
        if not name or root is None or name in self._building:
            return
        folder = root / name
        if not folder.is_dir() or not self._confirm(
            "Delete environment",
            f"Delete {folder}?\n\nThe next use builds it again from scratch, "
            "which takes minutes and downloads gigabytes. Running models are "
            "stopped first.",
        ):
            return
        self.release_workers()  # a running worker holds its environment's files
        try:
            shutil.rmtree(folder)
            self._log(f"Deleted {folder}")
        except OSError as e:
            self._log(f"Could not delete {folder}: {e}")
        self.refresh()

    def _on_prune(self):
        if self._building:
            self._log("Wait for the build to finish: it is using the cache.")
            return
        pkgs = _cache.conda_pkgs_dir()
        if pkgs is None:
            self._log("No pixi package cache found.")
            return
        self._log(f"Looking for unused packages in {pkgs} ...")
        self.prune_btn.setEnabled(False)
        self._prune_thread = _PruneScanThread(pkgs)
        self._prune_thread.found.connect(self._on_prune_found)
        self._prune_thread.start()

    def _on_prune_found(self, entries):
        self._prune_thread = None
        self.prune_btn.setEnabled(True)
        if not entries:
            self._log("Nothing to clean: every cached package is in use.")
            return
        unused = [e for e in entries if e.reason == _cache.UNUSED]
        unfinished = [e for e in entries if e.reason == _cache.UNFINISHED]
        total = sum(e.size for e in entries)
        biggest = sorted(entries, key=lambda e: e.size, reverse=True)[:8]
        lines = "\n".join(
            f"  {_human(e.size):>8}  {e.path.name}" for e in biggest
        )
        text = (
            f"Delete {len(unused)} unused and {len(unfinished)} unfinished "
            f"packages, {_human(total)}, from\n{entries[0].path.parent}?\n\n"
            f"Largest:\n{lines}\n\n"
            "Environments keep working. A later build downloads again what "
            "it needs. Close other pixi builds first."
        )
        if not self._confirm("Clean download cache", text):
            return
        self.prune_btn.setEnabled(False)
        self._prune_thread = _PruneThread(entries)
        self._prune_thread.done.connect(self._on_pruned)
        self._prune_thread.start()

    def _on_pruned(self, freed, errors):
        self._prune_thread = None
        self.prune_btn.setEnabled(True)
        self._log(f"Cleaned the download cache: {_human(freed)} freed.")
        for error in errors[:10]:
            self._log(f"Could not delete {error}")
        if len(errors) > 10:
            self._log(f"... and {len(errors) - 10} more")

    def _confirm(self, title, text) -> bool:
        answer = QMessageBox.question(
            self, title, text, QMessageBox.Yes | QMessageBox.No
        )
        return answer == QMessageBox.Yes

    def closeEvent(self, event):
        if self._size_thread is not None:
            self._size_thread.requestInterruption()
        if isinstance(self._prune_thread, _PruneScanThread):
            self._prune_thread.requestInterruption()
        super().closeEvent(event)
