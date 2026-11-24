"""The Environments panel: which model environments exist, and building them.

Each scikit-ops model runs in an environment of its own, built on first use
-- minutes and gigabytes, at whatever moment someone first clicks Segment or
Train. This panel shows each one's state ahead of time, and builds on demand:

* **Ready** -- built from the recipe now installed.
* **Needs rebuild** -- built from an older recipe. The next use rebuilds it,
  as a new scikit-ops release with a changed recipe requires.
* **Not built** -- the next use builds it from scratch.

Listed are the environments the installed op collections bring (scikit-ops,
and tnia-ops if present): one row each, however many ops share it, so a
handful rather than hundreds. Anything else in appose's directory -- old
folders, Fiji's -- is under "Show other environments on disk", with a filter.

Build news goes through the same channel the segmenters use, so a build
started here shows in the AI Lab's log, and one started by Segment or Train
shows here. Each build's full pixi output is kept in a log file beside the
environment (oprunner 0.1.0.dev3 and later), which "Show log" opens.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import sys
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
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..utilities.processor_thread import BuildRelay
from ..utilities.qt_progress_logger import QtProgressLogger

#: Op collections whose environments are listed, by importable module.
COLLECTIONS = ("skop", "tnia_ops")

READY, STALE, MISSING, OTHER = "Ready", "Needs rebuild", "Not built", "On disk"
BUILDING, FAILED = "Building…", "Failed"

#: "[environment build] Installing conda packages 12/40", from _BuildReporter.
_PROGRESS = re.compile(r"\[environment build\] (.*?)(?: (\d+)/(\d+))?$")


# -- the runner, old and new ------------------------------------------------
# oprunner 0.1.0.dev3 has Env.status, Env.log, Runner.collection_envs and
# Runner.build. Older ones have the same machinery under private names, so
# this works with either.


def _runner():
    from ..Segmenters.GlobalSegmenters.SkopSegmenter import _get_runner

    return _get_runner()


def _collection_envs(runner, module):
    if hasattr(runner, "collection_envs"):
        return runner.collection_envs(module)
    import importlib

    try:
        found = Path(importlib.import_module(module).__file__).resolve()
    except (ImportError, TypeError):
        return []
    for folder in found.parents:
        configs = sorted((folder / "envs").glob("*/pixi.toml"))
        if configs:
            return [runner._env_for(c.parent.name, module) for c in configs]
    return []


def _status(env) -> str:
    if hasattr(env, "status"):
        return {"ready": READY, "stale": STALE}.get(env.status, MISSING)
    from oprunner.runner import _is_built

    if not (env.folder / "appose.json").exists():
        return MISSING
    return READY if _is_built(env) else STALE


def _log_path(env) -> Path:
    return getattr(env, "log", env.folder.parent / f"{env.name}.build.log")


def _say(line: str):
    """Build news, to every log that listens -- this panel's included."""
    from ..Segmenters.GlobalSegmenters.SkopSegmenter import _report

    _report(line)


def _appose_root() -> Path | None:
    try:
        from appose.util.filepath import appose_envs_dir
    except ImportError:
        try:
            from appose.util.environment import appose_envs_dir
        except ImportError:
            return None
    return Path(appose_envs_dir())


def _users() -> dict[str, list[str]]:
    """Environment name -> the registered segmenters that run in it."""
    from opspec.op import OpSpec

    from ..Segmenters.GlobalSegmenters import GlobalSegmenterBase
    from ..Segmenters.InteractiveSegmenters import InteractiveSegmenterBase

    runner = _runner()
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


def _size(folder: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(folder):
        for f in files:
            with contextlib.suppress(OSError):
                total += os.lstat(os.path.join(root, f)).st_size
    return total


def _human(nbytes: int) -> str:
    size = float(nbytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit != "GB" else f"{size:.1f} GB"
        size /= 1024
    return ""


# -- threads ----------------------------------------------------------------


class _BuildThread(QThread):
    """Builds one environment off the GUI thread."""

    done = Signal(str, bool, str)  # env name, ok, message

    def __init__(self, env):
        super().__init__()
        self.env = env

    def run(self):
        runner = _runner()
        try:
            build = getattr(runner, "build", None) or runner._build
            build(self.env)
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
        self._setup_ui()
        self._listen_for_builds()
        self.refresh()

    # UI ---------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Each model runs in an environment of its own, built on first "
            "use: minutes and gigabytes, once. Build them here ahead of "
            "time, or rebuild one a new release has changed."
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

        self.logger = QtProgressLogger()
        layout.addWidget(self.logger.get_widget(), 1)

    def _listen_for_builds(self):
        """Hear every build's news -- this panel's and the segmenters'."""
        try:
            from ..Segmenters.GlobalSegmenters.SkopSegmenter import (
                add_build_listener,
            )
        except ImportError:
            return
        self._relay = BuildRelay()
        self._relay.message.connect(self._on_build_news)
        add_build_listener(self._relay.message.emit)

    # Contents ---------------------------------------------------------------

    def refresh(self):
        """Re-read every environment's state from disk."""
        try:
            runner = _runner()
        except Exception as e:  # noqa: BLE001 - scikit-ops missing
            self.logger.log_error(f"scikit-ops is not available: {e}")
            return
        try:
            users = _users()
        except Exception:  # noqa: BLE001 - informational only
            users = {}

        self._envs = {}
        for module in COLLECTIONS:
            for env in _collection_envs(runner, module):
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
            # Blank before any AI Lab has registered its segmenters.
            used = ", ".join(users.get(name, [])) or (
                "nothing registered here" if users and env is not None else ""
            )
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(used))
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
        return _status(env)

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
        on_disk = name is not None and (_appose_root() / name).is_dir()
        self.folder_btn.setEnabled(on_disk)
        self.delete_btn.setEnabled(on_disk and status != BUILDING)

    # Actions ----------------------------------------------------------------

    def _on_build(self):
        name = self._selected_name()
        env = self._envs.get(name)
        if env is None or name in self._building:
            return
        status = _status(env)
        if status == READY and not self._confirm(
            "Rebuild",
            f"'{name}' is already built from the current recipe. "
            "Rebuilding re-runs pixi, which installs nothing new. Go ahead?",
        ):
            return
        self._failed.pop(name, None)
        _say(f"Building the '{name}' environment, from {env.config}")
        thread = _BuildThread(env)
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
            _say(f"✅ The '{name}' environment is built and ready")
            self.logger.progressBar.setValue(100)
        else:
            self._failed[name] = message
            # oprunner >= 0.1.0.dev3 already says which environment failed.
            if not message.startswith("Building the"):
                message = (
                    f"Building the '{name}' environment failed: {message}"
                )
            _say(f"❌ {message}")
            env = self._envs.get(name)
            # oprunner >= 0.1.0.dev3 names the log in the message already.
            log = _log_path(env) if env is not None else None
            if log is not None and str(log) not in message and log.exists():
                _say(f"Full log: {log}")
        self.refresh()

    def _on_build_news(self, line):
        """A line from any build: log it, and move the bar if it says how far."""
        if line.startswith("❌"):
            self.logger.log_error(line[1:].strip())
        else:
            self.logger.log_info(line)
        found = _PROGRESS.match(line.strip())
        if found and found.group(3):
            current, total = int(found.group(2)), int(found.group(3))
            if total:
                self.logger.progressBar.setValue(int(100 * current / total))

    def _on_show_log(self):
        env = self._envs.get(self._selected_name())
        if env is None:
            return
        log = _log_path(env)
        if log.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(log)))
        else:
            self.logger.log_warning(
                f"No build log for '{env.name}' yet: logs are kept from the "
                "next build on."
            )

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
        from ..Segmenters.GlobalSegmenters.SkopSegmenter import close_runner

        close_runner()  # a running worker holds its environment's files
        try:
            shutil.rmtree(folder)
            self.logger.log_info(f"Deleted {folder}")
        except OSError as e:
            self.logger.log_error(f"Could not delete {folder}: {e}")
        self.refresh()

    def _confirm(self, title, text) -> bool:
        answer = QMessageBox.question(
            self, title, text, QMessageBox.Yes | QMessageBox.No
        )
        return answer == QMessageBox.Yes

    def closeEvent(self, event):
        if self._size_thread is not None:
            self._size_thread.requestInterruption()
        super().closeEvent(event)
