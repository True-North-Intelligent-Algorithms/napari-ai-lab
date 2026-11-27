"""An error box's text can be copied."""

from qtpy.QtWidgets import QApplication, QMessageBox

from napari_ai_lab.apps.nd_easy_segment import _show_error

TRACEBACK = (
    "Task failed: Traceback (most recent call last):\n"
    '  File "cellpose/train.py", line 284, in _process_train_test\n'
    "ZeroDivisionError: float division by zero"
)


def test_error_box_shows_the_last_line_and_copies_it_all(qtbot, monkeypatch):
    shown = []

    def fake_exec(box):
        shown.append(box)
        next(b for b in box.buttons() if b.text() == "Copy").click()

    monkeypatch.setattr(QMessageBox, "exec_", fake_exec)
    _show_error(None, "Training Error", "Training failed", TRACEBACK)

    (box,) = shown
    assert box.text().endswith("ZeroDivisionError: float division by zero")
    assert box.detailedText() == TRACEBACK
    assert QApplication.clipboard().text() == f"Training failed\n\n{TRACEBACK}"
    assert box.clickedButton() is None  # Copy left the box open
