from PyQt5 import QtCore
from arpes.io import example_data

from pytestqt.qtbot import QtBot
from pytestqt.qt_compat import qt_api
from arpes.plotting.qt_tool import QtTool


def test_open_qt_tool_and_basic_functionality(qtbot: QtBot):
    app = qt_api.QtWidgets.QApplication.instance()

    example_data.cut.S.show(app=app, no_exec=True)
    owner: QtTool = app.owner

    # Check transposition info
    assert list(owner.data.dims) == ["phi", "eV"]

    qtbot.keyPress(app.owner.cw, "t")
    assert list(owner.data.dims) == ["eV", "phi"]
    qtbot.keyPress(app.owner.cw, "y")
    assert list(owner.data.dims) == ["phi", "eV"]

    # Check cursor info
    assert app.owner.context["cursor"] == [120, 120]
    qtbot.keyPress(app.owner.cw, QtCore.Qt.Key_Left)
    assert app.owner.context["cursor"] == [118, 120]
    qtbot.keyPress(app.owner.cw, QtCore.Qt.Key_Up)
    assert app.owner.context["cursor"] == [118, 122]
    qtbot.keyPress(app.owner.cw, QtCore.Qt.Key_Right)
    qtbot.keyPress(app.owner.cw, QtCore.Qt.Key_Right)
    assert app.owner.context["cursor"] == [122, 122]
    qtbot.keyPress(app.owner.cw, QtCore.Qt.Key_Down)
    assert app.owner.context["cursor"] == [122, 120]

    app.owner.close()
