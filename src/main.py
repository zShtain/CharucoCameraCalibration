import sys
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

from CalibrationModel import CalibrationModel
from CalibrationView import CalibrationView, icon_path
from CalibrationController import CalibrationController
from theme import apply_dark_theme


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    app.setWindowIcon(QIcon(icon_path))
    model = CalibrationModel()
    view = CalibrationView()
    ctrl = CalibrationController(model, view)
    view.show()
    sys.exit(app.exec_())
