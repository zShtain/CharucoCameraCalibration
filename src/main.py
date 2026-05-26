import sys
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

from CalibrationModel import CalibrationModel
from CalibrationView import CalibrationView, icon_path
from CalibrationController import CalibrationController


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(icon_path))
    model = CalibrationModel()
    view  = CalibrationView()
    ctrl  = CalibrationController(model, view)
    view.show()
    sys.exit(app.exec_())
