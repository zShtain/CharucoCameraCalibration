import os
import sys
import cv2
import datetime
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QImage, QPixmap, QIcon, QTextCursor
from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QComboBox, QPushButton, QStyledItemDelegate,
    QListView, QGraphicsView, QMessageBox, QGridLayout, QHBoxLayout, QVBoxLayout,
    QGroupBox, QCheckBox, QGraphicsScene, QTextEdit)

icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "technion.png")


class CenteredComboBoxDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter


class CenteredListDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter


class QTextEditLogger(QObject):
    """Redirects stdout/stderr to a QTextEdit widget, adding timestamps and colors."""
    text_written = pyqtSignal(str)

    def __init__(self, text_edit):
        super().__init__()
        self._text_edit = text_edit
        self.text_written.connect(self._append_text)

    def write(self, text):
        text = text.strip()
        if text:
            timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
            color = "black"
            if "error" in text.lower():
                color = "red"
            elif "warning" in text.lower():
                color = "orange"
            elif "success" in text.lower() or "saved" in text.lower():
                color = "green"
            formatted = (f"<span style='color: gray;'>{timestamp}</span> "
                         f"<span style='color: {color};'>{text}</span>")
            self.text_written.emit(formatted)

    def flush(self):
        pass

    def _append_text(self, html_text):
        self._text_edit.moveCursor(QTextCursor.End)
        self._text_edit.insertHtml(html_text + "<br>")
        self._text_edit.ensureCursorVisible()


class CalibrationView(QWidget):
    load_requested           = pyqtSignal()
    calibrate_requested      = pyqtSignal()
    save_requested           = pyqtSignal()
    clear_all_requested      = pyqtSignal()
    clear_selected_requested = pyqtSignal()
    image_selected           = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        # --- Left Panel ---
        left_panel = QVBoxLayout()
        input_group = QGroupBox()
        input_layout = QVBoxLayout()

        def centered_lineedit():
            le = QLineEdit()
            le.setAlignment(Qt.AlignCenter)
            return le

        input_layout.addWidget(QLabel("No. Squares X"))
        self._txt_x = centered_lineedit()
        input_layout.addWidget(self._txt_x)

        input_layout.addWidget(QLabel("No. Squares Y"))
        self._txt_y = centered_lineedit()
        input_layout.addWidget(self._txt_y)

        input_layout.addWidget(QLabel("Square Length"))
        self._txt_square_len = centered_lineedit()
        input_layout.addWidget(self._txt_square_len)

        input_layout.addWidget(QLabel("Marker Length"))
        self._txt_marker_len = centered_lineedit()
        input_layout.addWidget(self._txt_marker_len)

        input_layout.addWidget(QLabel("Aruco Dictionary"))
        self._cmb_dict = QComboBox()
        self._cmb_dict.addItems([
            "DICT_4X4_50",  "DICT_4X4_100",  "DICT_4X4_250",  "DICT_4X4_1000",
            "DICT_5X5_50",  "DICT_5X5_100",  "DICT_5X5_250",  "DICT_5X5_1000",
            "DICT_6X6_50",  "DICT_6X6_100",  "DICT_6X6_250",  "DICT_6X6_1000",
            "DICT_7X7_50",  "DICT_7X7_100",  "DICT_7X7_250",  "DICT_7X7_1000"])
        self._cmb_dict.setItemDelegate(CenteredComboBoxDelegate(self._cmb_dict))
        self._cmb_dict.setEditable(True)
        self._cmb_dict.lineEdit().setAlignment(Qt.AlignCenter)
        self._cmb_dict.lineEdit().setReadOnly(True)
        self._cmb_dict.setStyleSheet("""QComboBox {text-align: center;}
        QComboBox QAbstractItemView {text-align: center;}""")
        input_layout.addWidget(self._cmb_dict)

        self._chk_legacy = QCheckBox("Use Legacy")
        input_layout.addWidget(self._chk_legacy)

        input_layout.addStretch()
        input_group.setLayout(input_layout)
        left_panel.addWidget(input_group)

        btn_group = QGroupBox()
        btn_layout = QVBoxLayout()
        btn_layout.setContentsMargins(10, 10, 10, 10)
        self._btn_load = QPushButton("Load Images")
        self._btn_run  = QPushButton("Run Calibration")
        self._btn_save = QPushButton("Save")
        btn_layout.addWidget(self._btn_load)
        btn_layout.addWidget(self._btn_run)
        btn_layout.addWidget(self._btn_save)
        btn_group.setLayout(btn_layout)
        left_panel.addWidget(btn_group)

        self._txt_log = QTextEdit()
        self._txt_log.setReadOnly(True)
        self._txt_log.setPlaceholderText("Application log will appear here...")
        self._txt_log.setMinimumHeight(120)
        self._txt_log.setStyleSheet("""QTextEdit {background-color: #f4f4f4;
        font-family: Consolas, monospace; font-size: 10pt;}""")
        left_panel.addWidget(self._txt_log)

        self._logger = QTextEditLogger(self._txt_log)
        sys.stdout = self._logger
        sys.stderr = self._logger

        left_panel.addStretch()

        # --- Middle Panel ---
        middle_panel = QVBoxLayout()
        middle_panel.addWidget(QLabel("Loaded Images"))
        self._list_loaded = QListView()
        self._list_model  = QStandardItemModel(self._list_loaded)
        self._list_loaded.setModel(self._list_model)
        self._list_loaded.setItemDelegate(CenteredListDelegate(self._list_loaded))
        self._list_loaded.selectionModel().selectionChanged.connect(self._on_selection_changed)
        middle_panel.addWidget(self._list_loaded)

        self._btn_clear_all      = QPushButton("Clear All")
        self._btn_clear_selected = QPushButton("Clear Selected")
        middle_panel.addWidget(self._btn_clear_all)
        middle_panel.addWidget(self._btn_clear_selected)

        results_group  = QGroupBox("Calibration Results")
        results_layout = QGridLayout()
        self._result_fields = {}
        for i, label_text in enumerate(["fx", "fy", "ppx", "ppy", "K1", "K2", "K3", "P1", "P2"]):
            txt = QLineEdit()
            txt.setAlignment(Qt.AlignCenter)
            txt.setReadOnly(True)
            results_layout.addWidget(QLabel(label_text), i, 0)
            results_layout.addWidget(txt, i, 1)
            self._result_fields[label_text] = txt
        results_group.setLayout(results_layout)
        middle_panel.addWidget(results_group)
        middle_panel.addStretch()

        # --- Right Panel ---
        right_panel = QVBoxLayout()
        self._gv_image1 = QGraphicsView()
        self._gv_image2 = QGraphicsView()
        right_panel.addWidget(self._gv_image1)
        right_panel.addWidget(self._gv_image2)

        # --- Main Layout ---
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_panel,   stretch=1)
        main_layout.addLayout(middle_panel, stretch=1)
        main_layout.addLayout(right_panel,  stretch=2)
        self.setLayout(main_layout)
        self.setWindowTitle("Camera Calibration GUI for Students of Photogrammetry 1 (014843)")
        self.setWindowIcon(QIcon(icon_path))
        self.setMinimumSize(1280, 720)

        # Connect buttons to outgoing signals
        self._btn_load.clicked.connect(self.load_requested)
        self._btn_run.clicked.connect(self.calibrate_requested)
        self._btn_save.clicked.connect(self.save_requested)
        self._btn_clear_all.clicked.connect(self.clear_all_requested)
        self._btn_clear_selected.clicked.connect(self.clear_selected_requested)

    def _on_selection_changed(self, selected, _deselected):
        if selected.indexes():
            self.image_selected.emit(selected.indexes()[0].row())

    # ------------------------------------------------------------------
    # Input getters
    # ------------------------------------------------------------------

    def get_squares_x(self):     return self._txt_x.text()
    def get_squares_y(self):     return self._txt_y.text()
    def get_square_length(self): return self._txt_square_len.text()
    def get_marker_length(self): return self._txt_marker_len.text()
    def get_dict_name(self):     return self._cmb_dict.currentText()
    def get_is_legacy(self):     return self._chk_legacy.isChecked()

    def get_selected_rows(self):
        return [idx.row() for idx in self._list_loaded.selectedIndexes()]

    def get_panel_size(self, panel):
        """Return (width, height) of the viewport for the given panel (0=left, 1=right)."""
        gv = self._gv_image1 if panel == 0 else self._gv_image2
        return gv.viewport().width(), gv.viewport().height()

    # ------------------------------------------------------------------
    # Display / update methods
    # ------------------------------------------------------------------

    def set_defaults(self, cfg):
        """Populate input fields from a config dict."""
        if not cfg:
            return
        self._txt_x.setText(str(cfg.get("nSquaresX", "")))
        self._txt_y.setText(str(cfg.get("nSquaresY", "")))
        self._txt_square_len.setText(str(cfg.get("sqrLength", "")))
        self._txt_marker_len.setText(str(cfg.get("markerLength", "")))
        dict_name = cfg.get("dictName", "")
        idx = self._cmb_dict.findText(dict_name)
        if idx >= 0:
            self._cmb_dict.setCurrentIndex(idx)
        else:
            print(f"Warning: dictionary '{dict_name}' not found in combo box — using default.")
        self._chk_legacy.setChecked(bool(cfg.get("isLegacy", False)))

    def append_image_item(self, name):
        self._list_model.appendRow(QStandardItem(name))

    def remove_image_item(self, row):
        self._list_model.removeRow(row)

    def clear_image_list(self):
        self._list_model.clear()

    def show_results(self, values):
        """Fill the calibration result fields from a {key: float} dict."""
        for key, val in values.items():
            if key in self._result_fields:
                self._result_fields[key].setText(f"{val:.6f}")

    def clear_results(self):
        for field in self._result_fields.values():
            field.clear()

    def show_image(self, img, panel, scale_already_applied=False):
        """Display an OpenCV image in the given panel (0=left, 1=right)."""
        gv = self._gv_image1 if panel == 0 else self._gv_image2
        self._show_image_in_graphicsview(img, gv, scale_already_applied)

    def clear_image_panels(self):
        self._clear_graphics_view(self._gv_image1)
        self._clear_graphics_view(self._gv_image2)

    def show_error(self, title, message):
        QMessageBox.critical(self, title, message)

    def show_info(self, title, message):
        QMessageBox.information(self, title, message)

    def show_warning(self, title, message):
        QMessageBox.warning(self, title, message)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _show_image_in_graphicsview(self, img, view, _scale_already_applied=False):
        try:
            if len(img.shape) == 2:
                qimg = QImage(img.data, img.shape[1], img.shape[0],
                              img.strides[0], QImage.Format_Grayscale8)
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                              rgb.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            scene = QGraphicsScene()
            scene.addPixmap(pixmap)
            scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
            view.setScene(scene)
            view.resetTransform()
            view.setAlignment(Qt.AlignCenter)
            view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        except Exception as e:
            print(f"Error displaying image in graphics view: {e}")

    def _clear_graphics_view(self, view):
        scene = view.scene()
        if scene:
            scene.clear()

    def _save_log_to_file(self):
        try:
            log_text = self._txt_log.toPlainText()
            if not log_text.strip():
                return
            logs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            os.makedirs(logs_path, exist_ok=True)
            timestamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = os.path.join(logs_path, f"calibration_log_{timestamp}.log")
            with open(log_filename, "w", encoding="utf-8") as f:
                f.write(log_text)
            print(f"Log saved to {log_filename}")
        except Exception as e:
            print(f"Error saving log file: {e}")

    def closeEvent(self, event):
        self._save_log_to_file()
        event.accept()
