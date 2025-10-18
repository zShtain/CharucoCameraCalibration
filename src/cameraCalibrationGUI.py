import os
import sys
import cv2
import json
import traceback
import matplotlib
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QComboBox, QPushButton, QStyledItemDelegate, QListView, QGraphicsView,
    QFileDialog, QMessageBox, QGridLayout,    QHBoxLayout, QVBoxLayout, QGroupBox, QCheckBox, QGraphicsScene)

matplotlib.use("Agg")  # use headless backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Path to your icon
icon_path = os.path.dirname(__file__) + "/config/technion.png"


class CenteredComboBoxDelegate(QStyledItemDelegate):
    """ Delegate that centers text inside a QComboBox """
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter


class CenteredListDelegate(QStyledItemDelegate):
    """ Delegate that centers text in QListView items """
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter



class CalibrationGui(QWidget):
    def __init__(self):
        super().__init__()
        self.loaded_images = []          # grayscale numpy arrays
        self.detected_markers = []       # list of (corners, ids) tuples
        self.detected_corners = []       # list of (charucoCorners, charucoIds)
        self.charuco_board = None        # cv2.aruco.CharucoBoard object
        self.init_ui()

    def init_ui(self):
        # Left panel – inputs & buttons
        left_panel = QVBoxLayout()
        input_group = QGroupBox()
        input_layout = QVBoxLayout()

        # Helper to create centered QLineEdit
        def centered_lineedit():
            le = QLineEdit()
            le.setAlignment(Qt.AlignCenter)
            return le

        # --- Inputs ---
        lbl_x = QLabel("No. Squares X")
        self.txt_x = centered_lineedit()
        input_layout.addWidget(lbl_x)
        input_layout.addWidget(self.txt_x)

        lbl_y = QLabel("No. Squares Y")
        self.txt_y = centered_lineedit()
        input_layout.addWidget(lbl_y)
        input_layout.addWidget(self.txt_y)

        lbl_square_len = QLabel("Square Length")
        self.txt_square_len = centered_lineedit()
        input_layout.addWidget(lbl_square_len)
        input_layout.addWidget(self.txt_square_len)

        lbl_marker_len = QLabel("Marker Length")
        self.txt_marker_len = centered_lineedit()
        input_layout.addWidget(lbl_marker_len)
        input_layout.addWidget(self.txt_marker_len)

        lbl_dict = QLabel("Aruco Dictionary")
        self.cmb_dict = QComboBox()
        # keep names that map to cv2.aruco attributes (we will getattr them later)
        self.cmb_dict.addItems([
            "DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_100",
            "DICT_6X6_250", "DICT_7X7_1000"])

        # Center both dropdown items and the currently displayed item
        delegate = CenteredComboBoxDelegate(self.cmb_dict)
        self.cmb_dict.setItemDelegate(delegate)
        self.cmb_dict.setEditable(True)  # enable internal line editor
        self.cmb_dict.lineEdit().setAlignment(Qt.AlignCenter)  # center the visible text
        self.cmb_dict.lineEdit().setReadOnly(True)  # prevent user typing
        self.cmb_dict.setStyleSheet("""QComboBox {text-align: center;} 
        QComboBox QAbstractItemView {text-align: center;}""")

        input_layout.addWidget(lbl_dict)
        input_layout.addWidget(self.cmb_dict)

        self.chk_legacy = QCheckBox("Use Legacy")
        input_layout.addWidget(self.chk_legacy)

        input_layout.addStretch()
        input_group.setLayout(input_layout)
        left_panel.addWidget(input_group)

        # --- Buttons ---
        btn_group = QGroupBox()
        btn_layout = QVBoxLayout()
        # btn_layout.setSpacing(4)
        btn_layout.setContentsMargins(10, 10, 10, 10)

        self.btn_load = QPushButton("Load Images")
        btn_layout.addWidget(self.btn_load)
        self.btn_run = QPushButton("Run Calibration")
        btn_layout.addWidget(self.btn_run)
        self.btn_save = QPushButton("Save")
        btn_layout.addWidget(self.btn_save)
        # btn_layout.addStretch()
        btn_group.setLayout(btn_layout)
        left_panel.addWidget(btn_group)
        left_panel.addStretch()

        # --- Middle Panel ---
        middle_panel = QVBoxLayout()
        lbl_loaded = QLabel("Loaded Images")
        middle_panel.addWidget(lbl_loaded)
        self.list_loaded = QListView()
        self.list_model = QStandardItemModel(self.list_loaded)
        self.list_loaded.setModel(self.list_model)
        self.list_loaded.selectionModel().selectionChanged.connect(self.on_image_selected)
        middle_panel.addWidget(self.list_loaded)

        # Apply center alignment delegate
        self.list_loaded.setItemDelegate(CenteredListDelegate(self.list_loaded))

        # --- Calibration Results Panel ---
        results_group = QGroupBox("Calibration Results")
        results_layout = QGridLayout()

        # Define labels and corresponding textboxes
        result_labels = ["fx", "fy", "ppx", "ppy", "K1", "K2", "K3", "P1", "P2"]
        self.result_fields = {}

        for i, label_text in enumerate(result_labels):
            lbl = QLabel(label_text)
            txt = QLineEdit()
            txt.setAlignment(Qt.AlignCenter)
            txt.setReadOnly(True)
            results_layout.addWidget(lbl, i, 0)
            results_layout.addWidget(txt, i, 1)
            self.result_fields[label_text] = txt

        results_group.setLayout(results_layout)
        middle_panel.addWidget(results_group)
        middle_panel.addStretch()

        # --- Right Panel ---
        right_panel = QVBoxLayout()
        self.gv_image1 = QGraphicsView()
        right_panel.addWidget(self.gv_image1)
        self.gv_image2 = QGraphicsView()
        right_panel.addWidget(self.gv_image2)

        # --- Main Layout ---
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_panel, stretch=1)
        main_layout.addLayout(middle_panel, stretch=1)
        main_layout.addLayout(right_panel, stretch=2)
        self.setLayout(main_layout)
        self.setWindowTitle("Camera Calibration GUI for Students of Photogrammetry 1 (014843)")
        self.setWindowIcon(QIcon(icon_path))
        self.setMinimumSize(900, 600)

        # --- Connect Buttons ---
        self.btn_load.clicked.connect(self.on_load_images)
        self.btn_run.clicked.connect(self.on_run_calibration)
        self.btn_save.clicked.connect(self.on_save)
        self._load_config_file()

    # ------------------------------------------------------------------
    # Functional Slots
    # ------------------------------------------------------------------

    def _load_config_file(self):
        """Check if config.json exists next to the script and load GUI defaults from it."""
        try:
            # Determine the directory of this script
            config_path = os.path.dirname(os.path.abspath(__file__)) + "/config/config.json"

            if not os.path.isfile(config_path):
                print("No config.json found — using default GUI values.")
                return

            with open(config_path, "r") as f:
                cfg = json.load(f)

            # --- Validate keys ---
            required_keys = ["nSquaresX", "nSquaresY", "sqrLength", "markerLength", "dictName", "isLegacy"]
            for k in required_keys:
                if k not in cfg:
                    print(f"Config missing key: {k}")
                    return

            # --- Set textbox values ---
            self.txt_x.setText(str(cfg["nSquaresX"]))
            self.txt_y.setText(str(cfg["nSquaresY"]))
            self.txt_square_len.setText(str(cfg["sqrLength"]))
            self.txt_marker_len.setText(str(cfg["markerLength"]))

            # --- Set combobox value ---
            dict_name = cfg["dictName"]
            idx = self.cmb_dict.findText(dict_name)
            if idx >= 0:
                self.cmb_dict.setCurrentIndex(idx)
            else:
                print(f"Warning: dictionary '{dict_name}' not found in combo box — using default.")

            # --- Set checkbox state ---
            self.chk_legacy.setChecked(bool(cfg["isLegacy"]))

            print(f"Configuration loaded from {config_path}")

        except Exception as e:
            print(f"Error reading config.json: {e}")


    def _get_aruco_dictionary(self, dict_name):
        """
        Robustly obtain an ArUco dictionary from cv2.aruco under different OpenCV versions.
        """
        try:
            if hasattr(cv2.aruco, "getPredefinedDictionary"):
                enum_val = getattr(cv2.aruco, dict_name, None)
                if enum_val is None:
                    raise AttributeError(f"Dictionary name '{dict_name}' not found in cv2.aruco")
                return cv2.aruco.getPredefinedDictionary(enum_val)
            elif hasattr(cv2.aruco, "Dictionary_get"):
                enum_val = getattr(cv2.aruco, dict_name, None)
                if enum_val is None:
                    raise AttributeError(f"Dictionary name '{dict_name}' not found in cv2.aruco")
                return cv2.aruco.Dictionary_get(enum_val)
            else:
                raise RuntimeError("This OpenCV build does not expose a dictionary getter (getPredefinedDictionary/Dictionary_get).")
        except Exception as e:
            raise


    def _create_charuco_board(self, squaresX, squaresY, squareLength, markerLength, aruco_dict):
        """
        Try multiple ways to create a CharucoBoard depending on cv2.aruco API differences.
        """
        # 1. Preferred modern API
        if hasattr(cv2.aruco, "CharucoBoard_create"):
            try:
                return cv2.aruco.CharucoBoard_create(
                    squaresX, squaresY, squareLength, markerLength, aruco_dict
                )
            except Exception:
                pass

        # 2. Class method (OpenCV >= 4.7)
        if hasattr(cv2.aruco, "CharucoBoard") and hasattr(cv2.aruco.CharucoBoard, "create"):
            try:
                return cv2.aruco.CharucoBoard.create(
                    squaresX, squaresY, squareLength, markerLength, aruco_dict
                )
            except Exception:
                pass

        # 3. Direct constructor fallback (works in all contrib builds)
        try:
            return cv2.aruco.CharucoBoard(
                (squaresX, squaresY), squareLength, markerLength, aruco_dict
            )
        except Exception as e:
            raise AttributeError(
                f"Unable to create CharucoBoard. Neither 'CharucoBoard_create', "
                f"'CharucoBoard.create', nor constructor succeeded. "
                f"Error: {e}"
            )


    def on_load_images(self):
        """Initialize Charuco board and detect markers in selected images, with robust error handling."""
        try:
            # --- Step 1: Read input values (with validation) ---
            try:
                squaresX = int(self.txt_x.text())
                squaresY = int(self.txt_y.text())
                squareLength = float(self.txt_square_len.text())
                markerLength = float(self.txt_marker_len.text())
            except ValueError:
                QMessageBox.critical(self, "Error", "Invalid numeric input. Please enter integer values for number of squares and numeric values for lengths.")
                return

            if markerLength >= squareLength:
                QMessageBox.critical(self, "Error", "Marker Length must be smaller than Square Length.")
                return

            # --- Step 2: Create ArUco dictionary (robust) ---
            dict_name = self.cmb_dict.currentText()
            try:
                aruco_dict = self._get_aruco_dictionary(dict_name)
            except Exception as e:
                QMessageBox.critical(self, "Error creating ArUco dictionary", f"Failed to create ArUco dictionary '{dict_name}':\n{e}")
                return

            # --- Step 3: Create Charuco board (robust) ---
            try:
                self.charuco_board = self._create_charuco_board(squaresX, squaresY, squareLength, markerLength, aruco_dict)
            except Exception as e:
                # Provide traceback to help diagnose mismatch of OpenCV build
                tb = traceback.format_exc()
                QMessageBox.critical(self, "Error creating ChArUco board", f"{e}\n\nTraceback:\n{tb}")
                return

            # If we reach here the board was created successfully
            print("ChArUco board initialized successfully.")

            self.charuco_board.setLegacyPattern(self.chk_legacy.isChecked())

            # --- Step 4: File dialog for image selection ---
            file_dialog = QFileDialog(self, "Select Image Files")
            file_dialog.setFileMode(QFileDialog.ExistingFiles)
            file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
            if not file_dialog.exec_():
                return  # user canceled

            selected_files = file_dialog.selectedFiles()
            if not selected_files:
                return

            # --- Step 5: Clear previous data ---
            self.loaded_images.clear()
            self.list_model.clear()
            self.detected_markers.clear()
            self.detected_corners.clear()

            # --- Step 6: Process each image robustly ---
            for file_path in selected_files:
                try:
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Warning: Could not read {file_path}")
                        continue

                    # Append image and list item
                    self.loaded_images.append(img)
                    filename = os.path.splitext(os.path.basename(file_path))[0]
                    self.list_model.appendRow(QStandardItem(filename))

                    # Detect markers (wrapped in try/except)
                    try:
                        # choose detection function - same aruco dict we already have
                        if hasattr(cv2.aruco, "detectMarkers"):
                            corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict)
                        else:
                            # very old API fallback - unlikely, but handle gracefully
                            corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict)

                        self.detected_markers.append((corners, ids))
                    except Exception as e:
                        print(f"Marker detection error for {filename}: {e}")
                        self.detected_markers.append((None, None))
                        self.detected_corners.append((None, None))
                        continue

                    # Interpolate ChArUco corners if markers found
                    try:
                        if ids is not None and len(ids) > 0:
                            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                                markerCorners=corners,
                                markerIds=ids,
                                image=img,
                                board=self.charuco_board)
                            self.detected_corners.append((charuco_corners, charuco_ids))
                            if charuco_ids is not None:
                                print(f"{filename}: Detected {len(charuco_ids)} ChArUco corners.")
                            else:
                                print(f"{filename}: interpolateCornersCharuco returned no charuco_ids.")
                        else:
                            self.detected_corners.append((None, None))
                            print(f"{filename}: No markers found.")
                    except Exception as e:
                        print(f"Charuco interpolation error for {filename}: {e}")
                        self.detected_corners.append((None, None))

                except Exception as per_image_e:
                    # Catch-all so a single bad file won't crash everything
                    print(f"Unexpected error processing {file_path}: {per_image_e}")
                    traceback.print_exc()
                    continue

            print(f"Loaded and processed {len(self.loaded_images)} images.")
            self._plot_detected_corners()

            selected_files = file_dialog.selectedFiles()
            if not selected_files:
                return

            # Store directory for later use
            self.last_image_dir = os.path.dirname(selected_files[0])

        except Exception as top_e:
            # Very top-level safety net
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Unexpected Error", f"An unexpected error occurred:\n{top_e}\n\nTraceback:\n{tb}")
            print("Unexpected error in on_load_images:", top_e)
            print(tb)

    def on_run_calibration(self):
        """Perform camera calibration using detected ChArUco corners."""
        try:
            # --- Step 1: sanity checks ---
            if self.charuco_board is None:
                QMessageBox.critical(self, "Error", "No ChArUco board initialized. Please load images first.")
                return
            if not self.detected_corners or all(c[0] is None for c in self.detected_corners):
                QMessageBox.critical(self, "Error", "No valid ChArUco corners detected. Please load valid images.")
                return
            if not self.loaded_images:
                QMessageBox.critical(self, "Error", "No images loaded.")
                return

            # --- Step 2: collect valid data ---
            all_charuco_corners = []
            all_charuco_ids = []
            image_size = None

            for i, (corners, ids) in enumerate(self.detected_corners):
                if corners is not None and ids is not None and len(corners) > 3:
                    all_charuco_corners.append(corners)
                    all_charuco_ids.append(ids)
                    if image_size is None:
                        h, w = self.loaded_images[i].shape[:2]
                        image_size = (w, h)

            if len(all_charuco_corners) == 0:
                QMessageBox.critical(self, "Error", "No sufficient ChArUco corners found for calibration.")
                return

            # --- Step 3: run calibration ---
            print(f"Running calibrateCameraCharucoExtended on {len(all_charuco_corners)} valid images...")

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 1000, 1e-9)
            (retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics,
             perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(charucoCorners=all_charuco_corners,
                                                                       charucoIds=all_charuco_ids,
                                                                       board=self.charuco_board,
                                                                       imageSize=image_size, cameraMatrix=None,
                                                                       distCoeffs=None, criteria=criteria)

            # --- Step 4: store results as private members ---
            self._reprojectionError = retval
            self._cameraMatrix = cameraMatrix
            self._distCoeffs = distCoeffs
            self._rvecs = rvecs
            self._tvecs = tvecs
            self._stdDeviationsIntrinsics = stdDeviationsIntrinsics
            self._stdDeviationsExtrinsics = stdDeviationsExtrinsics
            self._perViewErrors = perViewErrors

            # --- Step 5: print & show summary ---
            print("Calibration completed successfully.")
            print("Reprojection Error:", retval)
            print("Camera Matrix:\n", cameraMatrix)
            print("Distortion Coefficients:\n", distCoeffs)

            values = {
                "fx": cameraMatrix[0, 0],
                "fy": cameraMatrix[1, 1],
                "ppx": cameraMatrix[0, 2],
                "ppy": cameraMatrix[1, 2],
                "K1": distCoeffs[0, 0],
                "K2": distCoeffs[0, 1],
                "K3": distCoeffs[0, 4],
                "P1": distCoeffs[0, 2],
                "P2": distCoeffs[0, 3],
            }

            for key, val in values.items():
                if key in self.result_fields:
                    self.result_fields[key].setText(f"{val:.6f}")

            QMessageBox.information(
                self,
                "Calibration Complete",
                f"Calibration successful!\n\nReprojection Error: {retval:.3f}\n"
                f"Images used: {len(all_charuco_corners)}"
            )

        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(
                self,
                "Calibration Error",
                f"An error occurred during calibration:\n{e}\n\nTraceback:\n{tb}"
            )
            print("Calibration error:", e)
            print(tb)

    def on_save(self):
        """Save the computed camera calibration parameters to calibration.json in the image folder."""
        try:
            # --- Safety checks ---
            if not hasattr(self, "_cameraMatrix") or self._cameraMatrix is None:
                QMessageBox.warning(self, "Save Calibration",
                                    "No calibration results found. Please run calibration first.")
                return
            if not hasattr(self, "_distCoeffs") or self._distCoeffs is None:
                QMessageBox.warning(self, "Save Calibration",
                                    "Distortion coefficients are missing. Please run calibration first.")
                return
            if not hasattr(self, "loaded_images") or len(self.loaded_images) == 0:
                QMessageBox.warning(self, "Save Calibration",
                                    "No images were loaded. Cannot determine output directory.")
                return

            # --- Determine output folder ---
            if hasattr(self, "last_image_dir") and self.last_image_dir:
                out_dir = self.last_image_dir
            else:
                first_image_path = getattr(self, "last_loaded_file", None)
                if first_image_path:
                    out_dir = os.path.dirname(first_image_path)
                else:
                    QMessageBox.warning(self, "Save Calibration", "Cannot determine where images were loaded from.")
                    return

            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "calibration.json")

            # --- Extract calibration data safely ---
            cm = self._cameraMatrix
            dc = self._distCoeffs.flatten()
            rms = getattr(self, "_reprojectionError", None)
            n_images = len(getattr(self, "_rvecs", []))

            if len(dc) < 5:
                QMessageBox.warning(self, "Save Calibration", f"Unexpected distortion coefficient length: {len(dc)}")
                return

            calib_data = {
                "intrinsics": {
                    "focalLengthX": float(cm[0, 0]),
                    "focalLengthY": float(cm[1, 1]),
                    "principalPointX": float(cm[0, 2]),
                    "principalPointY": float(cm[1, 2])
                },
                "lensDistortions": {
                    "radial": {
                        "k1": float(dc[0]),
                        "k2": float(dc[1]),
                        "k3": float(dc[4]) if len(dc) > 4 else 0.0
                    },
                    "tangential": {
                        "p1": float(dc[2]),
                        "p2": float(dc[3])
                    }
                },
                "calibrationSummary": {
                    "rmsError": float(rms) if rms is not None else None,
                    "numImagesUsed": int(n_images)
                }
            }

            img_data = {}
            for i in range(len(self.loaded_images)):
                img_name = self.list_model.item(i).text()
                charuco_corners, charuco_ids = self.detected_corners[i]
                charuco_ids = charuco_ids.reshape((-1, ))
                corner_data = {}
                for id, pt in zip(charuco_ids, charuco_corners):
                    corner_data[int(id)] = {"x": float(pt[0, 0]), "y": float(pt[0, 1])}
                img_data[img_name] = corner_data

            calib_data['data'] = img_data

            # --- Save to JSON ---
            with open(out_path, "w") as f:
                json.dump(calib_data, f, indent=4)

            QMessageBox.information(self, "Save Calibration", f"Calibration saved successfully to:\n{out_path}")
            print(f"Calibration JSON saved to {out_path}")

        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Save Calibration Error",
                                 f"Failed to save calibration:\n{e}\n\nTraceback:\n{tb}")


    def on_image_selected(self, selected, deselected):
        """
        Display the selected image in gv_image1, overlaying detected ChArUco corners
        drawn after scaling so they align perfectly with the displayed image.
        """
        if not selected.indexes():
            return

        try:
            index = selected.indexes()[0].row()
            if index < 0 or index >= len(self.loaded_images):
                return

            img = self.loaded_images[index]
            if img is None:
                return

            # Convert grayscale to BGR for drawing colored circles
            orig_h, orig_w = img.shape[:2]
            display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Determine target width (use inner viewport width of the graphics view)
            target_width = max(1, self.gv_image1.viewport().width())

            # If image already fits, no resize needed; else compute scale and resize
            if orig_w != target_width:
                scale = target_width / float(orig_w)
                target_height = int(round(orig_h * scale))
                # Resize image (scale corners accordingly below)
                scaled_img = cv2.resize(display_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            else:
                scale = 1.0
                scaled_img = display_img.copy()

            # Draw detected ChArUco corners (scale coordinates first)
            if index < len(self.detected_corners):
                charuco_corners, charuco_ids = self.detected_corners[index]
                # charuco_corners is typically shape (N,1,2) with float coordinates
                if charuco_corners is not None and len(charuco_corners) > 0:
                    for pt in charuco_corners:
                        # pt can be shape (1,2) or (2,) depending on OpenCV version/format
                        if hasattr(pt, "shape") and pt.shape[-1] == 2:
                            x_float = float(pt[0][0]) if pt.ndim == 2 else float(pt[0])
                            y_float = float(pt[0][1]) if pt.ndim == 2 else float(pt[1])
                        else:
                            # fallback robust extraction
                            try:
                                x_float = float(pt[0][0])
                                y_float = float(pt[0][1])
                            except Exception:
                                continue
                        # scale coordinates
                        x_scaled = int(round(x_float * scale))
                        y_scaled = int(round(y_float * scale))
                        # draw a filled blue circle (BGR = (255,0,0))
                        cv2.circle(scaled_img, (x_scaled, y_scaled), 4, (255, 0, 0), -1)

                # --- If calibration exists, show undistorted image in gv_image2 ---
                if hasattr(self, "_cameraMatrix") and hasattr(self, "_distCoeffs"):
                    if self._cameraMatrix is not None and self._distCoeffs is not None:
                        try:
                            undistorted = cv2.undistort(img, self._cameraMatrix, self._distCoeffs)
                            scaled_undistorted = cv2.resize(undistorted, (target_width, target_height),
                                                            interpolation=cv2.INTER_LINEAR)
                            undistorted_color = cv2.cvtColor(scaled_undistorted, cv2.COLOR_GRAY2BGR)
                            self._show_image_in_graphicsview(undistorted_color, self.gv_image2)
                            print(f"Displayed undistorted image for index {index}.")
                        except Exception as e:
                            print(f"Undistortion failed for image {index}: {e}")
                            traceback.print_exc()

            # Now show the pre-scaled image as-is (no further scaling in the viewer)
            self._show_image_in_graphicsview(scaled_img, self.gv_image1, scale_already_applied=True)

        except Exception as e:
            QMessageBox.warning(self, "Display Error", f"Failed to display image:\n{e}")


    def _show_image_in_graphicsview(self, img, view, scale_already_applied=False):
        """
        Display an OpenCV image in a QGraphicsView.
        If scale_already_applied is True the image should already have the desired width,
        so the function will show it without further scaling.
        """
        try:
            # Convert to QImage
            if len(img.shape) == 2:
                qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
            else:
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0],
                              rgb_image.strides[0], QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qimg)

            scene = QGraphicsScene()
            scene.addPixmap(pixmap)
            scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
            view.setScene(scene)

            # Ensure no further automatic scaling: adjust view's transform so the pixmap pixels map 1:1
            view.resetTransform()
            # center the image in the view
            view.setAlignment(Qt.AlignCenter)

            # Optionally, if the view has scrollbars, disable them so it appears neatly:
            view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        except Exception as e:
            print(f"Error displaying image in graphics view: {e}")


    def _plot_detected_corners(self):
        """Display a scatter plot of all detected ChArUco corners from all images in gv_image2."""
        try:
            if not self.loaded_images or len(self.detected_corners) == 0:
                QMessageBox.warning(self, "Corner Plot", "No images or detected corners available.")
                return

                # --- Get base image dimensions (from first image) ---
            h, w = self.loaded_images[0].shape[:2]

            # --- Get graphics view size (in pixels) ---
            gv_width = max(200, self.gv_image2.viewport().width())
            gv_height = max(200, self.gv_image2.viewport().height())

            # Convert pixels to inches for Matplotlib (DPI = 100)
            dpi = 100
            fig_w = gv_width / dpi
            fig_h = gv_height / dpi

            # --- Prepare figure ---
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            ax.set_facecolor("white")
            # ax.set_title("Detected ChArUco Corners (All Images)", fontsize=10)
            # ax.set_xlabel("X (pixels)")
            # ax.set_ylabel("Y (pixels)")
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)  # invert Y to match OpenCV coordinates

            # --- Draw image boundary rectangle ---
            rect = plt.Rectangle((0, 0), w, h, linewidth=2, edgecolor='black', facecolor='none')
            ax.add_patch(rect)

            # --- Plot detected corners ---
            cmap = plt.get_cmap("tab10", len(self.detected_corners))
            for i, (charuco_corners, charuco_ids) in enumerate(self.detected_corners):
                if charuco_corners is None or len(charuco_corners) == 0:
                    continue
                pts = charuco_corners.reshape(-1, 2)
                ax.scatter(pts[:, 0], pts[:, 1], s=10, color=cmap(i), label=f"Image {i + 1}")

            # --- Aspect ---
            ax.set_aspect("equal", adjustable="box")
            fig.tight_layout(pad=0.0)

            # --- Render to NumPy image at the correct size ---
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            width, height = canvas.get_width_height()
            buf = canvas.buffer_rgba()
            image = np.asarray(buf, dtype=np.uint8).reshape(height, width, 4)[..., :3]
            plt.close(fig)

            # --- Display scaled image in gv_image2 ---
            self._show_image_in_graphicsview(cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                             self.gv_image2, scale_already_applied=True)

        except Exception as e:
            QMessageBox.critical(self, "Corner Plot Error", f"Failed to plot detected corners:\n{e}")
            traceback.print_exc()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(icon_path))
    window = CalibrationGui()
    window.show()
    sys.exit(app.exec_())
