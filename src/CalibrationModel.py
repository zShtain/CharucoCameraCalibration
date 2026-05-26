import os
import cv2
import json
import traceback
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal


class CalibrationModel(QObject):
    log_message      = pyqtSignal(str)
    images_loaded    = pyqtSignal(list)
    calibration_done = pyqtSignal(dict)
    save_done        = pyqtSignal(str)
    error_occurred   = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.loaded_images      = []
        self.detected_markers   = []
        self.detected_corners   = []
        self.image_names        = []
        self.charuco_board      = None
        self._aruco_dict        = None
        self.camera_matrix      = None
        self.dist_coeffs        = None
        self.reprojection_error = None
        self.rvecs              = None
        self.tvecs              = None
        self.std_dev_intrinsics = None
        self.std_dev_extrinsics = None
        self.per_view_errors    = None
        self.last_image_dir     = ""

    def load_config(self):
        """Load config/config.json and return as dict. Returns {} on failure."""
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.json")
        if not os.path.isfile(config_path):
            self.log_message.emit("No config.json found — using default GUI values.")
            return {}
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            for k in ["nSquaresX", "nSquaresY", "sqrLength", "markerLength", "dictName", "isLegacy"]:
                if k not in cfg:
                    self.log_message.emit(f"Config missing key: {k}")
                    return {}
            self.log_message.emit(f"Configuration loaded from {config_path}")
            return cfg
        except Exception as e:
            self.log_message.emit(f"Error reading config.json: {e}")
            return {}

    def initialize_board(self, squares_x, squares_y, square_length, marker_length, dict_name, is_legacy):
        """Create ArUco dictionary and CharucoBoard. Returns True on success."""
        try:
            self._aruco_dict = self._get_aruco_dictionary(dict_name)
        except Exception as e:
            self.error_occurred.emit("Error creating ArUco dictionary",
                                     f"Failed to create ArUco dictionary '{dict_name}':\n{e}")
            return False
        try:
            self.charuco_board = self._create_charuco_board(
                squares_x, squares_y, square_length, marker_length, self._aruco_dict)
        except Exception as e:
            tb = traceback.format_exc()
            self.error_occurred.emit("Error creating ChArUco board", f"{e}\n\nTraceback:\n{tb}")
            return False
        self.charuco_board.setLegacyPattern(is_legacy)
        self.log_message.emit("ChArUco board initialized successfully.")
        return True

    def load_images(self, file_paths):
        """Load images and detect ChArUco corners. Emits images_loaded on completion."""
        self.loaded_images.clear()
        self.detected_markers.clear()
        self.detected_corners.clear()
        self.image_names.clear()

        for file_path in file_paths:
            try:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    self.log_message.emit(f"Warning: Could not read {file_path}")
                    continue

                filename = os.path.splitext(os.path.basename(file_path))[0]
                self.loaded_images.append(img)
                self.image_names.append(filename)

                try:
                    corners, ids, _ = cv2.aruco.detectMarkers(img, self._aruco_dict)
                    self.detected_markers.append((corners, ids))
                except Exception as e:
                    self.log_message.emit(f"Marker detection error for {filename}: {e}")
                    self.detected_markers.append((None, None))
                    self.detected_corners.append((None, None))
                    continue

                try:
                    if ids is not None and len(ids) > 0:
                        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                            markerCorners=corners, markerIds=ids,
                            image=img, board=self.charuco_board)
                        self.detected_corners.append((charuco_corners, charuco_ids))
                        if charuco_ids is not None:
                            self.log_message.emit(f"{filename}: Detected {len(charuco_ids)} ChArUco corners.")
                        else:
                            self.log_message.emit(f"{filename}: interpolateCornersCharuco returned no charuco_ids.")
                    else:
                        self.detected_corners.append((None, None))
                        self.log_message.emit(f"{filename}: No markers found.")
                except Exception as e:
                    self.log_message.emit(f"Charuco interpolation error for {filename}: {e}")
                    self.detected_corners.append((None, None))

            except Exception as e:
                self.log_message.emit(f"Unexpected error processing {file_path}: {e}")
                continue

        self.last_image_dir = os.path.dirname(file_paths[0]) if file_paths else ""
        self.log_message.emit(f"Loaded and processed {len(self.loaded_images)} images.")
        self.images_loaded.emit(list(self.image_names))

    def run_calibration(self):
        """Run camera calibration. Emits calibration_done on success."""
        if self.charuco_board is None:
            self.error_occurred.emit("Error", "No ChArUco board initialized. Please load images first.")
            return
        if not self.detected_corners or all(c[0] is None for c in self.detected_corners):
            self.error_occurred.emit("Error", "No valid ChArUco corners detected. Please load valid images.")
            return
        if not self.loaded_images:
            self.error_occurred.emit("Error", "No images loaded.")
            return

        all_corners, all_ids, image_size = [], [], None
        for i, (corners, ids) in enumerate(self.detected_corners):
            if corners is not None and ids is not None and len(corners) > 3:
                all_corners.append(corners)
                all_ids.append(ids)
                if image_size is None:
                    h, w = self.loaded_images[i].shape[:2]
                    image_size = (w, h)

        if not all_corners:
            self.error_occurred.emit("Error", "No sufficient ChArUco corners found for calibration.")
            return

        try:
            self.log_message.emit(f"Running calibrateCameraCharucoExtended on {len(all_corners)} valid images...")
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 1000, 1e-9)
            (retval, cam_mat, dist_coeffs, rvecs, tvecs,
             std_int, std_ext, per_view) = cv2.aruco.calibrateCameraCharucoExtended(
                charucoCorners=all_corners, charucoIds=all_ids,
                board=self.charuco_board, imageSize=image_size,
                cameraMatrix=None, distCoeffs=None, criteria=criteria)

            self.camera_matrix      = cam_mat
            self.dist_coeffs        = dist_coeffs
            self.reprojection_error = retval
            self.rvecs              = rvecs
            self.tvecs              = tvecs
            self.std_dev_intrinsics = std_int
            self.std_dev_extrinsics = std_ext
            self.per_view_errors    = per_view

            self.log_message.emit("Calibration completed successfully.")
            self.log_message.emit(f"Reprojection Error: {retval}")
            self.log_message.emit(f"Camera Matrix:\n{cam_mat}")
            self.log_message.emit(f"Distortion Coefficients:\n{dist_coeffs}")

            self.calibration_done.emit({
                "fx":  float(cam_mat[0, 0]),
                "fy":  float(cam_mat[1, 1]),
                "ppx": float(cam_mat[0, 2]),
                "ppy": float(cam_mat[1, 2]),
                "K1":  float(dist_coeffs[0, 0]),
                "K2":  float(dist_coeffs[0, 1]),
                "K3":  float(dist_coeffs[0, 4]),
                "P1":  float(dist_coeffs[0, 2]),
                "P2":  float(dist_coeffs[0, 3]),
            })
        except Exception as e:
            tb = traceback.format_exc()
            self.error_occurred.emit("Calibration Error",
                                     f"An error occurred during calibration:\n{e}\n\nTraceback:\n{tb}")

    def save_calibration(self):
        """Save calibration results to calibration.json in the image folder. Emits save_done on success."""
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.error_occurred.emit("Save Calibration", "No calibration results. Please run calibration first.")
            return
        if not self.loaded_images:
            self.error_occurred.emit("Save Calibration", "No images loaded. Cannot determine output directory.")
            return
        if not self.last_image_dir:
            self.error_occurred.emit("Save Calibration", "Cannot determine where images were loaded from.")
            return

        try:
            os.makedirs(self.last_image_dir, exist_ok=True)
            out_path = os.path.join(self.last_image_dir, "calibration.json")

            cm = self.camera_matrix
            dc = self.dist_coeffs.flatten()
            if len(dc) < 5:
                self.error_occurred.emit("Save Calibration",
                                         f"Unexpected distortion coefficient length: {len(dc)}")
                return

            calib_data = {
                "intrinsics": {
                    "focalLengthX":    float(cm[0, 0]),
                    "focalLengthY":    float(cm[1, 1]),
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
                    "rmsError":      float(self.reprojection_error) if self.reprojection_error is not None else None,
                    "numImagesUsed": int(len(self.rvecs)) if self.rvecs is not None else 0
                }
            }

            img_data = {}
            for i in range(len(self.loaded_images)):
                charuco_corners, charuco_ids = self.detected_corners[i]
                if charuco_corners is None or charuco_ids is None:
                    continue  # bug fix: skip images with no detected corners
                img_name = self.image_names[i] if i < len(self.image_names) else f"image_{i}"
                flat_ids = charuco_ids.reshape((-1,))
                img_data[img_name] = {
                    int(id_): {"x": float(pt[0, 0]), "y": float(pt[0, 1])}
                    for id_, pt in zip(flat_ids, charuco_corners)
                }

            calib_data["data"] = img_data

            with open(out_path, "w") as f:
                json.dump(calib_data, f, indent=4)

            self.log_message.emit(f"Calibration JSON saved to {out_path}")
            self.save_done.emit(out_path)

        except Exception as e:
            tb = traceback.format_exc()
            self.error_occurred.emit("Save Calibration Error",
                                     f"Failed to save calibration:\n{e}\n\nTraceback:\n{tb}")

    def remove_images(self, indexes):
        """Remove images at the given indexes (processed descending to keep indexes valid)."""
        for i in sorted(indexes, reverse=True):
            if 0 <= i < len(self.loaded_images):
                del self.loaded_images[i]
                del self.detected_markers[i]
                del self.detected_corners[i]
                if i < len(self.image_names):
                    del self.image_names[i]

    def clear_all(self):
        """Clear all loaded data and calibration results."""
        self.loaded_images.clear()
        self.detected_markers.clear()
        self.detected_corners.clear()
        self.image_names.clear()
        self.camera_matrix      = None
        self.dist_coeffs        = None
        self.reprojection_error = None
        self.rvecs              = None
        self.tvecs              = None
        self.log_message.emit("All images and associated data cleared.")

    # ------------------------------------------------------------------
    # Private OpenCV helpers (unchanged logic, moved from CalibrationGui)
    # ------------------------------------------------------------------

    def _get_aruco_dictionary(self, dict_name):
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
            raise RuntimeError("This OpenCV build does not expose a dictionary getter "
                               "(getPredefinedDictionary / Dictionary_get).")

    def _create_charuco_board(self, squares_x, squares_y, square_length, marker_length, aruco_dict):
        if hasattr(cv2.aruco, "CharucoBoard_create"):
            try:
                return cv2.aruco.CharucoBoard_create(
                    squares_x, squares_y, square_length, marker_length, aruco_dict)
            except Exception:
                pass
        if hasattr(cv2.aruco, "CharucoBoard") and hasattr(cv2.aruco.CharucoBoard, "create"):
            try:
                return cv2.aruco.CharucoBoard.create(
                    squares_x, squares_y, square_length, marker_length, aruco_dict)
            except Exception:
                pass
        try:
            return cv2.aruco.CharucoBoard(
                (squares_x, squares_y), square_length, marker_length, aruco_dict)
        except Exception as e:
            raise AttributeError(
                f"Unable to create CharucoBoard. Neither 'CharucoBoard_create', "
                f"'CharucoBoard.create', nor constructor succeeded. Error: {e}")
