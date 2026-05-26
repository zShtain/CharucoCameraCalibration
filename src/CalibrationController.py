import traceback
import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QFileDialog

from CalibrationModel import CalibrationModel
from CalibrationView import CalibrationView


class CalibrationController(QObject):
    def __init__(self, model: CalibrationModel, view: CalibrationView):
        super().__init__()
        self._model = model
        self._view  = view
        self._connect_view_signals()
        self._connect_model_signals()
        self._apply_config()

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_view_signals(self):
        self._view.load_requested.connect(self._on_load_images)
        self._view.calibrate_requested.connect(self._on_run_calibration)
        self._view.save_requested.connect(self._on_save)
        self._view.clear_all_requested.connect(self._on_clear_all)
        self._view.clear_selected_requested.connect(self._on_clear_selected)
        self._view.image_selected.connect(self._on_image_selected)

    def _connect_model_signals(self):
        self._model.log_message.connect(print)
        self._model.images_loaded.connect(self._on_images_loaded)
        self._model.calibration_done.connect(self._on_calibration_done)
        self._model.save_done.connect(self._on_save_done)
        self._model.error_occurred.connect(self._view.show_error)

    def _apply_config(self):
        cfg = self._model.load_config()
        self._view.set_defaults(cfg)

    # ------------------------------------------------------------------
    # Slots for View signals
    # ------------------------------------------------------------------

    def _on_load_images(self):
        try:
            squares_x     = int(self._view.get_squares_x())
            squares_y     = int(self._view.get_squares_y())
            square_length = float(self._view.get_square_length())
            marker_length = float(self._view.get_marker_length())
        except ValueError:
            self._view.show_error(
                "Error",
                "Invalid numeric input. Please enter integer values for number of squares "
                "and numeric values for lengths.")
            return

        if marker_length >= square_length:
            self._view.show_error("Error", "Marker Length must be smaller than Square Length.")
            return

        if not self._model.initialize_board(squares_x, squares_y, square_length, marker_length,
                                             self._view.get_dict_name(), self._view.get_is_legacy()):
            return  # model already emitted error_occurred

        file_dialog = QFileDialog(self._view, "Select Image Files")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not file_dialog.exec_():
            return
        selected_files = file_dialog.selectedFiles()
        if not selected_files:
            return

        self._model.load_images(selected_files)

    def _on_run_calibration(self):
        self._model.run_calibration()

    def _on_save(self):
        self._model.save_calibration()

    def _on_clear_all(self):
        self._model.clear_all()
        self._view.clear_image_list()
        self._view.clear_results()
        self._view.clear_image_panels()

    def _on_clear_selected(self):
        rows = self._view.get_selected_rows()
        if not rows:
            print("No image selected for removal.")
            self._view.show_error("No Image Selected", "No image selected for removal.")
            return
        self._model.remove_images(rows)
        self._view.clear_image_list()
        for name in self._model.image_names:
            self._view.append_image_item(name)
        self._view.clear_image_panels()
        self._render_all_corners_plot()

    def _on_image_selected(self, index):
        if index < 0 or index >= len(self._model.loaded_images):
            return
        try:
            img = self._model.loaded_images[index]
            if img is None:
                return

            orig_h, orig_w = img.shape[:2]
            display_img    = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            target_width = max(1, self._view.get_panel_size(0)[0])

            if orig_w != target_width:
                scale         = target_width / float(orig_w)
                target_height = int(round(orig_h * scale))
                scaled_img    = cv2.resize(display_img, (target_width, target_height),
                                           interpolation=cv2.INTER_LINEAR)
            else:
                scale         = 1.0
                target_height = orig_h  # bug fix: always define target_height
                scaled_img    = display_img.copy()

            # Draw detected ChArUco corners
            if index < len(self._model.detected_corners):
                charuco_corners, _ = self._model.detected_corners[index]
                if charuco_corners is not None and len(charuco_corners) > 0:
                    for pt in charuco_corners:
                        if hasattr(pt, "shape") and pt.shape[-1] == 2:
                            x_f = float(pt[0][0]) if pt.ndim == 2 else float(pt[0])
                            y_f = float(pt[0][1]) if pt.ndim == 2 else float(pt[1])
                        else:
                            try:
                                x_f, y_f = float(pt[0][0]), float(pt[0][1])
                            except Exception:
                                continue
                        cv2.circle(scaled_img,
                                   (int(round(x_f * scale)), int(round(y_f * scale))),
                                   4, (255, 0, 0), -1)

            self._view.show_image(scaled_img, panel=0, scale_already_applied=True)

            # Show undistorted image if calibration is available
            if self._model.camera_matrix is not None and self._model.dist_coeffs is not None:
                try:
                    undistorted   = cv2.undistort(img, self._model.camera_matrix, self._model.dist_coeffs)
                    scaled_undist = cv2.resize(undistorted, (target_width, target_height),
                                               interpolation=cv2.INTER_LINEAR)
                    self._view.show_image(cv2.cvtColor(scaled_undist, cv2.COLOR_GRAY2BGR),
                                          panel=1, scale_already_applied=True)
                except Exception as e:
                    print(f"Undistortion failed for image {index}: {e}")

        except Exception as e:
            self._view.show_warning("Display Error", f"Failed to display image:\n{e}")

    # ------------------------------------------------------------------
    # Slots for Model signals
    # ------------------------------------------------------------------

    def _on_images_loaded(self, filenames):
        self._view.clear_image_list()
        for name in filenames:
            self._view.append_image_item(name)
        self._render_all_corners_plot()

    def _on_calibration_done(self, results):
        self._view.show_results(results)
        rms = self._model.reprojection_error
        n   = len(self._model.rvecs) if self._model.rvecs is not None else 0
        self._view.show_info(
            "Calibration Complete",
            f"Calibration successful!\n\nReprojection Error: {rms:.3f}\nImages used: {n}")

    def _on_save_done(self, path):
        self._view.show_info("Save Calibration", f"Calibration saved successfully to:\n{path}")

    # ------------------------------------------------------------------
    # Matplotlib scatter plot of all detected corners
    # ------------------------------------------------------------------

    def _render_all_corners_plot(self):
        try:
            if not self._model.loaded_images or not self._model.detected_corners:
                return

            h, w           = self._model.loaded_images[0].shape[:2]
            gv_w, gv_h     = self._view.get_panel_size(1)
            gv_w           = max(200, gv_w)
            gv_h           = max(200, gv_h)

            dpi    = 100
            fig, ax = plt.subplots(figsize=(gv_w / dpi, gv_h / dpi), dpi=dpi)
            ax.set_facecolor("white")
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.add_patch(plt.Rectangle((0, 0), w, h, linewidth=2,
                                        edgecolor="black", facecolor="none"))

            cmap = plt.get_cmap("tab10", len(self._model.detected_corners))
            for i, (corners, _) in enumerate(self._model.detected_corners):
                if corners is None or len(corners) == 0:
                    continue
                pts = corners.reshape(-1, 2)
                ax.scatter(pts[:, 0], pts[:, 1], s=10, color=cmap(i), label=f"Image {i + 1}")

            ax.set_aspect("equal", adjustable="box")
            fig.tight_layout(pad=0.0)

            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf_w, buf_h = canvas.get_width_height()
            image = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).reshape(buf_h, buf_w, 4)[..., :3]
            plt.close(fig)

            self._view.show_image(cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                  panel=1, scale_already_applied=True)

        except Exception as e:
            self._view.show_error("Corner Plot Error", f"Failed to plot detected corners:\n{e}")
            traceback.print_exc()
