r"""
Verification driver for CalibrationModel.load_images() dimension handling.

Exercises the fail-fast same-dimension check and the 90-degree rotation rescue
by driving the real model with live Qt signals connected (as the View would).

Run from anywhere:
    .\CamCalib\Scripts\python.exe src\unittests\test_image_dimensions.py
"""
import os, sys, tempfile, shutil
import cv2

# --- Make the model importable and locate the sample image regardless of CWD ---
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src
sys.path.insert(0, SRC_DIR)
SAMPLE = os.path.join(SRC_DIR, "tmp", "charuco_6x6_250_1080p_opencv4.12.0.png")

from PyQt5.QtWidgets import QApplication
from CalibrationModel import CalibrationModel

app = QApplication(sys.argv)

base = cv2.imread(SAMPLE, cv2.IMREAD_GRAYSCALE)
assert base is not None, f"sample image missing: {SAMPLE}"
print("base shape (h,w):", base.shape)            # (1080,1920)

d = tempfile.mkdtemp(prefix="cc_verify_")
p_ref   = os.path.join(d, "ref.png")              # 1080x1920
p_match = os.path.join(d, "match.png")            # 1080x1920 (same)
p_trans = os.path.join(d, "transposed.png")       # 1920x1080 -> should rotate
p_diff  = os.path.join(d, "different.png")        # 600x800   -> should fail
cv2.imwrite(p_ref,   base)
cv2.imwrite(p_match, base.copy())
cv2.imwrite(p_trans, cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE))   # shape (1920,1080)
cv2.imwrite(p_diff,  cv2.resize(base, (800, 600)))                # shape (600,800)


def make_model():
    m = CalibrationModel()
    logs, errors, loaded = [], [], []
    m.log_message.connect(lambda s: logs.append(s))
    m.error_occurred.connect(lambda t, msg: errors.append((t, msg)))
    m.images_loaded.connect(lambda names: loaded.append(list(names)))
    ok = m.initialize_board(4, 6, 0.042, 0.031, "DICT_6X6_250", True)
    assert ok, "board init failed"
    return m, logs, errors, loaded


def run(title, paths):
    m, logs, errors, loaded = make_model()
    m.load_images(paths)
    print("\n=== " + title + " ===")
    print("  loaded_images count :", len(m.loaded_images))
    print("  image_names         :", m.image_names)
    print("  stored shapes       :", [im.shape for im in m.loaded_images])
    print("  error_occurred      :", errors)
    print("  images_loaded signal:", loaded)
    print("  rotation log lines  :", [l for l in logs if "rotated" in l.lower()])
    return m, logs, errors, loaded


# Scenario A: ref + match + transposed  -> 3 loaded, one rotated, no error
run("A: ref + match + transposed (expect 3 loaded, 1 rotated, no error)",
    [p_ref, p_match, p_trans])

# Scenario B: ref + different -> abort, 0 loaded, error emitted
run("B: ref + different (expect 0 loaded, error emitted)",
    [p_ref, p_diff])

# Probe C: transposed FIRST sets reference, then ref must rotate
run("C (probe): transposed first, then ref (expect 2 loaded, 1 rotated)",
    [p_trans, p_ref])

# Probe D: different appears AFTER a rotated one -> still aborts, nothing committed
run("D (probe): ref + transposed + different (expect 0 loaded, error, nothing committed)",
    [p_ref, p_trans, p_diff])

# Probe E: empty list
run("E (probe): empty file list (expect 0 loaded, no error)", [])

shutil.rmtree(d, ignore_errors=True)
print("\nDONE")
