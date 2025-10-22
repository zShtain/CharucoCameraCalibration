# Charuco Camera Calibration

A Python-based camera calibration tool that uses OpenCV and PyQt5 to provide a graphical interface for calibrating cameras with Charuco boards.

![Charuco Boards](assets/charucoCollage.png)

## Features

- Camera calibration using OpenCV
- GUI built with PyQt5
- Visualization of calibration results
- Easy setup with Python virtual environment

## Installation

Follow these steps to set up the project:

### 1. Create a virtual environment

```bash
python -m venv CamCalib
```

### 2. Activate the virtual environment

On **Windows**:

```bash
CamCalib\Scripts activate
```

### 3. Upgrade `pip`

```bash
CamCalib\Scripts\python -m pip install --upgrade pip
```

### 4. Install dependencies

```bash
CamCalib\Scripts\pip install numpy matplotlib pyqt5 opencv-contrib-python
```
Or install from requirements.txt file
```bash
CamCalib\Scripts\pip install -r requirement.txt
```

## Usage

After installation, run the main application script:

```bash
CamCalib\Scripts\python src\cameraCalibrationGUI.py
```

Make sure your charuco images are ready for calibration.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
