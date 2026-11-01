# Classroom Face Recognition

A Python/OpenCV project for detecting and recognizing students in classroom
camera feeds. It uses InsightFace embeddings for live webcam recognition and
also includes scripts for ESP32 camera frames and prerecorded video.

## Features

- Live webcam face recognition with known/unknown labels.
- Recognition from an ESP32 camera endpoint.
- Face detection and counting in prerecorded classroom videos.
- Multiple reference images per student.
- CPU execution support through InsightFace and ONNX Runtime.

## Project layout

```text
.
├── main.py                         # Recommended live webcam recognition
├── try.py                          # ESP32 camera recognition
├── v1.py                           # YuNet face detection/counting for video
├── face_detection_yunet_2023mar.onnx
└── student_database/
    └── <student-name>/*.jpg        # Reference images for each student
```

The video files in the repository are sample inputs and are not required for
webcam recognition.

## Requirements

- Python 3.9 or newer
- A webcam for `main.py`, or an HTTP camera endpoint for `try.py`
- Windows, Linux, or macOS with a display available for OpenCV windows

Install the Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install opencv-python numpy insightface onnxruntime requests
```

InsightFace may download the `buffalo_l` or `buffalo_s` model pack on its first
run. Internet access may therefore be required during the first launch.

## Student database

Create one folder per student under `student_database`. Add one or more clear
face images to that folder:

```text
student_database/
├── alice/
│   ├── front.jpg
│   └── side.jpg
└── bob/
    └── front.jpg
```

The folder name becomes the label displayed by the application. Use images
containing a single, clearly visible face where possible.

## Usage

### Live webcam recognition

`main.py` is the recommended entry point:

```bash
python main.py
```

Press `Q` to stop. The script opens the default camera (`CAMERA_INDEX = 0`),
loads the reference images, and displays recognized people and processing
statistics.

Useful settings are near the top of `main.py`, including camera index,
recognition interval, detection size, and similarity thresholds.

### ESP32 camera recognition

Edit `ESP32_URL` near the top of `try.py` to match the URL served by the ESP32
camera:

```python
ESP32_URL = "http://192.168.1.100/bmp"
```

Then run:

```bash
python try.py
```

The ESP32 and computer must be reachable on the same network. Press `Q` to
stop the display.

### Prerecorded video face counting

`v1.py` uses the bundled YuNet model to detect and count faces in a video. Edit
`VIDEO_PATH` and `MODEL_PATH` at the top of the file if the files are moved,
then run:

```bash
python v1.py
```

Press `Space` to pause or resume and `Q` to exit.

## Notes

- Recognition thresholds are environment-dependent. Calibrate
  `SIMILARITY_THRESHOLD` with representative camera images before relying on
  results.
- The scripts process biometric data. Store reference images securely, obtain
  appropriate consent, and follow applicable privacy and data-protection
  requirements.
- This project is intended for experimentation and classroom prototyping; it
  does not provide a complete attendance or access-control system.
