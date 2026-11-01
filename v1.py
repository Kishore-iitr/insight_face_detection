## only bounding boxes

import cv2
import time

# ==========================================
# CONFIGURATION
# ==========================================

VIDEO_PATH = r"C:\Users\KISHORE S\Desktop\nto assign\classroom 2.mp4"

MODEL_PATH = r"C:\Users\KISHORE S\Desktop\nto assign\face_detection_yunet_2023mar.onnx"

DETECTION_WIDTH = 640
DISPLAY_WIDTH = 960

DETECT_EVERY_N_FRAMES = 3
CONFIDENCE_THRESHOLD = 0.75

# ==========================================
# INITIALIZE
# ==========================================

cv2.setNumThreads(4)

detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (DETECTION_WIDTH, 360),
    CONFIDENCE_THRESHOLD,
    0.3,
    5000
)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Cannot open video. Check VIDEO_PATH.")
    exit()

source_fps = cap.get(cv2.CAP_PROP_FPS)

if source_fps <= 0:
    source_fps = 25

frame_delay = 1.0 / source_fps

# Window
cv2.namedWindow(
    "Classroom Face Counter",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Classroom Face Counter",
    960,
    600
)

# ==========================================
# VARIABLES
# ==========================================

frame_number = 0
face_count = 0
cached_faces = []

print("Video processing started...")

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    loop_start = time.time()

    ret, frame = cap.read()

    if not ret:
        print("Video completed.")
        break

    frame_number += 1

    original_height, original_width = frame.shape[:2]

    # --------------------------------------
    # RESIZE FOR FASTER DETECTION
    # --------------------------------------

    detection_height = int(
        original_height *
        DETECTION_WIDTH /
        original_width
    )

    small_frame = cv2.resize(
        frame,
        (DETECTION_WIDTH, detection_height),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------
    # FACE DETECTION EVERY N FRAMES
    # --------------------------------------

    if frame_number % DETECT_EVERY_N_FRAMES == 1 or frame_number == 1:

        detector.setInputSize(
            (DETECTION_WIDTH, detection_height)
        )

        _, faces = detector.detect(small_frame)

        cached_faces = []

        if faces is not None:

            for face in faces:

                x, y, w, h = face[:4]

                confidence = float(face[-1])

                cached_faces.append(
                    (int(x), int(y), int(w), int(h), confidence)
                )

        face_count = len(cached_faces)

    # --------------------------------------
    # RESIZE FOR DISPLAY
    # --------------------------------------

    display_height = int(
        original_height *
        DISPLAY_WIDTH /
        original_width
    )

    display_frame = cv2.resize(
        frame,
        (DISPLAY_WIDTH, display_height),
        interpolation=cv2.INTER_AREA
    )

    scale_x = DISPLAY_WIDTH / DETECTION_WIDTH
    scale_y = display_height / detection_height

    # --------------------------------------
    # DRAW BOUNDING BOXES
    # --------------------------------------

    for x, y, w, h, confidence in cached_faces:

        x1 = int(x * scale_x)
        y1 = int(y * scale_y)

        x2 = int((x + w) * scale_x)
        y2 = int((y + h) * scale_y)

        cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = "Face " + str(round(confidence, 2))

        cv2.putText(
            display_frame,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # --------------------------------------
    # DISPLAY FACE COUNT
    # --------------------------------------

    cv2.rectangle(
        display_frame,
        (10, 10),
        (310, 85),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        display_frame,
        "FACES: " + str(face_count),
        (20, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    # --------------------------------------
    # DISPLAY FPS
    # --------------------------------------

    elapsed = time.time() - loop_start

    processing_fps = 1.0 / max(elapsed, 0.000001)

    cv2.putText(
        display_frame,
        "Processing FPS: " + str(round(processing_fps, 1)),
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # --------------------------------------
    # SHOW VIDEO
    # --------------------------------------

    cv2.imshow(
        "Classroom Face Counter",
        display_frame
    )

    # Maintain approximate original playback speed
    remaining = frame_delay - (time.time() - loop_start)

    wait_time = max(1, int(remaining * 1000))

    key = cv2.waitKey(wait_time) & 0xFF

    if key == ord('q'):
        break

    # Pause / Resume
    if key == ord(' '):

        while True:

            key = cv2.waitKey(0) & 0xFF

            if key == ord(' '):
                break

            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("Face counting completed!")
