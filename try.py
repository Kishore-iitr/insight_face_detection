import os
import cv2
import csv
import time
import requests
import numpy as np

from datetime import datetime
from insightface.app import FaceAnalysis

# =====================================================
# 1. CONFIGURATION
# =====================================================

# Replace this with the IP printed in ESP32 Serial Monitor.
ESP32_URL = "http://192.168.1.100/bmp"

DATABASE_PATH = "student_database"

# Use the model package you already downloaded.
MODEL_NAME = "buffalo_s"

# Initial threshold; calibrate with real camera images.
SIMILARITY_THRESHOLD = 0.45

# Minimum face detector confidence.
DETECTION_THRESHOLD = 0.50

# Display size
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# Camera request timeout
REQUEST_TIMEOUT = 5

# =====================================================
# 2. HELPER FUNCTIONS
# =====================================================

def normalize(vector):

    norm = np.linalg.norm(vector)

    return vector / (norm + 1e-10)


# =====================================================
# 3. LOAD BUFFALO MODEL
# =====================================================

print("\n====================================")
print("INITIALIZING BUFFALO FACE MODEL")
print("====================================")

app = FaceAnalysis(
    name=MODEL_NAME,
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(320, 320)
)

print("Buffalo model loaded successfully.")

# =====================================================
# 4. LOAD STUDENT DATABASE
# =====================================================

print("\n====================================")
print("LOADING STUDENT DATABASE")
print("====================================")

# Multiple reference embeddings per student.
student_embeddings = {}

if not os.path.isdir(DATABASE_PATH):
    raise FileNotFoundError(
        f"Student database not found: {DATABASE_PATH}"
    )

for student_name in sorted(os.listdir(DATABASE_PATH)):

    student_folder = os.path.join(
        DATABASE_PATH,
        student_name
    )

    if not os.path.isdir(student_folder):
        continue

    reference_vectors = []

    for filename in os.listdir(student_folder):

        image_path = os.path.join(
            student_folder,
            filename
        )

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            continue

        image = cv2.imread(image_path)

        if image is None:
            print("Could not read:", image_path)
            continue

        faces = app.get(image)

        if len(faces) == 0:
            print(
                f"No face detected in {student_name}/{filename}"
            )
            continue

        # Select the largest face in the reference image.
        face = max(
            faces,
            key=lambda f: (
                (f.bbox[2] - f.bbox[0]) *
                (f.bbox[3] - f.bbox[1])
            )
        )

        if face.det_score < DETECTION_THRESHOLD:
            continue

        embedding = normalize(face.embedding)

        reference_vectors.append(embedding)

        print(
            f"Loaded: {student_name}/{filename}"
        )

    if reference_vectors:

        student_embeddings[student_name] = reference_vectors

        print(
            f"{student_name}: "
            f"{len(reference_vectors)} reference embeddings"
        )

    else:

        print(
            f"WARNING: No usable images for {student_name}"
        )

if not student_embeddings:
    raise RuntimeError(
        "No student embeddings were loaded. Check your database."
    )

print("\nSTUDENT DATABASE READY")

print("Registered students:", len(student_embeddings))

# =====================================================
# 5. LIVE FACE MATCHING
# =====================================================

def recognize_face(live_embedding):

    best_name = "Unknown"
    best_score = -1.0

    for student_name, references in student_embeddings.items():

        # Compare against every reference image of this student.
        student_best_score = max(
            float(np.dot(live_embedding, reference))
            for reference in references
        )

        if student_best_score > best_score:

            best_score = student_best_score

            best_name = student_name

    if best_score < SIMILARITY_THRESHOLD:

        return "Unknown", best_score

    return best_name, best_score


# =====================================================
# 6. ATTENDANCE STORAGE
# =====================================================

# Each registered student is counted only once per session.
recognized_students = set()

# =====================================================
# 7. CONNECT TO ESP32
# =====================================================

session = requests.Session()

print("\n====================================")
print("CONNECTING TO ESP32 CAMERA")
print("====================================")

print("Camera URL:", ESP32_URL)

try:

    response = session.get(
        ESP32_URL,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    test_image = cv2.imdecode(
        np.frombuffer(response.content, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if test_image is None:
        raise RuntimeError("ESP32 returned an invalid image.")

    print(
        "Camera connected:",
        test_image.shape
    )

except Exception as error:

    raise RuntimeError(
        f"Could not connect to ESP32: {error}"
    )


# =====================================================
# 8. LIVE RECOGNITION LOOP
# =====================================================

print("\n====================================")
print("CLASSROOM RECOGNITION STARTED")
print("====================================")

previous_time = time.time()

frame_number = 0

visible_faces = 0
unknown_faces = 0

# Optional console reporting interval
last_console_update = time.time()

try:

    while True:

        # ---------------------------------------------
        # A. FETCH IMAGE FROM ESP32
        # ---------------------------------------------

        try:

            response = session.get(
                ESP32_URL,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            frame = cv2.imdecode(
                np.frombuffer(
                    response.content,
                    dtype=np.uint8
                ),
                cv2.IMREAD_COLOR
            )

            if frame is None:
                print("Invalid camera frame.")
                continue

        except requests.RequestException as error:

            print("Camera connection error:", error)

            time.sleep(0.5)

            continue

        frame_number += 1

        # ---------------------------------------------
        # B. DETECT ALL LIVE FACES USING BUFFALO
        # ---------------------------------------------

        faces = app.get(frame)

        visible_faces = len(faces)

        unknown_faces = 0

        registered_in_current_frame = set()

        # ---------------------------------------------
        # C. PROCESS EVERY DETECTED FACE
        # ---------------------------------------------

        for face in faces:

            if face.det_score < DETECTION_THRESHOLD:
                continue

            # ArcFace embedding from the live face
            live_embedding = normalize(
                face.embedding
            )

            # Compare with stored student embeddings
            identity, similarity = recognize_face(
                live_embedding
            )

            if identity == "Unknown":

                unknown_faces += 1

                color = (0, 0, 255)

            else:

                recognized_students.add(identity)

                registered_in_current_frame.add(identity)

                color = (0, 255, 0)

            # -----------------------------------------
            # D. DRAW BOUNDING BOX
            # -----------------------------------------

            x1, y1, x2, y2 = face.bbox.astype(int)

            # Scale coordinates to display resolution.
            original_height, original_width = frame.shape[:2]

            scale_x = DISPLAY_WIDTH / original_width
            scale_y = DISPLAY_HEIGHT / original_height

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)

            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            display_frame = cv2.resize(
                frame,
                (DISPLAY_WIDTH, DISPLAY_HEIGHT)
            )

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label = f"{identity} | {similarity:.2f}"

            cv2.putText(
                display_frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # ---------------------------------------------
        # E. DISPLAY FRAME AND STATISTICS
        # ---------------------------------------------

        # Ensure the display exists even if no faces were found.
        display_frame = cv2.resize(
            frame,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT)
        )

        # Draw the face boxes again on the final display.
        # This separate drawing pass keeps the display intact.

        for face in faces:

            if face.det_score < DETECTION_THRESHOLD:
                continue

            live_embedding = normalize(face.embedding)

            identity, similarity = recognize_face(
                live_embedding
            )

            x1, y1, x2, y2 = face.bbox.astype(int)

            scale_x = DISPLAY_WIDTH / frame.shape[1]
            scale_y = DISPLAY_HEIGHT / frame.shape[0]

            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)

            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)

            color = (
                (0, 255, 0)
                if identity != "Unknown"
                else (0, 0, 255)
            )

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                display_frame,
                f"{identity} ({similarity:.2f})",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # ---------------------------------------------
        # F. CALCULATE FPS
        # ---------------------------------------------

        current_time = time.time()

        fps = 1.0 / max(
            current_time - previous_time,
            1e-6
        )

        previous_time = current_time

        # ---------------------------------------------
        # G. DRAW STATISTICS
        # ---------------------------------------------

        cv2.rectangle(
            display_frame,
            (0, 0),
            (DISPLAY_WIDTH, 115),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            display_frame,
            f"Visible Faces: {visible_faces}",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Unknown Faces: {unknown_faces}",
            (15, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 100, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Unique Attendance: {len(recognized_students)}",
            (15, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_frame,
            f"FPS: {fps:.1f}",
            (500, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )

        cv2.imshow(
            "ESP32 Classroom Face Recognition",
            display_frame
        )

        # ---------------------------------------------
        # H. CONSOLE STATISTICS
        # ---------------------------------------------

        if time.time() - last_console_update >= 5:

            print("\n---------- LIVE STATISTICS ----------")

            print("Visible faces:", visible_faces)

            print("Unknown faces:", unknown_faces)

            print(
                "Unique registered students:",
                len(recognized_students)
            )

            print(
                "Recognized:",
                sorted(recognized_students)
            )

            print("-------------------------------------")

            last_console_update = time.time()

        # ---------------------------------------------
        # I. KEYBOARD CONTROLS
        # ---------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print("\nStopping recognition...")

            break

        elif key == ord("r"):

            recognized_students.clear()

            print("\nAttendance reset.")

        elif key == ord("s"):

            print("\nCURRENT ATTENDANCE")

            for student in sorted(recognized_students):
                print(student)

            print(
                "Total:",
                len(recognized_students)
            )

except KeyboardInterrupt:

    print("\nInterrupted by user.")

finally:

    # =================================================
    # 9. SAVE ATTENDANCE
    # =================================================

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_file = f"attendance_{timestamp}.csv"

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Student Name",
            "Attendance",
            "Timestamp"
        ])

        for student in sorted(recognized_students):

            writer.writerow([
                student,
                "Present",
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ])

    session.close()

    cv2.destroyAllWindows()

    print("\n====================================")
    print("FINAL ATTENDANCE")
    print("====================================")

    for student in sorted(recognized_students):

        print(student, "- Present")

    print(
        "\nTotal unique registered students:",
        len(recognized_students)
    )

    print(
        "Attendance saved to:",
        output_file
    )