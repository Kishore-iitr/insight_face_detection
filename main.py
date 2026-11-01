
import cv2
import numpy as np
import time
from pathlib import Path
from insightface.app import FaceAnalysis

# =====================================================
# CONFIGURATION
# =====================================================

DATABASE_PATH = "student_database"

CAMERA_INDEX = 0

# Camera processing resolution
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 480

# Display resolution
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 720

# Run face recognition once every N frames
RECOGNITION_INTERVAL = 3

# Face recognition settings
SIMILARITY_THRESHOLD = 0.45
MIN_MATCH_MARGIN = 0.05

# Smaller detection input = faster, but may miss distant faces
DETECTION_SIZE = (480, 480)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# =====================================================
# INITIALIZE MODEL
# =====================================================

print("Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=DETECTION_SIZE
)

print("Model loaded!")


# =====================================================
# NORMALIZE EMBEDDINGS
# =====================================================

def normalize_embedding(embedding):

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm


# =====================================================
# LOAD STUDENT DATABASE
# =====================================================

def load_student_database():

    database = {}

    root = Path(DATABASE_PATH)

    if not root.exists():
        raise FileNotFoundError(
            "student_database folder not found!"
        )

    for folder in root.iterdir():

        if not folder.is_dir():
            continue

        person_name = folder.name
        embeddings = []

        print("\nLoading:", person_name)

        for image_path in folder.iterdir():

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            faces = app.get(image)

            if len(faces) == 0:
                print("No face:", image_path.name)
                continue

            # Largest face in reference image
            face = max(
                faces,
                key=lambda f: (
                    (f.bbox[2] - f.bbox[0]) *
                    (f.bbox[3] - f.bbox[1])
                )
            )

            embeddings.append(
                normalize_embedding(face.embedding)
            )

            print("Loaded:", image_path.name)

        if embeddings:
            database[person_name] = np.stack(embeddings)

            print(
                "Reference images loaded:",
                len(embeddings)
            )

    return database


# =====================================================
# BUILD DATABASE
# =====================================================

print("\nBuilding student database...")

student_database = load_student_database()

if not student_database:
    raise RuntimeError("No valid student faces found!")

print("\nDatabase ready!")
print("Registered people:", len(student_database))


# =====================================================
# FAST VECTOR-BASED RECOGNITION
# =====================================================

def recognize_face(face_embedding):

    query = normalize_embedding(face_embedding)

    person_scores = {}

    for person_name, references in student_database.items():

        # Compare query against all reference images at once
        similarities = references @ query

        person_scores[person_name] = float(
            np.max(similarities)
        )

    ranked = sorted(
        person_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best_name, best_score = ranked[0]

    second_score = (
        ranked[1][1] if len(ranked) > 1 else -1.0
    )

    margin = best_score - second_score

    if (
        best_score >= SIMILARITY_THRESHOLD
        and margin >= MIN_MATCH_MARGIN
    ):
        return best_name, best_score

    return "Unknown", best_score


# =====================================================
# INITIALIZE WEBCAM
# =====================================================

print("\nOpening webcam...")

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():
    raise RuntimeError(
        "Cannot access webcam. Try CAMERA_INDEX = 1."
    )

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    PROCESS_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    PROCESS_HEIGHT
)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Webcam started!")


# =====================================================
# STATISTICS
# =====================================================

recognized_people = set()

cached_faces = []

frame_count = 0

fps = 0
fps_timer = time.time()
fps_frames = 0


# =====================================================
# MAIN WEBCAM LOOP
# =====================================================

print("\nLive recognition started!")
print("Press Q to exit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame unavailable.")
        break

    frame_count += 1
    fps_frames += 1

    # ---------------------------------------------
    # RESIZE FRAME FOR FASTER PROCESSING
    # ---------------------------------------------

    frame = cv2.resize(
        frame,
        (PROCESS_WIDTH, PROCESS_HEIGHT)
    )

    # ---------------------------------------------
    # RUN RECOGNITION ONLY EVERY N FRAMES
    # ---------------------------------------------

    if frame_count % RECOGNITION_INTERVAL == 0:

        faces = app.get(frame)

        cached_faces = []

        for face in faces:

            x1, y1, x2, y2 = face.bbox.astype(int)

            name, confidence = recognize_face(
                face.embedding
            )

            cached_faces.append(
                (x1, y1, x2, y2, name, confidence)
            )

            if name != "Unknown":
                recognized_people.add(name)

    # ---------------------------------------------
    # DRAW CACHED FACE BOXES
    # ---------------------------------------------

    visible_people = set()

    for x1, y1, x2, y2, name, confidence in cached_faces:

        if name == "Unknown":

            color = (0, 165, 255)

        else:

            color = (0, 255, 0)
            visible_people.add(name)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            name,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

    # ---------------------------------------------
    # FPS CALCULATION
    # ---------------------------------------------

    elapsed = time.time() - fps_timer

    if elapsed >= 1.0:

        fps = fps_frames / elapsed

        fps_frames = 0
        fps_timer = time.time()

    # ---------------------------------------------
    # STATISTICS PANEL
    # ---------------------------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (420, 145),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        "CLASSROOM FACE RECOGNITION",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Visible faces: " + str(len(cached_faces)),
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Known visible: " + str(len(visible_people)),
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Unique recognized: " + str(len(recognized_people)),
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Processing FPS: " + str(round(fps, 1)),
        (20, 138),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 0),
        1
    )

    # ---------------------------------------------
    # RESIZE FOR DISPLAY
    # ---------------------------------------------

    display_frame = cv2.resize(
        frame,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT)
    )

    cv2.imshow(
        "Optimized Classroom Face Recognition",
        display_frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# =====================================================
# CLEANUP
# =====================================================

cap.release()
cv2.destroyAllWindows()


# =====================================================
# FINAL RESULTS
# =====================================================

print("\n========== FINAL RESULTS ==========")

print(
    "Unique registered people recognized:",
    len(recognized_people)
)

for name in sorted(recognized_people):
    print("-", name)

print("\nProgram terminated.")
