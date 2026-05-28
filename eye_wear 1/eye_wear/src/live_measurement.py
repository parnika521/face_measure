import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import os

# Get the path to face_landmarker.task
task_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# Create face landmarker options
base_options = python.BaseOptions(model_asset_path=task_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
)

# Create face landmarker
face_landmarker = vision.FaceLandmarker.create_from_options(options)

# ------------------ LANDMARK IDS ------------------
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
LEFT_PUPIL = 468
RIGHT_PUPIL = 473
FACE_WIDTH = [234, 454]
NOSE_BRIDGE = [94, 331]

# ------------------ UTILS ------------------
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def open_camera():
    requested_index = os.environ.get("CAMERA_INDEX")
    candidate_indexes = []

    if requested_index is not None:
        try:
            candidate_indexes.append(int(requested_index))
        except ValueError:
            pass

    candidate_indexes.extend([0, 1, 2, 3])

    candidate_backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    for camera_index in dict.fromkeys(candidate_indexes):
        for backend in candidate_backends:
            cap = cv2.VideoCapture(camera_index, backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                return cap
            cap.release()
    return None


def draw_overlay(frame, measurements_mm, status_message):
    y = 30
    cv2.putText(frame, status_message, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y += 34

    for label, value in measurements_mm.items():
        cv2.putText(frame, f"{label}: {value}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 28


def measure_frame(frame, face_landmarker):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]

    def get_point(landmark_list, idx):
        lm = landmark_list[idx]
        return int(lm.x * w), int(lm.y * h)

    left_pupil = get_point(landmarks, LEFT_PUPIL)
    right_pupil = get_point(landmarks, RIGHT_PUPIL)
    face_left = get_point(landmarks, FACE_WIDTH[0])
    face_right = get_point(landmarks, FACE_WIDTH[1])
    nose_left = get_point(landmarks, NOSE_BRIDGE[0])
    nose_right = get_point(landmarks, NOSE_BRIDGE[1])

    left_eye_w = distance(
        get_point(landmarks, LEFT_EYE[0]),
        get_point(landmarks, LEFT_EYE[1])
    )

    right_eye_w = distance(
        get_point(landmarks, RIGHT_EYE[0]),
        get_point(landmarks, RIGHT_EYE[1])
    )

    pd_px = distance(left_pupil, right_pupil)
    if pd_px < 1.0:
        return None

    mm_per_px = REFERENCE_PD_MM / pd_px

    return {
        "Pupillary Distance (mm)": round(pd_px * mm_per_px, 2),
        "Face Width (mm)": round(distance(face_left, face_right) * mm_per_px, 2),
        "Bridge Width (mm)": round(distance(nose_left, nose_right) * mm_per_px, 2),
        "Left Lens Width (mm)": round(left_eye_w * mm_per_px, 2),
        "Right Lens Width (mm)": round(right_eye_w * mm_per_px, 2),
        "Points": {
            "left_pupil": left_pupil,
            "right_pupil": right_pupil,
            "face_left": face_left,
            "face_right": face_right,
            "nose_left": nose_left,
            "nose_right": nose_right,
        },
    }

# ------------------ MAIN ------------------
cap = open_camera()
print(f"Camera opened: {cap}")  # See if camera opened successfully

if cap is None:
    raise RuntimeError(
        "Error: No webcam could be opened. On Linux, make sure your user can access /dev/video* "
        "(for example by being in the video group), then reconnect the camera and retry."
    )

print("\n📸 Live face measurement started (mm mode)")
print("➡️ Press Q to quit and print final measurements\n")

final_measurements_mm = {}

# Average PD for scaling
REFERENCE_PD_MM = 63.0  # mm

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    measurements = measure_frame(frame, face_landmarker)

    if measurements:
        final_measurements_mm = {k: v for k, v in measurements.items() if k != "Points"}
        points = measurements["Points"]

        cv2.line(frame, points["left_pupil"], points["right_pupil"], (0, 255, 0), 2)
        cv2.line(frame, points["face_left"], points["face_right"], (255, 0, 0), 2)
        cv2.line(frame, points["nose_left"], points["nose_right"], (0, 0, 255), 2)

        status_message = "Face detected. Press Q to quit."
    else:
        status_message = "Align your face in the camera view."

    draw_overlay(frame, final_measurements_mm, status_message)
    cv2.imshow("Live Face Measurement (mm)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# ------------------ PRINT FINAL MEASUREMENTS ------------------
print("\n✅ FINAL FACE MEASUREMENTS (MM)")
print("-----------------------------------")
if final_measurements_mm:
    for k, v in final_measurements_mm.items():
        print(f"{k}: {v}")
else:
    print("No face detected. Please try again.")
