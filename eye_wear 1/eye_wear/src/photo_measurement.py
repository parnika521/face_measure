import cv2
import mediapipe as mp
import math
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.face_mesh as mp_face_mesh

# ----------------------------------
# MEDIAPIPE INITIALIZATION
# ----------------------------------
print("Mediapipe version:", mp.__version__)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ----------------------------------
# HELPER FUNCTIONS
# ----------------------------------

def to_pixel(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# ----------------------------------
# FACE MEASUREMENT FUNCTION
# ----------------------------------

def measure_face(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        print("No face detected")
        return

    landmarks = result.multi_face_landmarks[0].landmark

    # Key landmarks (MediaPipe indices)
    LEFT_TEMPLE = 234
    RIGHT_TEMPLE = 454
    LEFT_EYE = 33
    RIGHT_EYE = 263
    NOSE_BRIDGE = 168
    CHIN = 152

    left_temple = to_pixel(landmarks[LEFT_TEMPLE], w, h)
    right_temple = to_pixel(landmarks[RIGHT_TEMPLE], w, h)
    left_eye = to_pixel(landmarks[LEFT_EYE], w, h)
    right_eye = to_pixel(landmarks[RIGHT_EYE], w, h)
    nose_bridge = to_pixel(landmarks[NOSE_BRIDGE], w, h)
    chin = to_pixel(landmarks[CHIN], w, h)

    # Measurements (pixels)
    face_width = distance(left_temple, right_temple)
    eye_distance = distance(left_eye, right_eye)
    face_height = distance(nose_bridge, chin)

    # Print results
    print("\n--- FACE MEASUREMENTS (PIXELS) ---")
    print(f"Face width (temple–temple): {face_width:.2f}")
    print(f"Eye distance (outer–outer): {eye_distance:.2f}")
    print(f"Face height (nose–chin): {face_height:.2f}")

    # Draw measurement lines
    cv2.line(image, left_temple, right_temple, (0, 255, 0), 2)
    cv2.line(image, left_eye, right_eye, (255, 0, 0), 2)
    cv2.line(image, nose_bridge, chin, (0, 0, 255), 2)

    mp_drawing.draw_landmarks(
        image,
        result.multi_face_landmarks[0],
        mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )

    cv2.imshow("Spectacles Face Measurement", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ----------------------------------
# RUN
# ----------------------------------

if __name__ == "__main__":
    IMAGE_PATH = "data/images/front.jpg"

    measure_face(IMAGE_PATH)
