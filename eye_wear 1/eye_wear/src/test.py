import cv2
import mediapipe as mp

BaseOption = mp.solutions.base_solution.BaseSolution
FaceLandmark = mp.solutions.face_mesh.FaceMeshLandmark
FaceLandmarkOptions = mp.solutions.face_mesh.FaceMeshOptions
VisionRunningMode = mp.solutions.base_solution.VisionRunningMode

options = FaceLandmarkOptions(
    base_options=BaseOptions(model_asset_path="face_landmark_front.tflite"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
)

cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(options) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            for lm in result.face_landmarks[0]:
                  h, w = frame.shape[:2]
                  cx, cy = int(lm.x * w), int(lm.y * h)
                  cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)
  
        cv2.imshow("Face Landmarks", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
              break
  
cap.release()
cv2.destroyAllWindows()
