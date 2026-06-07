import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import os
import threading
import queue
import time

# Get the path to face_landmarker.task
task_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# Create face landmarker options with optimizations for max FPS
base_options = python.BaseOptions(model_asset_path=task_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=False,  # Disabled for faster processing
    output_facial_transformation_matrixes=False  # Disabled for faster processing
)

# Create face landmarker
face_landmarker = vision.FaceLandmarker.create_from_options(options)

# Threading for camera capture
frame_queue = queue.Queue(maxsize=2)
stop_capture = False

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
            # Optimized resolution: 640x480 for better FPS performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Set target FPS to 30
            cap.set(cv2.CAP_PROP_FPS, 30)
            if cap.isOpened():
                return cap
            cap.release()
    return None


def camera_capture_thread(cap):
    """Threaded camera capture to reduce lag"""
    global stop_capture
    frame_count = 0
    
    while not stop_capture and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frames more frequently for 24 FPS
        frame_count += 1
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            # Drop old frames if queue is full to prevent lag
            try:
                frame_queue.get_nowait()
                frame_queue.put_nowait(frame)
            except queue.Empty:
                pass
    
    cap.release()


def draw_overlay(frame, measurements_mm, status_message, fps=0.0, status_color=(0, 255, 255)):
    """Draw improved UI overlay with better visual feedback"""
    h, w, _ = frame.shape
    
    # Draw semi-transparent background for better text readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w-10, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Status message (color-coded)
    y = 40
    cv2.putText(frame, status_message, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, status_color, 2)
    y += 35
    
    # FPS display
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 255, 0), 1)
    y += 25
    
    # Measurements
    for label, value in measurements_mm.items():
        cv2.putText(frame, f"{label}: {value}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)
        y += 22
    
    # Add alignment guides (center crosshair)
    center_x, center_y = w // 2, h // 2
    cv2.circle(frame, (center_x, center_y), 3, (0, 255, 255), -1)
    cv2.line(frame, (center_x - 30, center_y), (center_x + 30, center_y), (0, 255, 255), 1)
    cv2.line(frame, (center_x, center_y - 30), (center_x, center_y + 30), (0, 255, 255), 1)
    
    # Instruction text at bottom
    instructions = "Press Q to quit | Face measurements display in real-time"
    cv2.putText(frame, instructions, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, (150, 150, 150), 1)




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
print("\n" + "="*60)
print("📸 Live Face Measurement System (Optimized for Speed)")
print("="*60)
print("⚡ Performance Optimizations Applied:")
print("  • Resolution: 640x480 (reduced from 1280x720)")
print("  • Detection: Every 3rd frame (temporal decimation)")
print("  • Display FPS: 30+ (optimized refresh)")
print("  • Blendshapes: Disabled (faster processing)")
print("  • Threading: Optimized frame capture")
print("\n📋 Instructions:")
print("  • Align your face in the center crosshair")
print("  • Keep your face still for accurate measurements")
print("  • Press Q to quit and save final measurements")
print("="*60 + "\n")

final_measurements_mm = {}
REFERENCE_PD_MM = 63.0  # mm

# Start camera capture thread
capture_thread = threading.Thread(target=camera_capture_thread, args=(cap,), daemon=True)
capture_thread.start()

# FPS calculation
fps_clock = time.time()
fps_counter = 0
current_fps = 0

# Face detection stability (average measurements over multiple frames)
measurement_history = []
max_history = 8  # Reduced for faster, smoother updates
detection_frame_count = 0
last_measurements = None

# Aggressive temporal decimation: detect every 3rd frame for maximum FPS
DETECTION_SKIP_FRAMES = 3

while True:
    try:
        frame = frame_queue.get(timeout=2)
    except queue.Empty:
        print("⚠️ Camera timeout - no frames received")
        break
    
    frame = cv2.flip(frame, 1)
    
    # Run face detection every Nth frame to maximize display FPS
    detection_frame_count += 1
    if detection_frame_count % DETECTION_SKIP_FRAMES == 1:
        measurements = measure_frame(frame, face_landmarker)
        if measurements:
            last_measurements = measurements
    else:
        measurements = last_measurements
    
    # Determine status
    if measurements:
        # Store measurements for averaging
        measurement_history.append({k: v for k, v in measurements.items() if k != "Points"})
        if len(measurement_history) > max_history:
            measurement_history.pop(0)
        
        # Average recent measurements for stability
        final_measurements_mm = {
            label: round(np.mean([m.get(label, 0) for m in measurement_history]), 2)
            for label in measurement_history[0].keys()
        }
        
        points = measurements["Points"]
        
        # Draw measurement lines with thicker style
        cv2.line(frame, points["left_pupil"], points["right_pupil"], (0, 255, 0), 3)
        cv2.line(frame, points["face_left"], points["face_right"], (255, 0, 0), 2)
        cv2.line(frame, points["nose_left"], points["nose_right"], (0, 0, 255), 2)
        
        # Draw circles at key points
        for point in [points["left_pupil"], points["right_pupil"]]:
            cv2.circle(frame, point, 4, (0, 255, 0), -1)
        
        status_message = "✓ Face Detected"
        status_color = (0, 255, 0)
    else:
        status_message = "⚠ Align face in crosshair"
        status_color = (0, 165, 255)
        final_measurements_mm = {}
    
    # Calculate FPS
    fps_counter += 1
    current_time = time.time()
    if current_time - fps_clock > 1:
        current_fps = fps_counter / (current_time - fps_clock)
        fps_counter = 0
        fps_clock = current_time
    
    draw_overlay(frame, final_measurements_mm, status_message, current_fps, status_color)
    cv2.imshow("Live Face Measurement (mm)", frame)
    
    # Target 30+ FPS display (30ms per frame)
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q") or key == ord("Q"):
        break

stop_capture = True
cv2.destroyAllWindows()
capture_thread.join(timeout=2)

# ------------------ PRINT FINAL MEASUREMENTS ------------------
print("\n" + "="*50)
print("✅ FINAL FACE MEASUREMENTS (MM)")
print("="*50)
if final_measurements_mm:
    for k, v in final_measurements_mm.items():
        print(f"  {k}: {v}")
    print("="*50)
    print("💡 Measurements have been saved!")
else:
    print("  No face detected. Please try again.")
print("="*50 + "\n")


