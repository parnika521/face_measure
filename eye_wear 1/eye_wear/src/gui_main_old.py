import customtkinter as ctk
import cv2
import threading
import queue
import time
from PIL import Image, ImageTk
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Initialize face landmarker
task_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
base_options = python.BaseOptions(model_asset_path=task_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

# Frame queue for threading
frame_queue = queue.Queue(maxsize=2)
stop_capture = False

# Landmark IDs
LEFT_PUPIL = 468
RIGHT_PUPIL = 473
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
FACE_WIDTH = [234, 454]
NOSE_BRIDGE = [94, 331]
REFERENCE_PD_MM = 63.0


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def open_camera():
    """Open camera with optimized settings"""
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
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
            if cap.isOpened():
                return cap
            cap.release()
    return None


def camera_capture_thread(cap):
    """Threaded camera capture"""
    global stop_capture
    frame_count = 0
    
    while not stop_capture and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                frame_queue.get_nowait()
                frame_queue.put_nowait(frame)
            except queue.Empty:
                pass
    
    cap.release()


def measure_frame(frame):
    """Measure face landmarks from frame"""
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


# Set modern theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class FaceMeasurementGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title("Face Measurement System")
        self.geometry("1200x800")
        self.resizable(True, True)
        
        # Variables
        self.cap = None
        self.is_running = False
        self.measurement_history = []
        self.last_measurements = {}
        self.fps = 0.0
        self.fps_counter = 0
        self.fps_clock = time.time()
        
        # Create UI
        self.create_ui()
        
    def create_ui(self):
        """Create the main UI layout"""
        # Main container
        main_container = ctk.CTkFrame(self)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # ===== LEFT SIDE: VIDEO FEED =====
        left_frame = ctk.CTkFrame(main_container)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Video title
        video_title = ctk.CTkLabel(left_frame, text="📹 Live Camera Feed", 
                                   font=("Arial", 16, "bold"))
        video_title.pack(pady=(0, 10))
        
        # Video display canvas
        self.canvas = ctk.CTkCanvas(left_frame, width=640, height=480, bg="#1a1a1a")
        self.canvas.pack(fill="both", expand=True)
        
        # ===== RIGHT SIDE: CONTROLS & MEASUREMENTS =====
        right_frame = ctk.CTkFrame(main_container, width=350)
        right_frame.pack(side="right", fill="both", padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # --- Control Section ---
        control_title = ctk.CTkLabel(right_frame, text="⚙️ Controls", 
                                    font=("Arial", 14, "bold"))
        control_title.pack(pady=(0, 10))
        
        button_frame = ctk.CTkFrame(right_frame)
        button_frame.pack(fill="x", padx=0, pady=10)
        
        self.start_btn = ctk.CTkButton(button_frame, text="▶ Start", 
                                       command=self.start_measurement,
                                       fg_color="#00AA00", font=("Arial", 12, "bold"))
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        self.stop_btn = ctk.CTkButton(button_frame, text="⏹ Stop", 
                                      command=self.stop_measurement,
                                      fg_color="#AA0000", font=("Arial", 12, "bold"))
        self.stop_btn.pack(side="left", fill="x", expand=True, padx=(5, 0))
        
        # --- Status Display ---
        status_title = ctk.CTkLabel(right_frame, text="📊 Status", 
                                   font=("Arial", 12, "bold"))
        status_title.pack(pady=(15, 5))
        
        self.status_label = ctk.CTkLabel(right_frame, text="Ready", 
                                        text_color="#FFD700", font=("Arial", 11))
        self.status_label.pack(anchor="w", padx=5)
        
        self.fps_label = ctk.CTkLabel(right_frame, text="FPS: 0.0", 
                                     text_color="#00FF00", font=("Arial", 10))
        self.fps_label.pack(anchor="w", padx=5, pady=(2, 10))
        
        # --- Measurements Section ---
        measurements_title = ctk.CTkLabel(right_frame, text="📐 Measurements (mm)", 
                                         font=("Arial", 12, "bold"))
        measurements_title.pack(pady=(10, 5))
        
        # Scrollable measurements frame
        self.measurements_frame = ctk.CTkScrollableFrame(right_frame, fg_color="#2a2a2a")
        self.measurements_frame.pack(fill="both", expand=True, padx=0, pady=5)
        
        # Create measurement labels
        self.measurement_labels = {}
        measurement_keys = [
            "Pupillary Distance",
            "Face Width",
            "Bridge Width",
            "Left Lens Width",
            "Right Lens Width"
        ]
        
        for key in measurement_keys:
            frame = ctk.CTkFrame(self.measurements_frame, fg_color="transparent")
            frame.pack(fill="x", padx=5, pady=3)
            
            label = ctk.CTkLabel(frame, text=f"{key}:", text_color="#FFFFFF", 
                                font=("Arial", 10), anchor="w", width=150)
            label.pack(side="left", fill="x")
            
            value = ctk.CTkLabel(frame, text="-- mm", text_color="#00FF00", 
                                font=("Arial", 10, "bold"), anchor="e")
            value.pack(side="left", fill="x", expand=True)
            
            self.measurement_labels[key] = value
        
        # --- Export Section ---
        export_title = ctk.CTkLabel(right_frame, text="💾 Export", 
                                   font=("Arial", 12, "bold"))
        export_title.pack(pady=(10, 5))
        
        export_btn = ctk.CTkButton(right_frame, text="Save Measurements", 
                                  command=self.save_measurements,
                                  fg_color="#0066CC", font=("Arial", 11, "bold"))
        export_btn.pack(fill="x", padx=0, pady=5)
        
    def start_measurement(self):
        """Start the measurement system"""
        if self.is_running:
            return
        
        self.cap = open_camera()
        if self.cap is None:
            self.status_label.configure(text="❌ Camera Error", text_color="#FF0000")
            return
        
        self.is_running = True
        self.status_label.configure(text="✓ Running", text_color="#00FF00")
        
        # Start camera thread
        self.capture_thread = threading.Thread(target=camera_capture_thread, 
                                              args=(self.cap,), daemon=True)
        self.capture_thread.start()
        
        # Start video display thread
        self.display_thread = threading.Thread(target=self.update_display, daemon=True)
        self.display_thread.start()
    
    def stop_measurement(self):
        """Stop the measurement system"""
        global stop_capture
        
        self.is_running = False
        stop_capture = True
        
        if self.cap:
            self.cap.release()
        
        self.status_label.configure(text="Stopped", text_color="#FFD700")
        self.canvas.delete("all")
        self.canvas.create_text(320, 240, text="Camera Stopped", 
                               fill="#666666", font=("Arial", 16))
    
    def update_display(self):
        """Update video display in main thread"""
        global stop_capture
        
        detection_frame_count = 0
        last_measurements = None
        
        while self.is_running and not stop_capture:
            try:
                frame = frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            # Process frame
            frame = cv2.flip(frame, 1)
            
            # Run face detection every 3rd frame
            detection_frame_count += 1
            if detection_frame_count % 3 == 1:
                measurements = measure_frame(frame)
                if measurements:
                    last_measurements = measurements
            else:
                measurements = last_measurements
            
            # Draw overlays
            if measurements:
                points = measurements["Points"]
                
                # Draw lines
                cv2.line(frame, points["left_pupil"], points["right_pupil"], (0, 255, 0), 3)
                cv2.line(frame, points["face_left"], points["face_right"], (255, 0, 0), 2)
                cv2.line(frame, points["nose_left"], points["nose_right"], (0, 0, 255), 2)
                
                # Draw circles
                for point in [points["left_pupil"], points["right_pupil"]]:
                    cv2.circle(frame, point, 4, (0, 255, 0), -1)
                
                # Store measurements
                self.last_measurements = {k: v for k, v in measurements.items() if k != "Points"}
                self.measurement_history.append(self.last_measurements)
            
            # Draw alignment guides
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            cv2.circle(frame, (center_x, center_y), 3, (0, 255, 255), -1)
            cv2.line(frame, (center_x - 30, center_y), (center_x + 30, center_y), (0, 255, 255), 1)
            cv2.line(frame, (center_x, center_y - 30), (center_x, center_y + 30), (0, 255, 255), 1)
            
            # Calculate FPS
            self.fps_counter += 1
            current_time = time.time()
            if current_time - self.fps_clock > 1:
                self.fps = self.fps_counter / (current_time - self.fps_clock)
                self.fps_counter = 0
                self.fps_clock = current_time
            
            # Convert frame to PhotoImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image.thumbnail((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            # Update canvas
            self.canvas.create_image(0, 0, image=photo, anchor="nw")
            self.canvas.image = photo
            
            # Update measurements display
            if self.last_measurements:
                mapping = {
                    "Pupillary Distance (mm)": "Pupillary Distance",
                    "Face Width (mm)": "Face Width",
                    "Bridge Width (mm)": "Bridge Width",
                    "Left Lens Width (mm)": "Left Lens Width",
                    "Right Lens Width (mm)": "Right Lens Width"
                }
                
                for key, display_key in mapping.items():
                    if key in self.last_measurements:
                        self.measurement_labels[display_key].configure(
                            text=f"{self.last_measurements[key]} mm"
                        )
            
            # Update FPS display
            self.fps_label.configure(text=f"FPS: {self.fps:.1f}")
            
            self.update()
            time.sleep(0.03)  # ~30 FPS
    
    def save_measurements(self):
        """Save measurements to file"""
        if not self.last_measurements:
            self.status_label.configure(text="⚠ No measurements to save", text_color="#FFAA00")
            return
        
        # Create output file
        output_file = "face_measurements.txt"
        with open(output_file, "w") as f:
            f.write("=" * 50 + "\n")
            f.write("FACE MEASUREMENTS (MM)\n")
            f.write("=" * 50 + "\n")
            for key, value in self.last_measurements.items():
                f.write(f"{key}: {value}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        self.status_label.configure(text=f"✓ Saved to {output_file}", text_color="#00FF00")

if __name__ == "__main__":
    app = FaceMeasurementGUI()
    app.mainloop()
