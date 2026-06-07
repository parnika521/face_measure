import customtkinter as ctk
import cv2
import threading
import queue
import time
from PIL import Image, ImageTk
import os
import sys

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

# New landmarks for additional measurements
NOSE_BRIDGE_TOP = 6
NOSE_GROOVE = 4  # Nose tip/groove area
LEFT_TEMPLE = 246
RIGHT_TEMPLE = 466
LEFT_EAR = 234
RIGHT_EAR = 454
LEFT_TEAR_DUCT = 33
RIGHT_TEAR_DUCT = 362
LEFT_JAW = 427
RIGHT_JAW = 205

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
    
    while not stop_capture and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
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

    # Calculate new measurements
    # 1. Nose Bridge & L V Groove (inside nose measurement)
    nose_bridge_top = get_point(landmarks, NOSE_BRIDGE_TOP)
    nose_groove = get_point(landmarks, NOSE_GROOVE)
    nose_groove_dist = distance(nose_bridge_top, nose_groove)
    
    # 2. Temporal Height (temple to ear)
    left_temple = get_point(landmarks, LEFT_TEMPLE)
    left_ear = get_point(landmarks, LEFT_EAR)
    left_temporal_height = distance(left_temple, left_ear)
    
    right_temple = get_point(landmarks, RIGHT_TEMPLE)
    right_ear = get_point(landmarks, RIGHT_EAR)
    right_temporal_height = distance(right_temple, right_ear)
    
    # 3. Front (Tear) to Rear (tear duct to rear of face/jaw)
    left_tear = get_point(landmarks, LEFT_TEAR_DUCT)
    left_rear = get_point(landmarks, LEFT_JAW)
    left_tear_to_rear = distance(left_tear, left_rear)
    
    right_tear = get_point(landmarks, RIGHT_TEAR_DUCT)
    right_rear = get_point(landmarks, RIGHT_JAW)
    right_tear_to_rear = distance(right_tear, right_rear)

    return {
        "Pupillary Distance (mm)": round(pd_px * mm_per_px, 2),
        "Face Width (mm)": round(distance(face_left, face_right) * mm_per_px, 2),
        "Bridge Width (mm)": round(distance(nose_left, nose_right) * mm_per_px, 2),
        "Left Lens Width (mm)": round(left_eye_w * mm_per_px, 2),
        "Right Lens Width (mm)": round(right_eye_w * mm_per_px, 2),
        "Nose Bridge & Groove (mm)": round(nose_groove_dist * mm_per_px, 2),
        "Left Temporal Height (mm)": round(left_temporal_height * mm_per_px, 2),
        "Right Temporal Height (mm)": round(right_temporal_height * mm_per_px, 2),
        "Left Tear to Rear (mm)": round(left_tear_to_rear * mm_per_px, 2),
        "Right Tear to Rear (mm)": round(right_tear_to_rear * mm_per_px, 2),
        "Points": {
            "left_pupil": left_pupil,
            "right_pupil": right_pupil,
            "face_left": face_left,
            "face_right": face_right,
            "nose_left": nose_left,
            "nose_right": nose_right,
            "nose_bridge_top": nose_bridge_top,
            "nose_groove": nose_groove,
            "left_temple": left_temple,
            "left_ear": left_ear,
            "right_temple": right_temple,
            "right_ear": right_ear,
            "left_tear": left_tear,
            "left_rear": left_rear,
            "right_tear": right_tear,
            "right_rear": right_rear,
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
        self.last_measurements = None
        self.fps = 0.0
        self.fps_counter = 0
        self.fps_clock = time.time()
        self.detection_frame_count = 0
        self.capture_thread = None
        self.photo = None  # Keep reference to prevent garbage collection
        
        # Stability tracking
        self.stability_counter = 0
        self.stability_threshold = 30  # Frames to wait for stability (1 second @ 30 FPS)
        self.measurements_per_second = {}
        self.second_measurements = []
        self.last_second_time = time.time()
        self.is_stable = False
        self.face_in_zone = False
        
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
            "Right Lens Width",
            "Nose Bridge & Groove",
            "Left Temporal Height",
            "Right Temporal Height",
            "Left Tear to Rear",
            "Right Tear to Rear"
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
        
        # --- Stability & Averaging Section ---
        stability_title = ctk.CTkLabel(right_frame, text="⏱️ Stability Status", 
                                      font=("Arial", 12, "bold"))
        stability_title.pack(pady=(10, 5))
        
        self.stability_label = ctk.CTkLabel(right_frame, text="Position face in blue square", 
                                           text_color="#FFD700", font=("Arial", 10))
        self.stability_label.pack(anchor="w", padx=5)
        
        self.stability_bar = ctk.CTkProgressBar(right_frame, height=6)
        self.stability_bar.pack(fill="x", padx=5, pady=5)
        self.stability_bar.set(0)
        
        # --- Averaged Measurements Section ---
        avg_title = ctk.CTkLabel(right_frame, text="📊 Averaged (per second)", 
                                font=("Arial", 12, "bold"))
        avg_title.pack(pady=(10, 5))
        
        # Scrollable averaged measurements frame
        self.avg_measurements_frame = ctk.CTkScrollableFrame(right_frame, fg_color="#1a3a1a")
        self.avg_measurements_frame.pack(fill="both", expand=True, padx=0, pady=5)
        
        # Create averaged measurement labels
        self.avg_measurement_labels = {}
        
        for key in measurement_keys:
            frame = ctk.CTkFrame(self.avg_measurements_frame, fg_color="transparent")
            frame.pack(fill="x", padx=5, pady=3)
            
            label = ctk.CTkLabel(frame, text=f"{key}:", text_color="#CCFFCC", 
                                font=("Arial", 9), anchor="w", width=150)
            label.pack(side="left", fill="x")
            
            value = ctk.CTkLabel(frame, text="-- mm", text_color="#00DD00", 
                                font=("Arial", 9, "bold"), anchor="e")
            value.pack(side="left", fill="x", expand=True)
            
            self.avg_measurement_labels[key] = value
        
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
        global stop_capture
        
        if self.is_running:
            return
        
        self.cap = open_camera()
        if self.cap is None:
            self.status_label.configure(text="❌ Camera Error", text_color="#FF0000")
            return
        
        self.is_running = True
        stop_capture = False
        self.status_label.configure(text="✓ Running", text_color="#00FF00")
        
        # Reset stability variables
        self.stability_counter = 0
        self.is_stable = False
        self.face_in_zone = False
        self.measurements_per_second = {}
        self.second_measurements = []
        self.last_second_time = time.time()
        self.stability_label.configure(text="Position face in blue square", text_color="#FFD700")
        self.stability_bar.set(0)
        
        # Start camera thread
        self.capture_thread = threading.Thread(target=camera_capture_thread, 
                                              args=(self.cap,), daemon=True)
        self.capture_thread.start()
        
        # Schedule display updates
        self.update_display()
    
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
    
    def is_face_in_zone(self, face_points, frame_width, frame_height):
        """Check if face is within the target zone (center rectangle)"""
        if not face_points:
            return False, (0, 0, 0, 0)
        
        # Define target zone (60% of frame centered)
        zone_width = int(frame_width * 0.6)
        zone_height = int(frame_height * 0.6)
        zone_x1 = (frame_width - zone_width) // 2
        zone_y1 = (frame_height - zone_height) // 2
        zone_x2 = zone_x1 + zone_width
        zone_y2 = zone_y1 + zone_height
        
        # Check if face center (average of key points) is in zone
        center_x = sum(p[0] for p in face_points) / len(face_points)
        center_y = sum(p[1] for p in face_points) / len(face_points)
        
        in_zone = zone_x1 <= center_x <= zone_x2 and zone_y1 <= center_y <= zone_y2
        
        return in_zone, (zone_x1, zone_y1, zone_x2, zone_y2)
    
    def update_display(self):
        """Update video display using after() for thread safety"""
        if not self.is_running:
            return
        
        try:
            frame = frame_queue.get_nowait()
            
            # Process frame
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Draw target zone rectangle
            zone_width = int(w * 0.6)
            zone_height = int(h * 0.6)
            zone_x1 = (w - zone_width) // 2
            zone_y1 = (h - zone_height) // 2
            zone_x2 = zone_x1 + zone_width
            zone_y2 = zone_y1 + zone_height
            
            # Draw zone rectangle (blue = not stable, green = stable)
            zone_color = (0, 255, 0) if self.is_stable else (255, 0, 0)
            cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), zone_color, 2)
            
            # Run face detection every 3rd frame
            self.detection_frame_count += 1
            if self.detection_frame_count % 3 == 1:
                measurements = measure_frame(frame)
                if measurements:
                    self.last_measurements = measurements
            else:
                measurements = self.last_measurements
            
            # Draw overlays and check stability
            if measurements:
                points = measurements["Points"]
                
                # Draw measurement lines
                # Original measurements
                cv2.line(frame, points["left_pupil"], points["right_pupil"], (0, 255, 0), 3)  # Green: Pupil distance
                cv2.line(frame, points["face_left"], points["face_right"], (255, 0, 0), 2)    # Blue: Face width
                cv2.line(frame, points["nose_left"], points["nose_right"], (0, 0, 255), 2)    # Red: Bridge width
                
                # New measurements
                cv2.line(frame, points["nose_bridge_top"], points["nose_groove"], (255, 255, 0), 2)  # Cyan: Nose bridge & groove
                cv2.line(frame, points["left_temple"], points["left_ear"], (255, 0, 255), 2)         # Magenta: Left temporal
                cv2.line(frame, points["right_temple"], points["right_ear"], (255, 0, 255), 2)       # Magenta: Right temporal
                cv2.line(frame, points["left_tear"], points["left_rear"], (0, 255, 255), 2)         # Yellow: Left tear to rear
                cv2.line(frame, points["right_tear"], points["right_rear"], (0, 255, 255), 2)       # Yellow: Right tear to rear
                
                # Draw circles at all measurement endpoints
                for point in [points["left_pupil"], points["right_pupil"]]:
                    cv2.circle(frame, point, 5, (0, 255, 0), -1)  # Green: Pupils
                for point in [points["face_left"], points["face_right"]]:
                    cv2.circle(frame, point, 4, (255, 0, 0), -1)  # Blue: Face edges
                for point in [points["nose_left"], points["nose_right"]]:
                    cv2.circle(frame, point, 4, (0, 0, 255), -1)  # Red: Nose bridge
                for point in [points["nose_bridge_top"], points["nose_groove"]]:
                    cv2.circle(frame, point, 4, (255, 255, 0), -1)  # Cyan: Nose
                for point in [points["left_temple"], points["left_ear"], points["right_temple"], points["right_ear"]]:
                    cv2.circle(frame, point, 4, (255, 0, 255), -1)  # Magenta: Temple/ear
                for point in [points["left_tear"], points["left_rear"], points["right_tear"], points["right_rear"]]:
                    cv2.circle(frame, point, 4, (0, 255, 255), -1)  # Yellow: Tear/rear
                
                # Store measurements
                self.last_measurements = {k: v for k, v in measurements.items() if k != "Points"}
                
                # Check if face is in zone
                face_points = [points["left_pupil"], points["right_pupil"], 
                              points["face_left"], points["face_right"]]
                self.face_in_zone, _ = self.is_face_in_zone(face_points, w, h)
                
                # Stability tracking
                if self.face_in_zone:
                    self.stability_counter += 1
                    
                    # Add to second measurements
                    self.second_measurements.append(self.last_measurements.copy())
                    
                    # Check if we've reached stability threshold (1 second)
                    if self.stability_counter >= self.stability_threshold and not self.is_stable:
                        self.is_stable = True
                        self.stability_label.configure(text="✓ Stable! Averaging...", 
                                                      text_color="#00FF00")
                    
                    # Every second, calculate averages
                    current_time = time.time()
                    if current_time - self.last_second_time >= 1.0 and self.is_stable:
                        if self.second_measurements:
                            self.measurements_per_second = self._calculate_averages(
                                self.second_measurements
                            )
                        self.second_measurements = []
                        self.last_second_time = current_time
                else:
                    self.stability_counter = 0
                    self.is_stable = False
                    self.stability_label.configure(text="⚠ Move face into blue square", 
                                                  text_color="#FFAA00")
                    self.stability_bar.set(0)
            else:
                self.stability_counter = 0
                self.is_stable = False
                self.face_in_zone = False
                self.stability_label.configure(text="⚠ No face detected", 
                                              text_color="#FF6666")
                self.stability_bar.set(0)
            
            # Update stability bar
            if self.face_in_zone and not self.is_stable:
                progress = min(self.stability_counter / self.stability_threshold, 1.0)
                self.stability_bar.set(progress)
            elif self.is_stable:
                self.stability_bar.set(1.0)
            
            # Draw alignment guides (center crosshair)
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
            self.photo = ImageTk.PhotoImage(image)
            
            # Update canvas
            self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
            
            # Update measurements display
            if self.last_measurements:
                mapping = {
                    "Pupillary Distance (mm)": "Pupillary Distance",
                    "Face Width (mm)": "Face Width",
                    "Bridge Width (mm)": "Bridge Width",
                    "Left Lens Width (mm)": "Left Lens Width",
                    "Right Lens Width (mm)": "Right Lens Width",
                    "Nose Bridge & Groove (mm)": "Nose Bridge & Groove",
                    "Left Temporal Height (mm)": "Left Temporal Height",
                    "Right Temporal Height (mm)": "Right Temporal Height",
                    "Left Tear to Rear (mm)": "Left Tear to Rear",
                    "Right Tear to Rear (mm)": "Right Tear to Rear"
                }
                
                for key, display_key in mapping.items():
                    if key in self.last_measurements:
                        self.measurement_labels[display_key].configure(
                            text=f"{self.last_measurements[key]} mm"
                        )
            
            # Update averaged measurements display
            if self.measurements_per_second:
                mapping = {
                    "Pupillary Distance (mm)": "Pupillary Distance",
                    "Face Width (mm)": "Face Width",
                    "Bridge Width (mm)": "Bridge Width",
                    "Left Lens Width (mm)": "Left Lens Width",
                    "Right Lens Width (mm)": "Right Lens Width",
                    "Nose Bridge & Groove (mm)": "Nose Bridge & Groove",
                    "Left Temporal Height (mm)": "Left Temporal Height",
                    "Right Temporal Height (mm)": "Right Temporal Height",
                    "Left Tear to Rear (mm)": "Left Tear to Rear",
                    "Right Tear to Rear (mm)": "Right Tear to Rear"
                }
                
                for key, display_key in mapping.items():
                    if key in self.measurements_per_second:
                        self.avg_measurement_labels[display_key].configure(
                            text=f"{self.measurements_per_second[key]} mm"
                        )
            
            # Update FPS display
            self.fps_label.configure(text=f"FPS: {self.fps:.1f}")
            
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error in update_display: {e}")
        
        # Schedule next update (30ms = ~30 FPS)
        self.after(30, self.update_display)
    
    def _calculate_averages(self, measurements_list):
        """Calculate average of all measurements in the list"""
        if not measurements_list:
            return {}
        
        averages = {}
        keys = measurements_list[0].keys()
        
        for key in keys:
            values = [m[key] for m in measurements_list if key in m]
            if values:
                averages[key] = round(np.mean(values), 2)
        
        return averages
    
    def save_measurements(self):
        """Save measurements to file"""
        if not self.last_measurements:
            self.status_label.configure(text="⚠ No measurements to save", text_color="#FFAA00")
            return
        
        # Create output file
        output_file = "face_measurements.txt"
        with open(output_file, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("FACE MEASUREMENTS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Current measurements
            f.write("CURRENT MEASUREMENTS (MM)\n")
            f.write("-" * 60 + "\n")
            for key, value in self.last_measurements.items():
                f.write(f"{key}: {value}\n")
            
            # Averaged measurements
            if self.measurements_per_second:
                f.write("\n" + "-" * 60 + "\n")
                f.write("AVERAGED MEASUREMENTS (PER SECOND, MM)\n")
                f.write("-" * 60 + "\n")
                for key, value in self.measurements_per_second.items():
                    f.write(f"{key}: {value}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Stability Status: {'Stable' if self.is_stable else 'Unstable'}\n")
            f.write("=" * 60 + "\n")
        
        self.status_label.configure(text=f"✓ Saved to {output_file}", text_color="#00FF00")


if __name__ == "__main__":
    app = FaceMeasurementGUI()
    app.mainloop()
