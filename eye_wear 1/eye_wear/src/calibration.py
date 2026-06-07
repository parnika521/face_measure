"""
Face Measurement Calibration & Validation System
Helps achieve <1mm accuracy for production-ready measurements
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import json
from datetime import datetime
import statistics

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

# Landmark IDs
LEFT_PUPIL = 468
RIGHT_PUPIL = 473
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
FACE_WIDTH = [234, 454]
NOSE_BRIDGE = [94, 331]

REFERENCE_PD_MM = 63.0  # Default - will be calibrated


class CameraCalibrator:
    """Calibrate camera intrinsic parameters"""
    
    def __init__(self, camera_id=0, checkerboard_size=(9, 6)):
        """
        Args:
            camera_id: Camera index
            checkerboard_size: Checkerboard pattern size (width, height) in corners
        """
        self.camera_id = camera_id
        self.checkerboard_size = checkerboard_size
        self.objpoints = []  # 3D points in real-world space
        self.imgpoints = []  # 2D points in image plane
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_error = None
        
    def calibrate_from_checkerboard(self, num_images=20):
        """
        Calibrate camera using checkerboard pattern
        
        Steps:
        1. Print checkerboard.png (9x6)
        2. Display it or place it flat
        3. Capture images from different angles
        """
        print(f"\n📐 CAMERA CALIBRATION MODE")
        print(f"Prepare a printed 9x6 checkerboard pattern")
        print(f"Capture {num_images} images from different angles")
        print(f"Press SPACE to capture, ESC to finish calibration\n")
        
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Prepare object points (0,0,0), (1,0,0), etc.
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= 0.025  # Assuming 2.5cm squares
        
        frame_count = 0
        while frame_count < num_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            if ret:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
                
                # Draw corners
                cv2.drawChessboardCorners(frame, self.checkerboard_size, corners, ret)
                cv2.putText(frame, f"Captured: {frame_count + 1}/{num_images}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Checkerboard not detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Camera Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and ret:
                frame_count += 1
                print(f"✓ Captured image {frame_count}/{num_images}")
            elif key == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(self.objpoints) < 3:
            print("❌ Not enough calibration images. Need at least 3.")
            return False
        
        # Calibrate camera
        ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, gray.shape[::-1], None, None
        )
        
        if ret:
            # Calculate calibration error
            total_error = 0
            total_points = 0
            for i in range(len(self.objpoints)):
                projected_points, _ = cv2.projectPoints(
                    self.objpoints[i], rvecs[i], tvecs[i], 
                    self.camera_matrix, self.dist_coeffs
                )
                error = cv2.norm(self.imgpoints[i], projected_points, cv2.NORM_L2) / len(projected_points)
                total_error += error
                total_points += 1
            
            self.calibration_error = total_error / total_points
            print(f"✓ Camera calibration complete!")
            print(f"  Calibration error: {self.calibration_error:.4f} pixels")
            print(f"  Camera matrix:\n{self.camera_matrix}")
            return True
        else:
            print("❌ Calibration failed")
            return False
    
    def save_calibration(self, filepath="camera_calibration.json"):
        """Save calibration data"""
        if self.camera_matrix is None:
            print("❌ No calibration data to save")
            return False
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "calibration_error": float(self.calibration_error),
            "checkerboard_size": self.checkerboard_size
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Calibration saved to {filepath}")
        return True
    
    def load_calibration(self, filepath="camera_calibration.json"):
        """Load calibration data"""
        if not os.path.exists(filepath):
            print(f"❌ Calibration file not found: {filepath}")
            return False
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.camera_matrix = np.array(data["camera_matrix"])
        self.dist_coeffs = np.array(data["dist_coeffs"])
        self.calibration_error = data["calibration_error"]
        
        print(f"✓ Calibration loaded from {filepath}")
        print(f"  Calibration error: {self.calibration_error:.4f} pixels")
        return True


class ValidationSystem:
    """Validate measurement accuracy against ground truth"""
    
    def __init__(self):
        self.validation_data = []
        self.measurement_errors = {}
        
    def add_measurement(self, person_id, measurement_name, predicted_mm, actual_mm, image_file=None):
        """
        Add a measurement for validation
        
        Args:
            person_id: Subject identifier
            measurement_name: Name of measurement (e.g., "Pupillary Distance")
            predicted_mm: Your system's prediction (mm)
            actual_mm: Ground truth from calipers (mm)
            image_file: Path to image used for prediction
        """
        error_mm = abs(predicted_mm - actual_mm)
        error_percent = (error_mm / actual_mm) * 100 if actual_mm != 0 else 0
        
        entry = {
            "person_id": person_id,
            "measurement": measurement_name,
            "predicted_mm": round(predicted_mm, 2),
            "actual_mm": round(actual_mm, 2),
            "error_mm": round(error_mm, 2),
            "error_percent": round(error_percent, 2),
            "timestamp": datetime.now().isoformat(),
            "image": image_file
        }
        
        self.validation_data.append(entry)
        
        # Track errors by measurement type
        if measurement_name not in self.measurement_errors:
            self.measurement_errors[measurement_name] = []
        self.measurement_errors[measurement_name].append(error_mm)
        
        return error_mm, error_percent
    
    def generate_report(self):
        """Generate validation report"""
        if not self.validation_data:
            print("❌ No validation data")
            return None
        
        report = {
            "total_measurements": len(self.validation_data),
            "timestamp": datetime.now().isoformat(),
            "measurements": {}
        }
        
        print("\n" + "="*70)
        print("📊 VALIDATION REPORT")
        print("="*70)
        
        overall_errors = []
        
        for measurement_name, errors in self.measurement_errors.items():
            avg_error = statistics.mean(errors)
            median_error = statistics.median(errors)
            max_error = max(errors)
            min_error = min(errors)
            std_dev = statistics.stdev(errors) if len(errors) > 1 else 0
            
            report["measurements"][measurement_name] = {
                "samples": len(errors),
                "avg_error_mm": round(avg_error, 2),
                "median_error_mm": round(median_error, 2),
                "max_error_mm": round(max_error, 2),
                "min_error_mm": round(min_error, 2),
                "std_dev_mm": round(std_dev, 2)
            }
            
            overall_errors.extend(errors)
            
            status = "✓ PASS" if avg_error <= 1.0 else "⚠ NEEDS TUNING"
            print(f"\n{measurement_name}: {status}")
            print(f"  Samples: {len(errors)}")
            print(f"  Average error: {avg_error:.2f}mm")
            print(f"  Median error: {median_error:.2f}mm")
            print(f"  Max error: {max_error:.2f}mm")
            print(f"  Std Dev: {std_dev:.2f}mm")
        
        # Overall statistics
        overall_avg = statistics.mean(overall_errors)
        overall_max = max(overall_errors)
        
        report["overall"] = {
            "avg_error_mm": round(overall_avg, 2),
            "max_error_mm": round(overall_max, 2),
            "production_ready": overall_avg <= 1.0
        }
        
        print("\n" + "-"*70)
        print(f"OVERALL AVERAGE ERROR: {overall_avg:.2f}mm")
        print(f"OVERALL MAX ERROR: {overall_max:.2f}mm")
        
        if overall_avg <= 1.0:
            print("✓ PRODUCTION READY!")
        else:
            print(f"⚠ Needs tuning (target: <1.0mm, current: {overall_avg:.2f}mm)")
        print("="*70 + "\n")
        
        return report
    
    def save_validation_data(self, filepath="validation_results.json"):
        """Save validation data"""
        with open(filepath, 'w') as f:
            json.dump(self.validation_data, f, indent=2)
        print(f"✓ Validation data saved to {filepath}")
    
    def export_csv(self, filepath="validation_results.csv"):
        """Export validation data as CSV for analysis"""
        if not self.validation_data:
            print("❌ No validation data")
            return
        
        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.validation_data[0].keys())
            writer.writeheader()
            writer.writerows(self.validation_data)
        print(f"✓ Validation data exported to {filepath}")


class PersonSpecificCalibration:
    """Calibrate measurements for individual users"""
    
    def __init__(self):
        self.user_profiles = {}
    
    def calibrate_user(self, user_id, known_pd_mm):
        """
        Calibrate for a specific user with their known PD
        
        Args:
            user_id: User identifier
            known_pd_mm: User's actual pupillary distance from optometrist
        """
        print(f"\n👤 PERSON-SPECIFIC CALIBRATION")
        print(f"User: {user_id}")
        print(f"Known PD: {known_pd_mm}mm")
        print(f"Position face in camera and press SPACE to capture")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        measurements = []
        captured = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.putText(frame, "Press SPACE to capture (Q to quit)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Person Calibration", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                # Measure PD from frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, 
                                                         data=frame_rgb))
                
                if results.face_landmarks:
                    landmarks = results.face_landmarks[0]
                    left_pupil = (int(landmarks[LEFT_PUPIL].x * frame.shape[1]),
                                int(landmarks[LEFT_PUPIL].y * frame.shape[0]))
                    right_pupil = (int(landmarks[RIGHT_PUPIL].x * frame.shape[1]),
                                 int(landmarks[RIGHT_PUPIL].y * frame.shape[0]))
                    
                    pd_pixels = np.hypot(right_pupil[0] - left_pupil[0],
                                        right_pupil[1] - left_pupil[1])
                    
                    if pd_pixels > 0:
                        mm_per_px = known_pd_mm / pd_pixels
                        measurements.append(mm_per_px)
                        print(f"  Capture {len(measurements)}: {mm_per_px:.6f} mm/px")
                        
                        if len(measurements) >= 5:
                            captured = True
                            break
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(measurements) >= 3:
            avg_mm_per_px = statistics.mean(measurements)
            std_dev = statistics.stdev(measurements) if len(measurements) > 1 else 0
            
            self.user_profiles[user_id] = {
                "known_pd_mm": known_pd_mm,
                "avg_mm_per_px": avg_mm_per_px,
                "std_dev": std_dev,
                "num_samples": len(measurements),
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"\n✓ Calibration complete for {user_id}")
            print(f"  Scale factor: {avg_mm_per_px:.6f} mm/px (±{std_dev:.6f})")
            print(f"  Use this mm_per_px for {user_id}'s measurements")
            
            return avg_mm_per_px
        else:
            print("❌ Not enough measurements")
            return None
    
    def save_profiles(self, filepath="user_calibrations.json"):
        """Save user calibration profiles"""
        with open(filepath, 'w') as f:
            json.dump(self.user_profiles, f, indent=2)
        print(f"✓ User profiles saved to {filepath}")
    
    def load_profiles(self, filepath="user_calibrations.json"):
        """Load user calibration profiles"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.user_profiles = json.load(f)
            print(f"✓ Loaded {len(self.user_profiles)} user profiles")
            return True
        return False


# ==============================================================================
# USAGE EXAMPLE - Production Workflow
# ==============================================================================

if __name__ == "__main__":
    print("\n🔧 FACE MEASUREMENT CALIBRATION TOOLKIT\n")
    
    # Step 1: Camera Calibration (one-time)
    # calibrator = CameraCalibrator()
    # calibrator.calibrate_from_checkerboard(num_images=20)
    # calibrator.save_calibration()
    
    # Step 2: Person-specific calibration
    # person_calib = PersonSpecificCalibration()
    # mm_per_px_person = person_calib.calibrate_user("john_doe", known_pd_mm=62.5)
    # person_calib.save_profiles()
    
    # Step 3: Validation
    # validator = ValidationSystem()
    # validator.add_measurement("john_doe", "Pupillary Distance", 62.3, 62.5)
    # validator.add_measurement("john_doe", "Face Width", 142.5, 140.2)
    # validator.generate_report()
    # validator.save_validation_data()
    # validator.export_csv()
    
    print("See comments above for usage examples")
