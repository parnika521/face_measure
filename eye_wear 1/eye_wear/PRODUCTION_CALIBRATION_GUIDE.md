# 🎯 Production Calibration Guide: <1mm Accuracy

## Overview
This guide walks through achieving **<1mm average measurement error** for production deployment.

---

## 📊 **Current Error Analysis (Where are we?)**

Your system uses:
- **Global scaling**: `mm_per_px = 63.0 / detected_pd_pixels`
- **Average error expected**: ±2-3mm without calibration
- **Target error**: <1mm for production

**Typical error breakdown:**
- Camera not calibrated: ±1-2mm error
- Using global PD (63mm): ±2-4mm error if user's actual PD ≠ 63mm
- Head pose variation: ±1-2mm error
- MediaPipe landmark noise: ±0.5-1mm error

---

## 🔧 **Step 1: One-Time Camera Calibration (30 min)**

Camera calibration removes lens distortion and establishes accurate focal length.

### What you need:
1. **Print a checkerboard pattern** (9×6 corners)
   - Download: `https://docs.opencv.org/master/pattern.png`
   - Print on A4 paper at actual size
   
2. **Run calibration script:**

```python
from calibration import CameraCalibrator

calibrator = CameraCalibrator(camera_id=0)
success = calibrator.calibrate_from_checkerboard(num_images=20)

if success:
    calibrator.save_calibration("camera_calibration.json")
```

### Steps:
1. Show checkerboard to camera from different angles (20 images)
2. Include: front, sides, tilted, close, far positions
3. Press SPACE when pattern is clearly visible
4. System calculates focal length, principal point, distortion

### Output:
- **camera_calibration.json** - Save this! Contains camera matrix

### Quality Check:
- Calibration error should be **<0.5 pixels** ✓
- If >1 pixel, retake with better lighting

---

## 👤 **Step 2: Person-Specific Calibration (5 min per user)**

Instead of assuming everyone has 63mm PD, measure their actual PD.

### Get actual PD:
1. **Visit an optometrist** - they measure PD for glasses (costs $0-10)
2. **Record**: "My PD is XX.X mm"
3. **OR use online**: PD can be measured with a ruler (less accurate)

### Calibrate in app:

```python
from calibration import PersonSpecificCalibration

calib = PersonSpecificCalibration()
calib.load_profiles()  # Load existing profiles

# For each user
mm_per_px = calib.calibrate_user("user_id", known_pd_mm=62.5)
calib.save_profiles("user_calibrations.json")
```

### Steps:
1. Enter user's known PD from optometrist
2. Position face in camera (good lighting)
3. Press SPACE 5 times from slightly different angles
4. System averages the scale factor for that user

### Output:
- Personalized `mm_per_px` scale factor
- ±0.2mm accuracy improvement

---

## 📐 **Step 3: Create Validation Dataset (1-2 hours)**

Validate your system against ground truth measurements.

### Collect reference data:

```python
from calibration import ValidationSystem
import cv2

validator = ValidationSystem()

# For each test subject:
# 1. Take photo with your system
# 2. Measure with digital calipers (±0.1mm)
# 3. Record both values

validator.add_measurement(
    person_id="person_1",
    measurement_name="Pupillary Distance",
    predicted_mm=62.3,  # Your system's measurement
    actual_mm=62.5,      # From calipers
    image_file="person_1_photo.jpg"
)

validator.add_measurement(
    person_id="person_1",
    measurement_name="Face Width",
    predicted_mm=142.5,
    actual_mm=140.2
)

# Repeat for multiple people (10-20 recommended)
# For each person: measure 5-10 dimensions

validator.generate_report()
validator.save_validation_data("validation_results.json")
```

### What to measure with calipers:
1. **Pupillary Distance** - between pupil centers
2. **Face Width** - temple to temple
3. **Bridge Width** - between pupils at bridge
4. **Lens Widths** - left and right eye openings
5. **New measurements** - nose bridge, temporal, tear to rear

### Equipment needed:
- Digital calipers (±0.1mm) - $15-30 on Amazon
- 10-20 volunteer subjects
- Good lighting for photos

---

## 📊 **Step 4: Analyze Results**

Your validation report will show:

```
Pupillary Distance: ✓ PASS
  Average error: 0.8mm
  Median error: 0.6mm
  Max error: 2.1mm
  Std Dev: 0.5mm

Face Width: ⚠ NEEDS TUNING
  Average error: 1.5mm
  Median error: 1.2mm
  Max error: 3.2mm
  Std Dev: 0.9mm
```

**Production threshold**: **Average error <1.0mm** ✓

---

## 🔧 **Step 5: Fine-Tuning (if needed)**

If error is >1mm, try these adjustments:

### **A. Landmark Refinement**
If specific measurements have high error:

```python
# Current landmarks in gui_main.py
NOSE_BRIDGE = [94, 331]  # These may not be optimal

# Try nearby landmarks for better accuracy
# Test different landmark pairs and measure error

# Example: Try landmark 6 & 5 instead of 94 & 331
# Rerun validation to compare errors
```

### **B. Head Pose Correction**
Head tilt causes measurement errors. Add pose estimation:

```python
# Get head rotation from MediaPipe
if results.facial_transformation_matrixes:
    rotation_matrix = results.facial_transformation_matrixes[0]
    # Extract pitch/yaw/roll
    # Apply correction factor based on head pose
```

### **C. Distance Normalization**
Different camera distances affect scale:

```python
# Estimate face distance from face bounding box size
# Normalize measurements based on expected face size
# Add distance correction factor
```

### **D. Per-Landmark Confidence Weighting**
MediaPipe provides confidence scores:

```python
# Use only high-confidence landmarks (>0.8)
# Discard measurements if landmarks are uncertain
# Warn user if confidence drops
```

---

## 📋 **Implementation Checklist**

- [ ] **Run camera calibration** → `camera_calibration.json`
- [ ] **Collect 10-20 test subjects** with known measurements
- [ ] **Create validation dataset** (50-100 measurements total)
- [ ] **Generate validation report** - check if <1mm average
- [ ] **If passed**: Deploy to production ✓
- [ ] **If failed**: Apply fine-tuning, revalidate
- [ ] **Document calibration** - save validation_results.json
- [ ] **User profiles** → Store personalized mm_per_px per user

---

## 🚀 **Updated gui_main.py Integration**

Integrate calibration data into your GUI:

```python
import json

class FaceMeasurementGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Load camera calibration
        try:
            with open("camera_calibration.json", 'r') as f:
                calib_data = json.load(f)
                self.camera_matrix = np.array(calib_data["camera_matrix"])
                self.dist_coeffs = np.array(calib_data["dist_coeffs"])
        except:
            self.camera_matrix = None
            self.dist_coeffs = None
        
        # Load user profiles
        try:
            with open("user_calibrations.json", 'r') as f:
                self.user_profiles = json.load(f)
        except:
            self.user_profiles = {}
    
    def start_measurement(self):
        # If user has profile, use their mm_per_px
        if hasattr(self, 'current_user_id') and self.current_user_id in self.user_profiles:
            profile = self.user_profiles[self.current_user_id]
            # Use profile["avg_mm_per_px"] instead of global scaling
        
        # If camera calibrated, undistort frames
        if self.camera_matrix is not None:
            # Apply cv2.undistort() to frames
        
        # Rest of measurement code...
```

---

## 📈 **Expected Improvements**

| Calibration Step | Error Reduction |
|------------------|-----------------|
| **Baseline** (no calibration) | ±2-3mm |
| **+ Camera calibration** | ±1.5-2mm |
| **+ Person-specific PD** | ±0.8-1.2mm |
| **+ Landmark optimization** | ±0.5-0.8mm |
| **+ Head pose correction** | ±0.3-0.5mm |

**Target**: Average error **<1.0mm** ✓

---

## 🎓 **Key Concepts**

### Why 1mm?
- Medical tolerance for eyewear measurements: ±1mm
- Below this, error becomes imperceptible
- 1mm error = 0.025mm error per pixel (with typical setup)

### Why person-specific calibration?
- PD varies: 55-75mm (20mm range!)
- Using 63mm on someone with 58mm PD = ±4mm error
- This alone can halve your error margin

### Why camera calibration?
- Lens distortion causes ~1-2% error
- Uncalibrated cameras show increasing error toward edges
- Checkerboard calibration compensates for this

---

## 📞 **Troubleshooting**

**Q: Error still >2mm after calibration**
- A: Recheck calipers accuracy, improve lighting, more test subjects

**Q: Error high only for some measurements**
- A: Those landmarks may be less reliable, try nearby landmark pairs

**Q: Validation data scattered, hard to interpret**
- A: Plot error vs head pose, distance, lighting - find correlation

**Q: One subject's error is huge**
- A: Validate calipers measurement is correct; may be user-specific issue

---

## 📚 **Files Created**

- `calibration.py` - Calibration toolkit (camera, person, validation)
- `camera_calibration.json` - Camera parameters (after Step 1)
- `user_calibrations.json` - Per-user profiles (after Step 2)
- `validation_results.json` - Validation measurements (after Step 3)
- `validation_results.csv` - CSV export for analysis

---

## ✅ **Production Ready Checklist**

Before deploying:

- [ ] Average error <1.0mm on validation set
- [ ] Error <2mm for 95% of measurements (P95)
- [ ] Tested on 10+ diverse subjects
- [ ] All measurement types validated
- [ ] Camera calibration saved and documented
- [ ] User calibration workflow documented
- [ ] Error monitoring in production (log measurements)

---

**You're now ready for production! 🚀**
