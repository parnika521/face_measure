#!/usr/bin/env python3
"""
Quick-Start Calibration Workflow
Run this to begin production calibration process
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from calibration import CameraCalibrator, PersonSpecificCalibration, ValidationSystem
import json


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def step1_camera_calibration():
    """Camera calibration (one-time setup)"""
    print_header("STEP 1: CAMERA CALIBRATION (One-Time)")
    
    print("""
This calibration removes lens distortion and establishes accurate
camera parameters. Only needs to be done once.

REQUIREMENTS:
  1. Print a 9×6 checkerboard pattern
     → Download: https://docs.opencv.org/master/pattern.png
     → Print on A4 paper at actual size
  
  2. You'll capture 20 images from different angles
  
INSTRUCTIONS:
  → Place checkerboard in front of camera
  → Press SPACE to capture (repeat 20 times)
  → Move pattern: front, sides, tilted, close, far
  → Press ESC when done
""")
    
    response = input("Ready to start camera calibration? (y/n): ").strip().lower()
    if response != 'y':
        print("Skipped camera calibration")
        return False
    
    calibrator = CameraCalibrator(camera_id=0)
    success = calibrator.calibrate_from_checkerboard(num_images=20)
    
    if success:
        # Save calibration
        save_path = os.path.join(os.path.dirname(__file__), 'eye_wear/src/camera_calibration.json')
        calibrator.save_calibration(save_path)
        print(f"\n✓ Calibration saved to: {save_path}")
        print(f"  Calibration error: {calibrator.calibration_error:.4f} pixels (should be <0.5)")
        return True
    else:
        print("\n❌ Camera calibration failed")
        return False


def step2_person_calibration():
    """Person-specific calibration"""
    print_header("STEP 2: PERSON-SPECIFIC CALIBRATION")
    
    print("""
This measures each user's pupillary distance (PD) to personalize
measurements. Much more accurate than using default 63mm PD.

TO GET YOUR PD:
  Option 1: Ask your optometrist (best - free with glasses fitting)
  Option 2: Measure online (less accurate)
  Option 3: Print ruler and measure yourself
  
TYPICAL RANGE: 55-75mm
""")
    
    calib = PersonSpecificCalibration()
    
    # Try to load existing profiles
    profiles_path = os.path.join(os.path.dirname(__file__), 'eye_wear/src/user_calibrations.json')
    calib.load_profiles(profiles_path) if os.path.exists(profiles_path) else None
    
    while True:
        user_id = input("\nEnter user ID (or 'skip'/'done'): ").strip()
        
        if user_id.lower() in ['skip', 'done', 'q']:
            break
        
        if not user_id:
            continue
        
        try:
            pd_mm = float(input(f"Enter {user_id}'s PD in mm (e.g., 62.5): "))
            if pd_mm < 40 or pd_mm > 80:
                print("⚠ Warning: PD seems unusual (typical: 55-75mm)")
            
            mm_per_px = calib.calibrate_user(user_id, known_pd_mm=pd_mm)
            
            if mm_per_px:
                print(f"✓ {user_id} calibrated: {mm_per_px:.6f} mm/px")
        
        except ValueError:
            print("❌ Invalid input. Enter a number.")
            continue
    
    if calib.user_profiles:
        calib.save_profiles(profiles_path)
        print(f"\n✓ Saved {len(calib.user_profiles)} user profiles")


def step3_validation():
    """Create validation dataset"""
    print_header("STEP 3: VALIDATION SETUP")
    
    print("""
Now we need to validate your system against ground truth measurements.

WHAT YOU'LL NEED:
  1. Digital calipers (±0.1mm) - $15-30 on Amazon
  2. 10-20 volunteer test subjects
  3. Your GUI app running to capture measurements
  4. Good lighting

WORKFLOW:
  1. For each person:
     a) Take photos with your GUI
     b) Record your system's measurements
     c) Measure same points with calipers (ground truth)
     d) Enter both into validation system
  
  2. Collect data for:
     - Pupillary Distance
     - Face Width
     - Bridge Width
     - Lens Widths (both sides)
     - New measurements (nose bridge, temporal, tear to rear)

This will take 1-2 hours for good validation data.
""")
    
    response = input("Ready to add validation measurements? (y/n): ").strip().lower()
    if response != 'y':
        return
    
    validator = ValidationSystem()
    
    print("\nEnter validation data:")
    print("Format: person_id, measurement_name, predicted_mm, actual_mm")
    print("Example: john, 'Pupillary Distance', 62.3, 62.5")
    print("(Type 'done' to finish)\n")
    
    while True:
        entry = input("Enter measurement (or 'done'): ").strip()
        
        if entry.lower() == 'done':
            break
        
        try:
            parts = [p.strip().strip("'\"") for p in entry.split(',')]
            if len(parts) != 4:
                print("❌ Need 4 values: person_id, measurement_name, predicted_mm, actual_mm")
                continue
            
            person_id, measurement_name, predicted, actual = parts
            predicted_mm = float(predicted)
            actual_mm = float(actual)
            
            error_mm, error_percent = validator.add_measurement(
                person_id, measurement_name, predicted_mm, actual_mm
            )
            
            print(f"  ✓ Recorded: {error_mm:.2f}mm error ({error_percent:.1f}%)")
        
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
    
    if validator.validation_data:
        # Generate report
        report = validator.generate_report()
        
        # Save data
        results_path = os.path.join(os.path.dirname(__file__), 'eye_wear/src/validation_results.json')
        validator.save_validation_data(results_path)
        csv_path = os.path.join(os.path.dirname(__file__), 'eye_wear/src/validation_results.csv')
        validator.export_csv(csv_path)
        
        print(f"✓ Results saved to:")
        print(f"  - {results_path}")
        print(f"  - {csv_path}")


def main():
    """Main workflow"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                 PRODUCTION CALIBRATION WORKFLOW                    ║
║              Achieve <1mm Measurement Accuracy                     ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    print("""
This workflow has 3 steps:

  1️⃣  CAMERA CALIBRATION (30 min, one-time)
     → Removes lens distortion
     → Establishes accurate focal length
     → Need: checkerboard pattern
  
  2️⃣  PERSON-SPECIFIC CALIBRATION (5 min per user)
     → Measures individual PD
     → Improves accuracy by ~50%
     → Need: user's known PD from optometrist
  
  3️⃣  VALIDATION (1-2 hours)
     → Test against ground truth measurements
     → Measure error vs. prediction
     → Determine if production-ready
     → Need: digital calipers, test subjects

TARGET: <1mm average error ✓
""")
    
    # Step 1
    if step1_camera_calibration():
        print("\n✓ STEP 1 COMPLETE\n")
    else:
        print("\n⚠ Camera calibration failed or skipped")
    
    # Step 2
    step2_person_calibration()
    print("\n✓ STEP 2 COMPLETE\n")
    
    # Step 3
    step3_validation()
    print("\n✓ STEP 3 COMPLETE\n")
    
    print_header("NEXT STEPS")
    print("""
1. Review validation results (validation_results.json)

2. If average error <1.0mm:
   ✓ YOU'RE PRODUCTION READY!
   
   Next: Deploy with calibration files:
   - camera_calibration.json
   - user_calibrations.json

3. If average error >1.0mm:
   → Check PRODUCTION_CALIBRATION_GUIDE.md for fine-tuning steps
   → Common issues: camera not calibrated, lighting, PD not accurate
   → Try: more test subjects, better calipers, check landmarks

4. Integrate calibration into GUI:
   → See PRODUCTION_CALIBRATION_GUIDE.md for code examples
   → Load camera_calibration.json on startup
   → Use person-specific mm_per_px from user_calibrations.json
""")
    
    print("\n📖 For detailed info, see: PRODUCTION_CALIBRATION_GUIDE.md\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
