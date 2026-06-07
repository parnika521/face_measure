# Face Measurement GUI - User Guide

## Quick Start

### Run the GUI Application

```bash
cd "/home/rohil/Code_Repo/Face_measure/eye_wear 1/eye_wear/src"
/home/rohil/Code_Repo/Face_measure/.venv/bin/python gui_main.py
```

Or from the workspace root:
```bash
python "eye_wear 1/eye_wear/src/gui_main.py"
```

## Features

### 📹 Live Camera Feed
- Real-time video streaming at 30+ FPS
- Alignment guides (center crosshair)
- Face detection visualization with measurement lines

### 📊 Measurements (Updated in Real-Time)
- **Pupillary Distance (PD)** - Distance between pupils
- **Face Width** - Full face width measurement
- **Bridge Width** - Distance between eye bridges
- **Left Lens Width** - Left eye lens area width
- **Right Lens Width** - Right eye lens area width

### ⚙️ Controls
- **▶ Start** - Activate camera and start measurements
- **⏹ Stop** - Stop camera and reset display
- **FPS Counter** - Monitor real-time performance

### 💾 Export
- **Save Measurements** - Export current measurements to `face_measurements.txt`

## How to Use

1. **Start the Application** - Click the "▶ Start" button
2. **Align Your Face** - Position your face in the center crosshair on the camera feed
3. **Keep Still** - Allow the system 1-2 seconds to stabilize measurements
4. **View Measurements** - Measurements update in real-time on the right panel
5. **Save Results** - Click "Save Measurements" to export to a file

## Performance Optimizations

✅ **Resolution**: 640x480 (optimized for fast processing)  
✅ **Detection**: Every 3rd frame (reduces ML inference load)  
✅ **Display**: 30+ FPS (smooth real-time video)  
✅ **Threading**: Non-blocking camera capture  

## Troubleshooting

### Camera not connecting
- Ensure camera permissions: `usermod -aG video $USER`
- Reconnect camera or restart application

### Low FPS
- Ensure good lighting
- Close other applications using camera
- Check GPU resources

### Measurements not showing
- Ensure face is clearly visible in frame
- Increase lighting around face
- Keep face centered in view

## Output Format

When you save measurements, they are stored in `face_measurements.txt`:

```
==================================================
FACE MEASUREMENTS (MM)
==================================================
Pupillary Distance (mm): 63.45
Face Width (mm): 142.30
Bridge Width (mm): 28.50
Left Lens Width (mm): 32.10
Right Lens Width (mm): 31.95
==================================================
Timestamp: 2026-06-02 14:30:45
==================================================
```

## System Requirements

- Python 3.12+
- Webcam/USB Camera
- Linux OS (tested on Ubuntu)
- MediaPipe Tasks
- CustomTkinter
- OpenCV
- NumPy
- Pillow (PIL)

Enjoy your Face Measurement System! 🚀
