# 🚗 Driver Drowsiness System & Vehicle Counter

A high-performance, real-time computer vision project designed to enhance road safety through **Driver Drowsiness Detection** and **Vehicle Traffic Analysis**.

## 🌟 Key Features

*   **Real-time Drowsiness Detection**: Monitors Eye Aspect Ratio (EAR) and triggers an alarm if fatigue is detected.
*   **Aural Alerts**: Immediate beep sound notification when eyes stay closed for too long.
*   **Vehicle Counter**: Robust traffic analysis counting vehicles crossing a detection line.
*   **User-Friendly Interface**: Real-time feedback with intuitive UI overlays and easy exit options.

---

## 🛠️ Installation

### 1. Prerequisites
*   Python 3.8 or higher
*   A working webcam (for drowsiness detection)

### 2. Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

> [!TIP]
> **Windows Users**: If you face issues installing `dlib`, we have pre-configured the project to use `dlib-bin`, which installs instantly without requiring complex C++ build tools.

---

## 🚀 How to Run & Control

### 1. Driver Drowsiness System
Monitor alertness in real-time using your webcam.
```bash
python dds.py
```
*   **How it works**: If your eyes stay closed for ~1 second, a "DROWSY ALERT!" appears and a beep sounds.

### 2. Vehicle Counter
Analyze traffic from the included video file.
```bash
python vehicle.py
```
*   **How it works**: Increments the counter every time a vehicle crosses the horizontal detection line.

---

## 🚪 How to Close & Shutdown

Each application window can be closed using any of the following methods:

1.  **Keyboard Shortcuts**:
    *   Press **`q`** to quit immediately.
    *   Press **`ESC`** to exit.
    *   Press **`Enter`** (specifically in the Vehicle Counter).
2.  **Mouse Control**:
    *   Click the **Red "X" Button** on the top-right of the window.
3.  **Terminal Shutdown**:
    *   If you need to force-stop the script from the command line, press **`Ctrl + C`** in your terminal.

---

## 🧠 Technical Details (EAR Logic)
The drowsiness system calculates the **Eye Aspect Ratio (EAR)** using 6 facial landmarks per eye.
*   **Threshold**: `0.25`
*   **Consecutive Frames**: `20` (triggers alarm if eyes are closed for 20 frames).

---

## 📁 Project Structure
*   `dds.py`: Main Drowsiness Detection script.
*   `vehicle.py`: Traffic Analysis & Vehicle Counting script.
*   `face_detection_demo.py`: Simple Haar Cascade face detection test.
*   `shape_predictor_68_face_landmarks.dat`: Pre-trained dlib landmark model.
*   `requirements.txt`: Python package dependencies.

---

## 📄 License
This project is licensed under the MIT License.
