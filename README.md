# BlindAssistAI 🦯

> Real-time object detection and audio feedback system designed to assist visually impaired individuals using YOLOv8n and Text-to-Speech (TTS).

---

## 🚀 Overview
BlindAssistAI is a lightweight assistive technology application that detects everyday objects (people, vehicles, stairs, obstacles) via a standard webcam and announces them with audio prompts. Engineered for low-latency inference on consumer hardware.

## ✨ Features
- **Real-Time Detection:** Powered by Ultralytics YOLOv8n for fast inference.
- **Audio Feedback:** Real-time Text-to-Speech (TTS) narration via `pyttsx3`.
- **Graphical Interface:** Easy-to-use desktop GUI built with Tkinter.
- **Hardware Efficient:** Optimized to run locally without cloud API dependencies.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MasterRohitPatil/BlindAssistAi.git
   cd BlindAssistAi
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. Click **Start Camera** in the GUI to begin real-time detection and audio narration.

---

## 📦 Tech Stack
- **Language:** Python 3.9+
- **Model:** Ultralytics YOLOv8
- **Computer Vision:** OpenCV
- **Audio / TTS:** pyttsx3
- **GUI:** Tkinter
