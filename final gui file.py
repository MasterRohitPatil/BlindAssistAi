import cv2
from ultralytics import YOLO
import time
import os
import tkinter as tk
from tkinter import ttk, messagebox
import threading


class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BlindAssistAI")

        # Model and camera
        self.model = None
        self.cap = None
        self.running = False
        self.last_announce_time = 0
        self.announce_interval = 3  # seconds
        self.priority_objects = ["person", "car", "stairs", "door"]

        # Create GUI
        self.create_widgets()

        # Load model in background
        threading.Thread(target=self.load_model, daemon=True).start()

    def load_model(self):
        """Load YOLO model in background"""
        self.status_var.set("Loading model...")
        try:
            self.model = YOLO('yolov8n.pt')
            self.status_var.set("Model loaded. Click Start Camera")
            self.start_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Model load failed")

    def create_widgets(self):
        """Create the GUI interface"""
        # Main frames
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Announcement display
        self.announcement_frame = ttk.LabelFrame(self.root, text="Announcements", padding="10")
        self.announcement_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.announcement_text = tk.Text(
            self.announcement_frame,
            height=10,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.announcement_text.pack(fill=tk.BOTH, expand=True)

        # Status bar
        status_frame = ttk.Frame(self.root, padding="10")
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Control buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.LEFT)

        self.start_btn = ttk.Button(
            btn_frame,
            text="Start Camera",
            command=self.start_camera,
            state=tk.DISABLED
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(
            btn_frame,
            text="Stop Camera",
            command=self.stop_camera,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Settings button
        ttk.Button(
            control_frame,
            text="Settings ⚙",
            command=self.open_settings
        ).pack(side=tk.RIGHT, padx=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Initializing...")
        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        ).pack(fill=tk.X)

    def open_settings(self):
        """Open settings window"""
        settings = tk.Toplevel(self.root)
        settings.title("Settings")

        ttk.Label(settings, text="Announcement Interval (seconds):").pack(pady=5)
        interval = ttk.Spinbox(settings, from_=1, to=10, value=self.announce_interval)
        interval.pack(pady=5)

        def save_settings():
            try:
                self.announce_interval = int(interval.get())
                settings.destroy()
                self.add_announcement(f"Settings updated: Announcements every {self.announce_interval} seconds")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number")

        ttk.Button(settings, text="Save", command=save_settings).pack(pady=10)

    def add_announcement(self, text):
        """Add announcement to the text box"""
        self.announcement_text.config(state=tk.NORMAL)
        self.announcement_text.insert(tk.END, text + "\n")
        self.announcement_text.see(tk.END)
        self.announcement_text.config(state=tk.DISABLED)

    def start_camera(self):
        """Start the camera feed in separate window"""
        if self.running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Camera running - detecting objects...")
        self.add_announcement("Camera started - object detection active")

        # Start video processing in separate thread
        threading.Thread(target=self.process_video, daemon=True).start()

    def stop_camera(self):
        """Stop the camera feed"""
        self.running = False
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Camera stopped")
        self.add_announcement("Camera stopped")

    def speak(self, text):
        """Windows text-to-speech with logging"""
        print(f"Announcing: {text}")
        self.add_announcement(f"Announcing: {text}")
        clean_text = text.replace('"', '')
        os.system(
            f'powershell -Command "Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak(\'{clean_text}\')"')

    def process_video(self):
        """Process video frames and detect objects"""
        cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.status_var.set("Camera error")
                break

            # Object detection
            results = self.model(frame, verbose=False)
            current_objects = set()

            for result in results:
                for box in result.boxes:
                    if float(box.conf) > 0.5:
                        label = self.model.names[int(box.cls)]
                        current_objects.add(label)

                        # Distance estimation for priority objects
                        if label in self.priority_objects:
                            width = box.xyxy[0][2] - box.xyxy[0][0]
                            distance = int(5000 / width)  # Approximate distance in cm
                            current_objects.add(f"{label} ({distance}cm)")

            # Show detection results in camera window
            annotated_frame = results[0].plot()
            cv2.imshow("Camera Feed", annotated_frame)
            cv2.waitKey(1)

            # Announce at intervals
            if time.time() - self.last_announce_time >= self.announce_interval:
                if current_objects:
                    # Separate important objects
                    important = [obj for obj in current_objects
                                 if any(p in obj for p in self.priority_objects)]
                    normal = [obj for obj in current_objects
                              if not any(p in obj for p in self.priority_objects)]

                    if important:
                        self.speak("Important: " + ", ".join(important))
                    if normal:
                        self.speak("Also seeing: " + ", ".join(normal))
                else:
                    self.speak("No objects detected")
                self.last_announce_time = time.time()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("600x400")
    app = ObjectDetectionApp(root)
    root.mainloop()