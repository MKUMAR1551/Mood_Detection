import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import cv2
import mood_detection as logic

# --- THEME COLORS ---
BG_COLOR = "#2b2b2b"       # Dark Charcoal
FG_COLOR = "#ffffff"       # White text
ACCENT_COLOR = "#00e5ff"   # Cyber Cyan
BTN_NORMAL = "#424242"     # Dark Grey
BTN_HOVER = "#616161"      # Lighter Grey
SUCCESS_COLOR = "#00e676"  # Bright Green
ERROR_COLOR = "#ff5252"    # Bright Red
NEON_GREEN = "#39ff14"     # <--- NEW NEON GREEN COLOR

class RoundedButton(tk.Canvas):
    def __init__(self, parent, width, height, cornerradius, padding, color, bg, text, command=None):
        tk.Canvas.__init__(self, parent, borderwidth=0, relief="flat", highlightthickness=0, bg=bg)
        self.command = command
        self.color = color
        self.original_color = color
        self.hover_color = "#616161" if color == BTN_NORMAL else color
        
        # Dimensions
        self.width = width
        self.height = height
        self.rad = cornerradius
        
        # Draw Shapes
        self.id = self.round_rect(padding, padding, width-padding, height-padding, cornerradius, fill=color, outline=color)
        self.text_id = self.create_text(width/2, height/2, text=text, fill='white', font=("Segoe UI", 11, "bold"))
        
        # Events
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        
        self.configure(width=width, height=height)

    def round_rect(self, x1, y1, x2, y2, radius=25, **kwargs):
        points = [x1+radius, y1,
                  x1+radius, y1,
                  x2-radius, y1,
                  x2-radius, y1,
                  x2, y1,
                  x2, y1+radius,
                  x2, y1+radius,
                  x2, y2-radius,
                  x2, y2-radius,
                  x2, y2,
                  x2-radius, y2,
                  x2-radius, y2,
                  x1+radius, y2,
                  x1+radius, y2,
                  x1, y2,
                  x1, y2-radius,
                  x1, y2-radius,
                  x1, y1+radius,
                  x1, y1+radius,
                  x1, y1]
        return self.create_polygon(points, **kwargs, smooth=True)

    def _on_press(self, event):
        self.move(self.text_id, 1, 1)

    def _on_release(self, event):
        self.move(self.text_id, -1, -1)
        if self.command: self.command()

    def _on_enter(self, event):
        if self.color == SUCCESS_COLOR:
            self.itemconfig(self.id, fill="#33ff99", outline="#33ff99")
        elif self.color == ERROR_COLOR:
            self.itemconfig(self.id, fill="#ff8080", outline="#ff8080")
        else:
            self.itemconfig(self.id, fill=BTN_HOVER, outline=BTN_HOVER)

    def _on_leave(self, event):
        self.itemconfig(self.id, fill=self.color, outline=self.color)

    def set_text(self, text):
        self.itemconfig(self.text_id, text=text)

    def set_color(self, color):
        self.color = color
        self.itemconfig(self.id, fill=color, outline=color)

class MoodApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mood AI Dashboard Pro")
        self.root.geometry("900x750")
        self.root.configure(bg=BG_COLOR)

        # --- HEADER ---
        header = tk.Frame(root, bg="#1a1a1a", height=80)
        header.pack(fill="x", side="top")
        tk.Label(header, text="AI MOOD DETECTION SYSTEM", font=("Segoe UI", 20, "bold"), 
                 bg="#1a1a1a", fg=ACCENT_COLOR).pack(pady=20)

        # --- MAIN CONTAINER ---
        main_frame = tk.Frame(root, bg=BG_COLOR)
        main_frame.pack(fill="both", expand=True, padx=30, pady=30)

        # --- LEFT PANEL (CONTROLS) ---
        left_panel = tk.Frame(main_frame, bg=BG_COLOR, width=320)
        left_panel.pack(side="left", fill="y", padx=(0, 30))

        # Status
        status_frame = tk.LabelFrame(left_panel, text=" SYSTEM STATUS ", font=("Segoe UI", 10, "bold"),
                                     bg=BG_COLOR, fg=FG_COLOR, bd=0)
        status_frame.pack(fill="x", pady=(0, 30))
        
        self.status_lbl = tk.Label(status_frame, text="Initializing...", font=("Segoe UI", 12), 
                                   bg=BG_COLOR, fg="orange", anchor="w")
        self.status_lbl.pack(fill="x", pady=5)
        
        self.progress = ttk.Progressbar(status_frame, orient="horizontal", length=200, mode='indeterminate')
        self.progress.pack(fill="x", pady=10)

        # Buttons
        cmd_frame = tk.LabelFrame(left_panel, text=" COMMANDS ", font=("Segoe UI", 10, "bold"),
                                  bg=BG_COLOR, fg=FG_COLOR, bd=0)
        cmd_frame.pack(fill="x", pady=10)

        self.btn_retrain = RoundedButton(cmd_frame, 280, 50, 20, 2, BTN_NORMAL, BG_COLOR, "⚡ Retrain Model", 
                                         self.start_training_thread)
        self.btn_retrain.pack(pady=10)

        self.btn_load = RoundedButton(cmd_frame, 280, 50, 20, 2, BTN_NORMAL, BG_COLOR, "📂 Load Saved Model", 
                                      self.reload_saved_model)
        self.btn_load.pack(pady=10)

        self.btn_graph = RoundedButton(cmd_frame, 280, 50, 20, 2, BTN_NORMAL, BG_COLOR, "📊 Analyze Accuracy", 
                                       self.start_graph_thread)
        self.btn_graph.pack(pady=10)

        # --- RIGHT PANEL (LOGS & CAM) ---
        right_panel = tk.Frame(main_frame, bg=BG_COLOR)
        right_panel.pack(side="right", fill="both", expand=True)

        # Camera Button
        self.is_running = False
        self.btn_cam = RoundedButton(right_panel, 500, 70, 30, 2, SUCCESS_COLOR, BG_COLOR, "START CAMERA", 
                                     self.toggle_webcam)
        self.btn_cam.pack(pady=(0, 20))

        # Log Console (UPDATED COLOR HERE)
        tk.Label(right_panel, text="SYSTEM LOGS:", font=("Segoe UI", 10, "bold"), bg=BG_COLOR, fg="#888").pack(anchor="w")
        self.log_area = scrolledtext.ScrolledText(right_panel, width=50, height=20, font=("Consolas", 10),
                                                  bg="#1e1e1e", fg=NEON_GREEN, insertbackground="white", bd=0) # <--- NEON GREEN
        self.log_area.pack(fill="both", expand=True)

        # --- INIT ---
        self.faces = []
        self.labels = []
        self.model = None
        
        threading.Thread(target=self.initial_setup, daemon=True).start()

    # --- HELPERS ---
    def log(self, message):
        self.log_area.insert(tk.END, f">> {message}\n")
        self.log_area.see(tk.END)

    def initial_setup(self):
        logic.download_haar_if_missing(self.log)
        self.check_model_status()

    def check_model_status(self):
        if os.path.exists(logic.MODEL_FILE):
            self.update_status("READY (Model Found)", SUCCESS_COLOR)
            self.reload_saved_model(silent=True)
        else:
            self.update_status("WAITING (Train Model)", "orange")

    def update_status(self, text, color):
        self.status_lbl.config(text=text, fg=color)

    def start_loading_anim(self):
        self.progress.start(10)

    def stop_loading_anim(self):
        self.progress.stop()

    def reload_saved_model(self, silent=False):
        loaded_model = logic.load_saved_model(self.log if not silent else lambda x: None)
        if loaded_model:
            self.model = loaded_model
            self.update_status("SYSTEM READY", SUCCESS_COLOR)
            if not silent: self.log("Saved model loaded successfully.")
        else:
            self.log("No valid saved model found.")
            self.update_status("NO MODEL", ERROR_COLOR)

    # --- THREADED ACTIONS ---
    def start_training_thread(self):
        threading.Thread(target=self.run_training, daemon=True).start()

    def run_training(self):
        self.start_loading_anim()
        self.update_status("TRAINING IN PROGRESS...", ACCENT_COLOR)
        self.faces, self.labels = logic.load_dataset(self.log)
        
        if not self.faces:
            self.log("Error: No data found.")
            self.stop_loading_anim()
            self.update_status("DATA ERROR", ERROR_COLOR)
            return

        self.model = logic.train_model(self.faces, self.labels, self.log)
        self.stop_loading_anim()
        self.update_status("TRAINING COMPLETE", SUCCESS_COLOR)

    def start_graph_thread(self):
        threading.Thread(target=self.run_graph, daemon=True).start()

    def run_graph(self):
        if not self.faces:
            self.log("Loading dataset for analysis...")
            self.start_loading_anim()
            self.faces, self.labels = logic.load_dataset(self.log)
            if not self.faces: 
                self.stop_loading_anim()
                return

        self.log("Calculating Accuracy Curve (Wait ~30s)...")
        self.start_loading_anim()
        self.update_status("ANALYZING DATA...", ACCENT_COLOR)
        
        steps, t_acc, v_acc = logic.simulate_learning_curve_data(self.faces, self.labels, self.log)
        
        self.stop_loading_anim()
        self.update_status("SYSTEM READY", SUCCESS_COLOR)
        
        self.root.after(0, lambda: self.show_plot(steps, t_acc, v_acc))

    def show_plot(self, steps, t, v):
        if not steps: return
        self.log(f"Graph Generated. Final Val Acc: {v[-1]:.1f}%")
        
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Performance Analysis")
        graph_window.geometry("600x500")
        graph_window.configure(bg="#222")
        
        fig = plt.figure(figsize=(6, 5))
        plt.plot(steps, t, label='Training', marker='o', color='#00e5ff', linewidth=2)
        plt.plot(steps, v, label='Validation', marker='o', color='#00e676', linewidth=2)
        plt.title("AI Learning Curve")
        plt.style.use('dark_background')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # --- CAMERA LOGIC ---
    def toggle_webcam(self):
        if not self.is_running:
            if self.model is None:
                messagebox.showerror("System Error", "Model not loaded!")
                return
            self.is_running = True
            # TURN BUTTON RED
            self.btn_cam.set_color(ERROR_COLOR)
            self.btn_cam.set_text("STOP CAMERA")
            threading.Thread(target=self.run_webcam_loop, daemon=True).start()
        else:
            self.is_running = False
            self.btn_cam.set_text("STOPPING...")

    def run_webcam_loop(self):
        self.log("Camera engine started.")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened(): cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(logic.HAAR_CASCADE_FILE) 
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret: break
            
            img_copy = frame.copy()
            face_roi, rect = logic.process_image(frame, face_cascade)
            
            if face_roi is not None:
                face_resized = cv2.resize(face_roi, (200, 200))
                label_id, confidence = self.model.predict(face_resized)
                mood = logic.SUBJECTS[label_id] if label_id < len(logic.SUBJECTS) else "Unknown"
                color = (0, 255, 0) if confidence < 90 else (0, 165, 255)
                
                if rect is not None:
                    (x, y, w, h) = rect
                    cv2.rectangle(img_copy, (x, y), (x+w, y+h), color, 2)
                    cv2.rectangle(img_copy, (x, y-35), (x+w, y), color, -1)
                    cv2.putText(img_copy, f"{mood} {int(confidence)}", (x+5, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            
            cv2.imshow("Mood Cam (Dark Mode)", img_copy)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False
                break
            
            if cv2.getWindowProperty("Mood Cam (Dark Mode)", cv2.WND_PROP_VISIBLE) < 1:
                self.is_running = False
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.log("Camera engine stopped.")
        
        self.root.after(0, self.reset_cam_button)

    def reset_cam_button(self):
        self.is_running = False
        # RESET BUTTON TO GREEN
        self.btn_cam.set_color(SUCCESS_COLOR)
        self.btn_cam.set_text("START CAMERA")

if __name__ == "__main__":
    root = tk.Tk()
    app = MoodApp(root)
    root.mainloop()