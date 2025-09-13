import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import ocr.tesseract_text_extraction as tesseract_te
import ocr.paddle_ocr_text_extraction as paddle_te
import yolo.plate_recognition as pr
import helpers.img_utils as iu
import numpy as np
import threading
import tkinter.ttk as ttk

WEBCAM_INDEX = 1

class LicensePlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Detector")
        self.root.state("zoomed")

        self.img_path = None
        self.img_preview = None
        self.cv_image = None
        self.plate_position = None

        self.root.configure(bg="#f0f0f0")

        # ---------- Layout ----------
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # LEFT SIDE: Button panel
        button_panel = tk.Frame(main_frame, width=250, bg="#f8f8f8", relief=tk.RIDGE, bd=2)
        button_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Mode toggle
        mode_frame = tk.Frame(button_panel, bg="#f8f8f8")
        mode_frame.pack(pady=(10, 20))

        self.mode_var = tk.StringVar(value="upload")

        self.upload_radio = tk.Radiobutton(mode_frame, text="Image Upload", variable=self.mode_var, value="upload",
                                      command=self.on_mode_change, bg="#f8f8f8")
        self.upload_radio.pack(side=tk.LEFT, padx=5)

        self.webcam_radio = tk.Radiobutton(mode_frame, text="Webcam Live", variable=self.mode_var, value="webcam",
                                     command=self.on_mode_change, bg="#f8f8f8")
        self.webcam_radio.pack(side=tk.LEFT, padx=5)

        # Buttons
        self.upload_btn = tk.Button(button_panel, text="Upload Image", command=self.upload_image,
                                    width=20, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.upload_btn.pack(pady=(0, 10))

        self.locate_plate_btn = tk.Button(button_panel, text="Detect Plate", command=self.locate_plate,
                                         state=tk.DISABLED, width=20)
        self.locate_plate_btn.pack(pady=10)

        self.enhance_btn = tk.Button(button_panel, text="Enhance Plate", command=self.enhance_plate,
                                     state=tk.DISABLED, width=20)
        self.enhance_btn.pack(pady=10)

        self.read_btn = tk.Button(button_panel, text="Read Plate Text", command=self.read_plate_text,
                                  state=tk.DISABLED, width=20)
        self.read_btn.pack(pady=10)

        self.auto_read_btn = tk.Button(button_panel, text="Auto Detect & Read", command=self.auto_detect_and_read,
                                      state=tk.DISABLED, width=20)
        self.auto_read_btn.pack(pady=30)

        # ---------- CONFIG PANEL ----------
        config_frame = tk.LabelFrame(button_panel, text="Config", bg="#f8f8f8", padx=10, pady=10)
        config_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20)
       
        # YOLO Model selection
        tk.Label(config_frame, text="YOLO Model:", bg="#f8f8f8").pack(anchor="w")
        self.yolo_var = tk.StringVar(value="yolo11m_plates_srb_large.pt")
        yolo_choices = [
            "yolo11m_plates_srb.pt",
            "yolo11m_plates_srb_mid.pt",
            "yolo11m_plates_srb_large.pt"
        ]
        self.yolo_menu = ttk.Combobox(config_frame, textvariable=self.yolo_var, values=yolo_choices, state="readonly", postcommand=self.on_yolo_change)
        self.yolo_menu.pack(fill=tk.X, pady=5)

        # Image enhancement selection
        tk.Label(config_frame, text="Image Enhancement:", bg="#f8f8f8").pack(anchor="w")
        self.enhance_var = tk.StringVar(value="Basic")
        enhance_choices = ["None", "Basic", "Advanced"]
        self.enhance_menu = ttk.Combobox(config_frame, textvariable=self.enhance_var, values=enhance_choices, state="readonly")
        self.enhance_menu.pack(fill=tk.X, pady=5)

        # OCR selection
        tk.Label(config_frame, text="OCR:", bg="#f8f8f8").pack(anchor="w")
        self.ocr_var = tk.StringVar(value="Tesseract")
        ocr_choices = ["Tesseract", "PaddleOCR"]
        self.ocr_menu = ttk.Combobox(config_frame, textvariable=self.ocr_var, values=ocr_choices, state="readonly")
        self.ocr_menu.pack(fill=tk.X, pady=5)

        # ---------- IMAGE PANEL ----------
        image_panel = tk.Frame(main_frame, bg="#ffffff", relief=tk.SOLID, bd=1)
        image_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.image_label = tk.Label(image_panel, bg="#dcdcdc", anchor="center")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # ---------- RESULT PANEL ----------
        self.result_label = tk.Label(self.root, text="", font=("Arial", 14), bg="#e0e0e0", fg="#333333",
                                     wraplength=1000, justify="center", relief=tk.SUNKEN, anchor="center")
        self.result_label.pack(side=tk.BOTTOM, fill=tk.X, pady=5, padx=10, ipady=10)

        # Webcam variables
        self.cap = None
        self.webcam_running = False

        # Initialize models
        tesseract_te.initialize_ocr()
        paddle_te.initialize_ocr()
        pr.initialize_yolo(f"models/yolo/{self.yolo_var.get()}")

        icon = tk.PhotoImage(file="statics/icon.png")
        self.root.iconphoto(True, icon)


    def on_yolo_change(self):
        pr.initialize_yolo(f"models/yolo/{self.yolo_var.get()}")

    def on_mode_change(self):
        mode = self.mode_var.get()
        if mode == "webcam":
            threading.Thread(target=self.start_webcam, daemon=True).start()
            self.hide_buttons()
        else:
            self.stop_webcam()
            self.show_buttons()

        self.clear_image_preview()

    def clear_image_preview(self):
        self.image_label.config(image="")
        self.img_preview = None
        self.cv_image = None
        self.result_label.config(text="")

    def hide_buttons(self):
        self.upload_btn.pack_forget()
        self.locate_plate_btn.pack_forget()
        self.enhance_btn.pack_forget()
        self.read_btn.pack_forget()
        self.auto_read_btn.pack_forget()

    def show_buttons(self):
        self.upload_btn.pack(pady=(0, 10))
        self.locate_plate_btn.pack(pady=10)
        self.enhance_btn.pack(pady=10)
        self.read_btn.pack(pady=10)
        self.auto_read_btn.pack(pady=30)

    def start_webcam(self):
        if self.webcam_running:
            return
        self.cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam Error", "Could not open webcam.")
            self.mode_var.set("upload")
            self.result_label.config(text="Failed to start webcam, switched to upload mode.")
            return
        self.webcam_running = True
        self.update_webcam_frame()
        self.auto_detect_and_read()

    def stop_webcam(self):
        self.webcam_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def update_webcam_frame(self):
        if not self.webcam_running:
            return

        ret, frame = self.cap.read()
        if ret:
            self.cv_image = frame
            cv_img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv_img_rgb)
            label_w = self.image_label.winfo_width() or 640
            label_h = self.image_label.winfo_height() or 480
            pil_img.thumbnail((label_w, label_h))
            self.img_preview = ImageTk.PhotoImage(pil_img)
            self.image_label.config(image=self.img_preview)

        self.root.after(100, self.update_webcam_frame) 

    def upload_image(self):
        filetypes = (
            ("Image files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        )
        path = filedialog.askopenfilename(title="Select an image", filetypes=filetypes)
        if path:
            self.img_path = path
            pil_img = Image.open(path)
            self.cv_image = np.array(pil_img.convert("RGB"))
            self.show_image_preview(pil_img)
            self.result_label.config(text="Image loaded.")
            self.enable_buttons()

    def show_image_preview(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img.thumbnail((400, 300))
        self.img_preview = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.img_preview)

    def auto_detect_and_read(self):
        if not self.img_path and self.cv_image is None:
            return

        if self.mode_var.get() != "webcam":
            self.result_label.config(text="Locating license plate...")
            self.disable_buttons()

        def task():

            plate_position, yolo_confidence = pr.predict(self.cv_image)

            if plate_position:
                self.result_label.config(text="Cropping plate...")

                cropped_image = iu.crop_plate(self.cv_image, plate_position)

                self.result_label.config(text="Extracting text from plate...")

                if self.enhance_var.get() == "None":
                    enhanced_image = cropped_image
                elif self.enhance_var.get() == "Basic":
                    enhanced_image = iu.enhance_photo(cropped_image)
                elif self.enhance_var.get() == "Advanced":
                    enhanced_image = iu.enhance_photo_advanced(cropped_image)
            
                try:
                    if self.ocr_var.get() == "Tesseract":
                        enhanced_image_plate_text, te_confidence = tesseract_te.extract_plate_text(enhanced_image)
                    else:
                        enhanced_image_plate_text, te_confidence = paddle_te.extract_plate_text(enhanced_image)

                    if te_confidence > 0:
                        plate_text = enhanced_image_plate_text
                        self.result_label.config(text=f"Plate number: {plate_text.upper()}, Confidence: {yolo_confidence*te_confidence:.2f}")
                        self.blink_result_label()
                    else:
                        self.result_label.config(text="Couldn't read plate text.")
                        self.blink_result_label("#FF0000")

                except Exception as e:
                    self.result_label.config(text="Error reading plate text.")
                    self.blink_result_label("#FF0000")

            else:
                if self.mode_var.get() != "webcam":    
                    self.result_label.config(text="Couldn't detect plate")
                    self.blink_result_label("#FF0000")

            if self.mode_var.get() == "webcam" and self.webcam_running:
                self.root.after(1000, self.auto_detect_and_read)
                return
        
            if self.mode_var.get() == "upload":
                self.enable_buttons()

        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def locate_plate(self):
        if self.mode_var.get() == "webcam":
            messagebox.showinfo("Info", "Detect Plate is disabled in webcam live mode.")
            return
        if not self.img_path and self.cv_image is None:
            messagebox.showwarning("No image", "Please upload an image first.")
            return

        self.disable_buttons()
        self.result_label.config(text="Locating license plate...")

        def task():

            plate_position, yolo_confidence = pr.predict(self.cv_image)

            if plate_position:
                cropped_image = iu.crop_plate(self.cv_image, plate_position)

                self.img_preview = ImageTk.PhotoImage(Image.fromarray(cropped_image))
                self.cv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                self.image_label.config(image=self.img_preview)

                self.result_label.config(text=f"Plate detected and cropped. Confidence: {yolo_confidence:.2f}")
                self.blink_result_label()
            else:
                self.result_label.config(text="Couldn't detect plate.")
                self.blink_result_label("#FF0000")

            self.enable_buttons()

        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def read_plate_text(self):
        if self.mode_var.get() == "webcam":
            messagebox.showinfo("Info", "Read Plate Text is disabled in webcam live mode.")
            return
        self.disable_buttons()
        self.result_label.config(text="Extracting text from plate...")

        def task():
            if self.ocr_var.get() == "Tesseract":
                plate_text, te_confidence = tesseract_te.extract_plate_text(self.cv_image)
            else:
                try:
                    plate_text, te_confidence = paddle_te.extract_plate_text(self.cv_image)
                except Exception as e:
                    for row in self.cv_image:
                        print(row)
                    plate_text, te_confidence = None, 0.0

            if te_confidence > 0:
                self.result_label.config(text=f"Plate number: {plate_text.upper()}, Confidence: {te_confidence:.2f}")
                self.blink_result_label()
            else:
                self.result_label.config(text="Couldn't read plate text.")
                self.blink_result_label("#FF0000")

            self.enable_buttons()
        
        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def enhance_plate(self):
        if self.mode_var.get() == "webcam":
            messagebox.showinfo("Info", "Enhance Plate is disabled in webcam live mode.")
            return
        self.disable_buttons()
        self.result_label.config(text="Enhancing plate image...")

        def task():
            if self.enhance_var.get() == "None":
                enhanced_image = self.cv_image
            elif self.enhance_var.get() == "Basic":
                enhanced_image = iu.enhance_photo(self.cv_image)
            elif self.enhance_var.get() == "Advanced":
                enhanced_image = iu.enhance_photo_advanced(self.cv_image)

            self.img_preview = ImageTk.PhotoImage(Image.fromarray(enhanced_image))
            self.cv_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
            self.image_label.config(image=self.img_preview)

            self.result_label.config(text="Plate image enhanced.")
            self.blink_result_label()

            self.enable_buttons()

        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def disable_buttons(self):
        self.locate_plate_btn.config(state=tk.DISABLED)
        self.read_btn.config(state=tk.DISABLED)
        self.auto_read_btn.config(state=tk.DISABLED)
        self.enhance_btn.config(state=tk.DISABLED)
        self.upload_btn.config(state=tk.DISABLED)
        self.upload_radio.config(state=tk.DISABLED)
        self.webcam_radio.config(state=tk.DISABLED)
        self.yolo_menu.config(state=tk.DISABLED)
        self.ocr_menu.config(state=tk.DISABLED)
        self.enhance_menu.config(state=tk.DISABLED)

    def enable_buttons(self):
        self.locate_plate_btn.config(state=tk.NORMAL)
        self.read_btn.config(state=tk.NORMAL)
        self.auto_read_btn.config(state=tk.NORMAL)
        self.enhance_btn.config(state=tk.NORMAL)
        self.upload_btn.config(state=tk.NORMAL)
        self.upload_radio.config(state=tk.NORMAL)
        self.webcam_radio.config(state=tk.NORMAL)
        self.yolo_menu.config(state=tk.NORMAL)
        self.ocr_menu.config(state=tk.NORMAL)
        self.enhance_menu.config(state=tk.NORMAL)

    def blink_result_label(self, blink_color="#4CAF50", duration=500):
        original_color = self.result_label.cget("bg")

        def to_accent():
            self.result_label.config(bg=blink_color)
            self.root.after(duration, to_original)

        def to_original():
            self.result_label.config(bg=original_color)

        to_accent()

    

if __name__ == "__main__":
    pass
