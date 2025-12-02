import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

import torch
import torch.nn.functional as F

from model import SignatureNet          # 400x400 model
from preprocess import preprocess_image # bizim preprocess pipeline

# -------------------------
# Model + Threshold Setup
# -------------------------

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = (
    "C:/Users/efegr/OneDrive/Belgeler/PythonProjects/"
    "SignatureAuthentication/SiameseModel/models/signature_siamese_best.pth"
)

THRESHOLD = 1.2

_model_cache = None


def load_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = SignatureNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    _model_cache = model
    return model


# -------------------------
# GUI Logic
# -------------------------

class SiameseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Signature Verification (Siamese Network)")
        self.root.geometry("1100x600")
        self.root.configure(bg="#f2f2f2")

        self.img1_path = None
        self.img2_path = None

        self.proc_img1 = None  # PIL image (preprocessed 400x400)
        self.proc_img2 = None

        self.tk_img1 = None
        self.tk_img2 = None

        tk.Label(
            root,
            text="Signature Verification (Siamese Network)",
            font=("Arial", 20, "bold"),
            bg="#f2f2f2"
        ).pack(pady=15)

        frame = tk.Frame(root, bg="#f2f2f2")
        frame.pack(pady=10)

        # Left panel
        self.left_label = tk.Label(
            frame,
            text="Image 1 (Preprocessed)",
            bg="#dddddd",
            relief="sunken"
        )

        self.left_label.grid(row=0, column=0, padx=30)

        # Right panel
        self.right_label = tk.Label(
            frame,
            text="Image 2 (Preprocessed)",
            bg="#dddddd",
            relief="sunken"
        )
        self.right_label.grid(row=0, column=2, padx=30)

        # Buttons
        tk.Button(frame, text="Load Image 1", command=self.load_img1).grid(row=1, column=0, pady=5)
        tk.Button(frame, text="Load Image 2", command=self.load_img2).grid(row=1, column=2, pady=5)

        self.result_label = tk.Label(root, text="", font=("Arial", 16), bg="#f2f2f2")
        self.result_label.pack(pady=25)

        tk.Button(
            root,
            text="Guess",
            command=self.run_inference,
            font=("Arial", 14, "bold"),
            bg="#4CAF50", fg="white",
            width=15
        ).pack()

    # ------------------------------------------------------
    # LOAD IMAGE 1 + PREPROCESS + DISPLAY
    # ------------------------------------------------------
    def load_img1(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if not path:
            return

        self.img1_path = path
        self.proc_img1 = self.get_preprocessed_display_image(path)
        self.tk_img1 = ImageTk.PhotoImage(self.proc_img1)

        self.left_label.config(image=self.tk_img1, text="")

    # ------------------------------------------------------
    # LOAD IMAGE 2 + PREPROCESS + DISPLAY
    # ------------------------------------------------------
    def load_img2(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if not path:
            return

        self.img2_path = path
        self.proc_img2 = self.get_preprocessed_display_image(path)
        self.tk_img2 = ImageTk.PhotoImage(self.proc_img2)

        self.right_label.config(image=self.tk_img2, text="")

    # ------------------------------------------------------
    # Convert preprocess tensor → displayable PIL Image
    # ------------------------------------------------------
    def get_preprocessed_display_image(self, path):
        tensor = preprocess_image(path)   # shape: (1,400,400)

        # Denormalize
        arr = tensor.squeeze().numpy()
        arr = (arr * 0.5 + 0.5) * 255
        arr = arr.clip(0, 255).astype("uint8")

        img = Image.fromarray(arr, mode="L")  # 400x400
        return img.resize((300, 300))         # GUI için büyütülmüş görünüm

    # ------------------------------------------------------
    # INFERENCE (Model run)
    # ------------------------------------------------------
    def run_inference(self):
        if not self.img1_path or not self.img2_path:
            messagebox.showerror("Error", "Please load 2 images first.")
            return

        try:
            model = load_model()

            t1 = preprocess_image(self.img1_path).unsqueeze(0).to(DEVICE)
            t2 = preprocess_image(self.img2_path).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                emb1 = model(t1)
                emb2 = model(t2)
                dist = F.pairwise_distance(emb1, emb2).item()

            same = dist < THRESHOLD
            decision = "SAME (genuine)" if same else "DIFFERENT (imposter)"

            self.result_label.config(
                text=f"Distance: {dist:.4f}   →   {decision}",
                fg="green" if same else "red"
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))


# -------------------------
# RUN GUI
# -------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = SiameseGUI(root)
    root.mainloop()
