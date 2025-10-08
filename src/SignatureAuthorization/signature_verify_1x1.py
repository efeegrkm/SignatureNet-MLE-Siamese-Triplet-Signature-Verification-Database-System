import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np

# =========================
# Siamese Model Definition
# =========================
class SiameseResNet18(nn.Module):
    def __init__(self, embedding_dim=256):
        super(SiameseResNet18, self).__init__()
        base_model = models.resnet18(weights=None)
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # grayscale input
        base_model.fc = nn.Linear(base_model.fc.in_features, embedding_dim)
        self.encoder = base_model

    def forward_once(self, x):
        return self.encoder(x)

    def forward(self, img1, img2):
        out1 = self.forward_once(img1)
        out2 = self.forward_once(img2)
        return out1, out2


# =========================
# Model Loading Function
# =========================
def load_model(model_path, device):
    from collections import OrderedDict

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = SiameseResNet18()
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            k = k.replace("backbone.", "encoder.")
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully from: {model_path}")
    return model


# =========================
# Preprocess Function
# =========================
def preprocess_image_array(input_path, size=(224, 224)):
    """
    Eğitimdekiyle aynı preprocessing: grayscale -> threshold -> crop -> pad -> resize
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Okunamadı: {input_path}")

    # Invert + Otsu threshold
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(img_bin)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = img_bin[y:y+h, x:x+w]
    else:
        cropped = img_bin

    pad_ratio = 0.2
    new_w, new_h = int(w * (1 + pad_ratio)), int(h * (1 + pad_ratio))
    max_side = max(new_w, new_h)
    canvas = np.zeros((max_side, max_side), dtype=np.uint8)
    y_offset, x_offset = (max_side - h) // 2, (max_side - w) // 2
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

    resized = cv2.resize(canvas, size, interpolation=cv2.INTER_AREA)
    return resized


# =========================
# Verification Function
# =========================
def verify_signature(model, img_path1, img_path2, device, threshold=1.0):
    """
    İki imzanın aynı kişiye ait olup olmadığını belirler.
    """
    if not os.path.exists(img_path1) or not os.path.exists(img_path2):
        raise FileNotFoundError("❌ Image path(s) not found.")

    # Görselleri eğitimdekiyle aynı şekilde preprocess et
    img1 = preprocess_image_array(img_path1)
    img2 = preprocess_image_array(img_path2)

    # Torch tensöre dönüştür
    img1 = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    img2 = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

    # Normalize
    img1 = (img1 - 0.5) / 0.5
    img2 = (img2 - 0.5) / 0.5

    img1, img2 = img1.to(device), img2.to(device)

    with torch.no_grad():
        out1, out2 = model(img1, img2)
        distance = F.pairwise_distance(out1, out2).item()

    print(f"\n🔹 Euclidean Distance: {distance:.4f}")
    if distance < threshold:
        print("✅ SAME PERSON (Genuine Match)")
        return True
    else:
        print("❌ DIFFERENT PERSON (Forgery or Mismatch)")
        return False


# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    MODEL_PATH = "C://Users//efegr//OneDrive//Belgeler//PythonProjects//SignatureAuthentication//outputs//checkpoints//FirstEpoch96Val//best_model_epoch1.pth"
    IMG_PATH1 = "C://Users//efegr//OneDrive//Belgeler//PythonProjects//SignatureAuthentication//data//RealWorldTestData//RealWorldTestDataRaw//Yagmur_Genuine1.png"
    IMG_PATH2 = "C://Users//efegr//OneDrive//Belgeler//PythonProjects//SignatureAuthentication//data//RealWorldTestData//RealWorldTestDataRaw//Yagmur_Genuine2.png"
    THRESHOLD = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    model = load_model(MODEL_PATH, device)
    verify_signature(model, IMG_PATH1, IMG_PATH2, device, threshold=THRESHOLD)
