import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import os
from model import SignatureNet


def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignatureNet().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def preprocess(image_path):
    img = Image.open(image_path).convert("L")

    transform = transforms.Compose([
        transforms.Resize((128, 224)),          # ❗ sabit resize (aspect ratio yok)
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    return transform(img).unsqueeze(0)


def test_my_signatures():
    THRESHOLD = 0.10   # eski modelde bulunan doğru threshold

    model_path = "triplet_model/triplet_model/models/signature_cnn_augmented.pth"
    img1_path = "triplet_model/deneme/e1.jpg"
    img2_path = "triplet_model/deneme/e2.jpg"
    img_fake_path = "triplet_model/deneme/my3.jpg"

    model, device = load_model(model_path)

    img1 = preprocess(img1_path).to(device)
    img2 = preprocess(img2_path).to(device)
    img_fake = preprocess(img_fake_path).to(device)

    with torch.no_grad():
        dist_real = F.pairwise_distance(model(img1), model(img2)).item()
        dist_fake = F.pairwise_distance(model(img1), model(img_fake)).item()

    print("\n--- SIGNATURE TEST ---")
    print(f"Gerçek vs Gerçek: {dist_real:.4f}")
    print(f"Gerçek vs Sahte:  {dist_fake:.4f}")
