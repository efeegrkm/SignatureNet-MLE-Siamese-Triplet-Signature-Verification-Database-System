# preprocess.py

import argparse
import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms

# (width, height)
TARGET_SIZE = (400, 400)


def preprocess_image(path: str) -> torch.Tensor:
    """
    1) Görseli griye çevir
    2) (Opsiyonel) autocontrast ile kağıdı beyazlaştır, imzayı koyulaştır
    3) Oranı bozulmadan 400x400 içine sığacak kadar küçült
    4) 400x400 beyaz tuvalin ortasına yapıştır
    5) Tensor + normalize
    """
    img = Image.open(path).convert("L")

    # İstersen bunu kapatabilirsin, ama genelde iş görüyor:
    img = ImageOps.autocontrast(img)

    # Oranı bozulmadan gerekirse küçült (genişlik veya yükseklik 400'ü geçmeyecek)
    img.thumbnail(TARGET_SIZE, Image.LANCZOS)

    # 400x400 beyaz canvas
    canvas_w, canvas_h = TARGET_SIZE
    canvas = Image.new("L", TARGET_SIZE, 255)

    offset_x = (canvas_w - img.width) // 2
    offset_y = (canvas_h - img.height) // 2
    canvas.paste(img, (offset_x, offset_y))

    # Tensor + normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    return transform(canvas)


# ------------------------------
# MAIN – debug için önizleme
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview 400x400 preprocessing.")
    parser.add_argument("--img", required=True, help="Input signature image path")
    parser.add_argument("--save", default=None, help="Optional: save preprocessed image")
    args = parser.parse_args()

    print(f"[INFO] Preprocessing -> {args.img}")
    tensor = preprocess_image(args.img)

    # Normalize'i geri alıp görselleştir
    arr = tensor.squeeze().numpy()
    arr = (arr * 0.5 + 0.5) * 255
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    preview = Image.fromarray(arr, mode="L")
    preview.show()

    if args.save:
        preview.save(args.save)
        print(f"[INFO] Saved preprocessed image to {args.save}")
