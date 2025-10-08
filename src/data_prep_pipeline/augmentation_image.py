import cv2
import numpy as np
import os
from tqdm import tqdm
import random
import shutil
from pathlib import Path

# --- Augmentation fonksiyonları ---
def rotate_image(img, angle_deg):
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle_deg, 1)
    return cv2.warpAffine(img, M, (cols, rows), borderValue=0)

def scale_image(img, scale):
    rows, cols = img.shape
    resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    r, c = resized.shape
    new_img = np.zeros_like(img)

    start_row = max((rows - r)//2, 0)
    start_col = max((cols - c)//2, 0)
    end_row = start_row + min(r, rows)
    end_col = start_col + min(c, cols)
    src_start_row = max((r - rows)//2, 0)
    src_start_col = max((c - cols)//2, 0)
    src_end_row = src_start_row + (end_row - start_row)
    src_end_col = src_start_col + (end_col - start_col)

    new_img[start_row:end_row, start_col:end_col] = resized[src_start_row:src_end_row, src_start_col:src_end_col]
    return new_img

def translate_image(img, tx, ty):
    rows, cols = img.shape
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (cols, rows), borderValue=0)

def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
    noisy = img.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def dilate_erode_image(img, mode='dilate', ksize=2):
    kernel = np.ones((ksize, ksize), np.uint8)
    if mode == 'dilate':
        return cv2.dilate(img, kernel, iterations=1)
    elif mode == 'erode':
        return cv2.erode(img, kernel, iterations=1)
    else:
        return img


# --- Dinamik pathler ---
current_path = Path(__file__).resolve()
project_root = current_path.parent.parent.parent
preprocessed_root = project_root / "data" / "processed_data" / "train" / "genuine"
augmented_root = project_root / "data" / "augmented_data" / "train" / "genuine"

# --- Klasörü temizle / oluştur ---
if augmented_root.exists():
    shutil.rmtree(augmented_root)
augmented_root.mkdir(parents=True, exist_ok=True)

num_augmentations = 8  # her imza için kaç varyasyon

# --- Augmentation pipeline ---
for writer_id in tqdm(os.listdir(preprocessed_root)):
    in_dir = preprocessed_root / writer_id
    out_dir = augmented_root / writer_id
    out_dir.mkdir(exist_ok=True)

    for img_name in os.listdir(in_dir):
        img_path = in_dir / img_name
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Resim okunamadı: {img_path}")
            continue

        # Orijinal resmi kopyala
        cv2.imwrite(str(out_dir / img_name), img)

        for i in range(num_augmentations):
            aug = img.copy()

            # Random augmentation karışımı
            if random.random() < 0.5:
                aug = rotate_image(aug, random.uniform(-30, 30))
            if random.random() < 0.5:
                aug = scale_image(aug, random.uniform(0.8, 1.1))
            if random.random() < 0.5:
                tx = random.randint(-15, 15)
                ty = random.randint(-20, 20)
                aug = translate_image(aug, tx, ty)
            if random.random() < 0.5:
                aug = add_gaussian_noise(aug, sigma=random.randint(5, 15))
            if random.random() < 0.5:
                mode = random.choice(['dilate', 'erode'])
                k = 1 if mode == 'erode' else random.randint(1, 2)
                aug = dilate_erode_image(aug, mode=mode, ksize=k)


            out_name = f"{img_name.split('.')[0]}_aug{i}.png"
            cv2.imwrite(str(out_dir / out_name), aug)
