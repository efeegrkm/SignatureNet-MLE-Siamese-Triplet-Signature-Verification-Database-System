# scripts/embedding.py
# Basit, hızlı embedding: 64x64 binarize imzayı düz vektör + Sobel histogramı
import cv2, numpy as np

def _ensure_gray(img):
    return img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def preprocess_basic(img):
    """Ham görüntü -> basit binarizasyon + pad + 224x224 (preprocess.py benzeri, hızlı sürüm)"""
    g = _ensure_gray(img)
    g = cv2.GaussianBlur(g, (3,3), 0)
    th = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,35,15)
    h, w = th.shape[:2]; size = max(h, w)
    top = (size - h)//2; bottom = size - h - top
    left = (size - w)//2; right = size - w - left
    sq = cv2.copyMakeBorder(th, top,bottom,left,right, cv2.BORDER_CONSTANT, value=0)
    out = cv2.resize(sq, (224,224), interpolation=cv2.INTER_AREA)
    return out

def embed_from_processed(img224):
    """224x224 binarize img -> 64x64 flatten + sobel(8-bin) -> L2 normalize"""
    x = cv2.resize(img224, (64,64), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    flat = x.reshape(-1)  # 4096

    # Sobel kenar büyüklüğü
    sx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    hist, _ = np.histogram(mag, bins=8, range=(0.0, mag.max() if mag.max()>0 else 1.0))
    hist = hist.astype(np.float32); hist = hist / (np.linalg.norm(hist)+1e-8)  # 8

    v = np.concatenate([flat, hist], axis=0).astype(np.float32)  # 4096+8 = 4104
    v = v / (np.linalg.norm(v)+1e-8)
    return v

def embed_from_file(path, already_processed=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {path}")
    if not already_processed:
        img = preprocess_basic(img)
    return embed_from_processed(img)
