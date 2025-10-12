import csv, os, cv2, numpy as np
RAW_ROOT = "data"
IN_CSV = os.path.join(RAW_ROOT, "labels.csv")
OUT_DIR = os.path.join(RAW_ROOT, "processed")
TARGET_SIZE = (224, 224)

def binarize(img_gray):
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    th = cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,35,15)
    return th

def pad_to_square(img):
    h, w = img.shape[:2]
    size = max(h, w)
    top = (size - h)//2; bottom = size - h - top
    left = (size - w)//2; right = size - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

def process_image(path_in, out_path):
    img = cv2.imread(path_in, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[WARN] okunamadı: {path_in}"); return False
    bin_img = binarize(img)
    sq = pad_to_square(bin_img)
    resized = cv2.resize(sq, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, resized)
    return True

def main():
    ok = total = 0
    with open(IN_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            total += 1
            in_path = row["filepath"].replace("/", os.sep)
            rel = os.path.relpath(in_path, start=RAW_ROOT)
            out_path = os.path.join(OUT_DIR, rel)
            if process_image(in_path, out_path): ok += 1
    print(f"[DONE] {ok}/{total} görüntü işlendi. Çıktılar: {OUT_DIR}")

if __name__ == "__main__":
    main()
