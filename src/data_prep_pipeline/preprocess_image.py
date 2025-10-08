
import os
import cv2
import numpy as np
from tqdm import tqdm

# ---------------- helpers ----------------
def find_split_root_by_name(start_dir, target_name='data', max_up_levels=6):
    """
    start_dir'den başlayıp yukarı çıkarak 'target_name' adlı klasörü arar.
    Bulursa tam yolu döner. Bulamazsa FileNotFoundError fırlatır.
    """
    cur = os.path.abspath(start_dir)
    tried = []
    for _level in range(max_up_levels + 1):
        candidate = os.path.join(cur, target_name)
        tried.append(candidate)
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(cur)
        if parent == cur:  # reached filesystem root
            break
        cur = parent
    raise FileNotFoundError(
        f"'{target_name}' klasörü bulunamadı. Aşağıdaki yollar kontrol edildi:\n" +
        "\n".join(tried)
    )

def preprocess_image(input_path, output_path, size=(224, 224)):
    """
    İmzayı ortalayarak kare tuvale yerleştirir, padding ekler ve fit-scale ile yeniden boyutlandırır.
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[WARN] Okunamadı: {input_path}")
        return

    # Otsu threshold
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # bounding box crop
    coords = cv2.findNonZero(img_bin)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = img_bin[y:y+h, x:x+w]
    else:
        cropped = img_bin

    # --- Kare tuval oluştur (padding dahil) ---
    pad_ratio = 0.2  # %30 padding
    new_w = int(w * (1 + pad_ratio))
    new_h = int(h * (1 + pad_ratio))
    max_side = max(new_w, new_h)

    canvas = np.zeros((max_side, max_side), dtype=np.uint8)
    y_offset = (max_side - h) // 2
    x_offset = (max_side - w) // 2
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

    # --- Fit to target size ---
    resized = cv2.resize(canvas, size, interpolation=cv2.INTER_AREA)

    cv2.imwrite(output_path, resized)


# ---------------- main ----------------
if __name__ == "__main__":
    # script'in bulunduğu dizin
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # split_root klasörü 'data' adıyla aynı seviyede aranacak (yukarı düzeylerde)
    try:
        input_root = find_split_root_by_name(current_dir, target_name='data', max_up_levels=6)
    except FileNotFoundError as e:
        print("[ERROR]", e)
        raise

    # output: data/processed_data
    output_root = os.path.join(input_root, "processed_data")
    os.makedirs(output_root, exist_ok=True)

    print(f"[INFO] Bulunan split_root (data): {input_root}")
    print(f"[INFO] Oluşturulan output_root: {output_root}")

    # expected alt dizinler
    for subset in ["train", "test"]:
        for kind in ["genuine", "forgery"]:
            input_dir = os.path.join(input_root, subset, kind)
            output_dir = os.path.join(output_root, subset, kind)
            if not os.path.isdir(input_dir):
                print(f"[WARN] Beklenen yol yok (atlandı): {input_dir}")
                continue
            os.makedirs(output_dir, exist_ok=True)

            # yazarlara göre işle
            writer_list = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
            for writer_id in tqdm(writer_list, desc=f"{subset}/{kind}"):
                in_writer = os.path.join(input_dir, writer_id)
                out_writer = os.path.join(output_dir, writer_id)
                os.makedirs(out_writer, exist_ok=True)

                for fname in os.listdir(in_writer):
                    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
                        continue
                    src = os.path.join(in_writer, fname)
                    dst = os.path.join(out_writer, fname)
                    preprocess_image(src, dst)

    print("[DONE] Preprocessing tamamlandı.")