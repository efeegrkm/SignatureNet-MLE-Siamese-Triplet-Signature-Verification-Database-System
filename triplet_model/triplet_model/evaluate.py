import torch
import torch.nn.functional as F
import os
from PIL import Image
import torchvision.transforms as transforms
from model import SignatureNet

# --- AYARLAR ---
MODEL_PATH = 'models/signature_cnn_augmented.pth'
TEST_DIR = 'sign_data/split/test'
THRESHOLD = 0.90  # Belirlediğimiz eşik değeri

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignatureNet().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model, device
    return None, None

def get_embedding(model, device, img_path):
    transform = transforms.Compose([
        transforms.Resize((128, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    try:
        img = Image.open(img_path).convert("L")
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(img)
        return emb
    except Exception:
        return None

def evaluate_accuracy():
    model, device = load_model()
    if model is None:
        print("Model bulunamadı!")
        return

    print(f"Test Başlıyor... (Eşik Değeri: {THRESHOLD})")
    
    # İstatistikler
    correct_preds = 0
    total_preds = 0
    
    tp = 0 # True Positive (Gerçeği bildi)
    tn = 0 # True Negative (Sahteyi bildi)
    fp = 0 # False Positive (Sahteye gerçek dedi - TEHLİKELİ)
    fn = 0 # False Negative (Gerçeğe sahte dedi)

    users = [d for d in os.listdir(TEST_DIR) if not d.endswith('_forg')]

    for user in users:
        user_path = os.path.join(TEST_DIR, user)
        forg_path = os.path.join(TEST_DIR, user + '_forg')
        
        # Resimleri topla
        real_imgs = [os.path.join(user_path, f) for f in os.listdir(user_path) if f.endswith('.png')]
        forg_imgs = []
        if os.path.exists(forg_path):
            forg_imgs = [os.path.join(forg_path, f) for f in os.listdir(forg_path) if f.endswith('.png')]

        if len(real_imgs) < 2:
            continue

        # Referans olarak ilk gerçek imzayı al
        ref_img_path = real_imgs[0]
        ref_emb = get_embedding(model, device, ref_img_path)
        if ref_emb is None: continue

        # 1. POZİTİF TESTLER (Gerçek vs Gerçek)
        # İlk imza hariç diğer tüm gerçek imzaları referansla kıyasla
        for real_img_path in real_imgs[1:]:
            test_emb = get_embedding(model, device, real_img_path)
            if test_emb is None: continue
            
            dist = F.pairwise_distance(ref_emb, test_emb).item()
            total_preds += 1
            
            if dist < THRESHOLD:
                correct_preds += 1
                tp += 1
            else:
                fn += 1 # Hata: Gerçeğe sahte dedi

        # 2. NEGATİF TESTLER (Gerçek vs Sahte)
        for forg_img_path in forg_imgs:
            test_emb = get_embedding(model, device, forg_img_path)
            if test_emb is None: continue
            
            dist = F.pairwise_distance(ref_emb, test_emb).item()
            total_preds += 1
            
            if dist > THRESHOLD:
                correct_preds += 1
                tn += 1
            else:
                fp += 1 # Hata: Sahteye gerçek dedi

    # SONUÇLARI YAZDIR
    accuracy = 100 * correct_preds / total_preds
    
    print("-" * 30)
    print(f"📊 TOPLAM TEST SAYISI: {total_preds}")
    print(f"✅ DOĞRU TAHMİN: {correct_preds}")
    print(f"🏆 GENEL BAŞARI (ACCURACY): %{accuracy:.2f}")
    print("-" * 30)
    print("Detaylar:")
    print(f"   Doğru Kabul (True Positive): {tp}")
    print(f"   Doğru Red (True Negative): {tn}")
    print(f"   Yanlış Kabul (False Positive): {fp} (En Kritik Hata)")
    print(f"   Yanlış Red (False Negative): {fn}")

if __name__ == "__main__":
    evaluate_accuracy()