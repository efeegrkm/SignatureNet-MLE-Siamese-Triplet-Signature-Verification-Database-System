import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms
import os
from model import SignatureNet  

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignatureNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    return None, None

def preprocess_and_save(image_path, save_name):
    img = Image.open(image_path).convert("L")

    # 1. Hafif blur (gürültüyü azaltır ama detayı bozmaz)
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    # 2. Hafif kontrast artırma
    img = ImageOps.autocontrast(img)

    # 3. Resize + Normalize (binarization YOK)
    img.save(f"DEBUG_{save_name}")

    transform = transforms.Compose([
        transforms.Resize((128, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    return transform(img).unsqueeze(0)

def test_my_signatures():
    # DİKKAT: Yeni eğittiğimiz 'augmented' modeli kullanıyoruz
    model_path = 'triplet_model/triplet_model/models/signature_cnn_augmented.pth'
    
    # EŞİK DEĞERİ
    # Eğitim sonuçlarına göre 1.0 güvenli bir limandır.
    THRESHOLD = 0.1
    
    # DOSYA YOLLARI
    path_img1 = 'triplet_model/deneme/003_03.PNG'  # Gerçek
    path_img3 = 'triplet_model/deneme/003_01_aug1.png'  # Gerçek
    path_img2 = 'triplet_model/deneme/004_01_aug6.png'  # SAHTE
    
    model, device = load_model(model_path)
    if model is None: return

    print(f"\n--- TEST BAŞLIYOR (Model: Augmented | Eşik: {THRESHOLD}) ---")

    # Fonksiyon ismini düzelttik, artık hata vermeyecek
    img1 = preprocess_and_save(path_img1, "1_gercek.jpg")
    img3 = preprocess_and_save(path_img3, "3_gercek.jpg")
    img2 = preprocess_and_save(path_img2, "2_sahte.jpg")

    if img1 is None or img3 is None or img2 is None: 
        print("Hata: Resimlerden biri yüklenemedi.")
        return

    img1, img3, img2 = img1.to(device), img3.to(device), img2.to(device)

    with torch.no_grad():
        emb1 = model(img1)
        emb3 = model(img3)
        emb2 = model(img2)
        
        dist_real = F.pairwise_distance(emb1, emb3).item()
        dist_fake = F.pairwise_distance(emb1, emb2).item()

    print("\n" + "="*40)
    
    # 1. SEN vs SEN
    print(f"1. Sen vs Sen (Gerçek): {dist_real:.4f}")
    if dist_real < THRESHOLD:
        print("   ✅ ONAYLANDI: Aynı Kişi")
    else:
        print("   ❌ HATA: Kendi imzanı tanıyamadı (Resim çok mu bozuk?)")

    print("-" * 40)

    # 2. SEN vs SAHTE
    print(f"2. Sen vs Sahte:        {dist_fake:.4f}")
    if dist_fake > THRESHOLD:
        print("   ✅ BAŞARILI: Sahte İmza Yakalandı!")
        print(f"      (Güvenlik Marjı: {dist_fake - THRESHOLD:.4f})")
    else:
        print("   ❌ BAŞARISIZ: Sahteyi Yedi")
    
    print("="*40)
    print("\n⚠️ LÜTFEN KONTROL ET: Klasörde oluşan 'DEBUG_...' resimlerini aç.")
    print("   - Eğer bembeyaz boş bir sayfa görüyorsan -> Eşik değeri yanlış.")
    print("   - Eğer simsiyah bir sayfa görüyorsan -> Eşik değeri yanlış.")
    print("   - Net bir imza görüyorsan -> Her şey yolunda.")

if __name__ == "__main__":
    print(">>> custom_test.py başladı")
    test_my_signatures()