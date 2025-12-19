import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import os

# Senin yazdığın model dosyasını çağırıyoruz
from model import SignatureNet  

def load_model(model_path):
    """
    Eğitilmiş modeli yükler ve test moduna alır.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignatureNet().to(device)
    
    if os.path.exists(model_path):
        # map_location='cpu' -> Eğer GPU'da eğitip CPU'da test ediyorsan hata almamak için
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model yüklendi: {model_path}")
        return model, device
    else:
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

def preprocess_image(image_path):
    """
    Resmi modele girmeden önce hazırlar (Resize, Normalize).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Resim bulunamadı: {image_path}")

    # Eğitimdeki dönüşümlerin aynısı 
    transform = transforms.Compose([
        transforms.Resize((128, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image = Image.open(image_path).convert("L") # Siyah beyaz
    image = transform(image).unsqueeze(0) # Batch boyutu ekle: [1, 1, 128, 224]
    return image

def compare_signatures(model, device, img1_path, img2_path, threshold=0.90):
    """
    İki imza arasındaki mesafeyi ölçer ve kararı verir.
    """
    try:
        img1 = preprocess_image(img1_path).to(device)
        img2 = preprocess_image(img2_path).to(device)

        with torch.no_grad(): 
            emb1 = model(img1)
            emb2 = model(img2)
            
            distance = F.pairwise_distance(emb1, emb2).item()
            
        print(f"\n Karşılaştırma:")
        print(f"   Resim 1: {os.path.basename(img1_path)}")
        print(f"   Resim 2: {os.path.basename(img2_path)}")
        print(f"   Mesafe: {distance:.4f}")
        print(f"   Eşik (Threshold): {threshold}")
        
        if distance < threshold:
            print("   SONUÇ: AYNI KİŞİ (Gerçek İmza)")
        else:
            print("   SONUÇ: FARKLI KİŞİ (Sahte İmza)")
            
    except FileNotFoundError as e:
        print(f"   HATA: {e}")

if __name__ == "__main__":
    # 1. Model Yolu
    model_path = 'models/signature_cnn_augmented.pth'
    
    try:
        model, device = load_model(model_path)
        
        # 2. TEST SENARYOLARI
        
        print("\n--- Test 1: Gerçek vs Gerçek (Aynı Kişi - 049) ---")
        img1 = 'sign_data/split/test/049/01_049.png' 
        img2 = 'sign_data/split/test/049/02_049.png'
        compare_signatures(model, device, img1, img2, threshold=0.90)

        print("\n--- Test 2: Gerçek vs Sahte (Forgery - 049) ---")
        img1 = 'sign_data/split/test/049/01_049.png'
        img3 = 'sign_data/split/test/049_forg/01_0114049.png' 
        compare_signatures(model, device, img1, img3, threshold=0.90)
        
        
    except Exception as e:
        print(f"\nGenel Hata: {e}")