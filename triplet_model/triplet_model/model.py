import torch
import torch.nn as nn
import torch.nn.functional as F

class SignatureNet(nn.Module):
    def __init__(self, embedding_dim=256):
        """
        Sıfırdan eğitilecek özel CNN Mimarisi.
        Giriş: (Batch_Size, 1, 128, 224) -> Gri tonlamalı imza resmi
        Çıkış: (Batch_Size, 128) -> İmzanın kimlik vektörü (Embedding)
        """
        super(SignatureNet, self).__init__()
        
        # --- EVRİŞİM KATMANLARI (Feature Extraction) ---
        
        # 1. Blok: Temel çizgileri algıla
        # Giriş: 1 kanal (Siyah-Beyaz) -> Çıkış: 32 kanal
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)       # Eğitimi hızlandırır ve dengeler
        self.pool1 = nn.MaxPool2d(2, 2)     # Boyutu yarıya indirir (128x224 -> 64x112)
        
        # 2. Blok: Şekilleri algıla
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)     # (64x112 -> 32x56)
        
        # 3. Blok: Karmaşık detayları algıla
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)     # (32x56 -> 16x28)
        
        # 4. Blok: Soyut özellikleri çıkar
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)     # (16x28 -> 8x14)
        
        # --- SINIFLANDIRMA KATMANLARI (Embedding) ---
        # Son havuzlama işleminden sonra elimizde 256 adet 8x14'lük harita kalır.
        self.fc_input_dim = 256 * 8 * 14 
        
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.dropout = nn.Dropout(0.5)      # Aşırı öğrenmeyi (overfitting) engeller
        self.fc2 = nn.Linear(512, embedding_dim)

    def forward(self, x):
        # Evrişim Bloklarından geçiş
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Veriyi düzleştir (Matrix -> Vektör)
        x = x.view(-1, self.fc_input_dim)
        
        # Tam bağlantılı katmanlar
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # KRİTİK ADIM: L2 Normalizasyonu
        # Triplet Loss için vektörlerin boyunun 1'e eşitlenmesi (birim çember)
        # mesafelerin (distance) doğru hesaplanması için çok önemlidir.
        x = F.normalize(x, p=2, dim=1)
        
        return x

# --- TEST BLOĞU ---
if __name__ == "__main__":
    # Modelin boyutlarının doğru çalışıp çalışmadığını test et
    dummy_input = torch.randn(5, 1, 128, 224) # 5 adet rastgele resim
    model = SignatureNet()
    output = model(dummy_input)
    
    print("CNN Model Testi:")
    print(f"Giriş Boyutu: {dummy_input.shape}")
    print(f"Çıkış Boyutu: {output.shape}") # Beklenen: [5, 128]
    
    if output.shape == (5, 128):
        print("✅ Model başarıyla oluşturuldu. Triplet eğitimine hazır.")
    else:
        print("❌ Model çıkış boyutunda hata var.")