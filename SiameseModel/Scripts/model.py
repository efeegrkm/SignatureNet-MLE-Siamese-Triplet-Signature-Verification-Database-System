import torch
import torch.nn as nn
import torch.nn.functional as F

class SignatureNet(nn.Module):
    def __init__(self, embedding_dim=128):
        """
        CNN tabanlı imza embedding modeli.
        Giriş: (Batch_Size, 1, 400, 400)
        Çıkış: (Batch_Size, embedding_dim) -> İmza embedding vektörü
        """
        super(SignatureNet, self).__init__()
        
        # --- EVRİŞİM KATMANLARI ---
        
        # 1. Blok: 400x400 -> 200x200
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # 2. Blok: 200x200 -> 100x100
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 3. Blok: 100x100 -> 50x50
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # 4. Blok: 50x50 -> 25x25
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # --- TAM BAĞLANTI (Embedding) ---
        # Son havuzlama: 256 kanal * 25 * 25
        self.fc_input_dim = 256 * 25 * 25
        
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, embedding_dim)

    def forward(self, x):
        # Evrişim katmanlarından geçir
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(-1, self.fc_input_dim)
        
        # Fully Connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # L2 Normalize (Siamese/Triplet için kritik)
        x = F.normalize(x, p=2, dim=1)
        
        return x


# --- TEST BLOĞU ---
if __name__ == "__main__":
    dummy_input = torch.randn(5, 1, 400, 400)  # yeni giriş boyutu
    model = SignatureNet()
    output = model(dummy_input)
    
    print("CNN Model Testi:")
    print("Giriş  :", dummy_input.shape)
    print("Çıkış :", output.shape)  # Beklenen: [5, embedding_dim]

    if output.shape == (5, model.fc2.out_features):
        print("✅ Model başarıyla oluşturuldu. Eğitim için hazır.")
    else:
        print("❌ Boyut hatası. fc_input_dim yanlış olabilir.")
