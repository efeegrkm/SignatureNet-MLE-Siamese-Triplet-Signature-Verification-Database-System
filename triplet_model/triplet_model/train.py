import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from dataloader import get_train_loader
from model import SignatureNet

def train():
    # --- YENİ AYARLAR ---
    root_dir = 'sign_data/split/train'
    epochs = 200          # Epoch sayısını ciddi şekilde artırdık
    batch_size = 16       # Batch'i küçülttük ki model daha sık güncellensin (Veri az çünkü)
    learning_rate = 0.0005 # Daha hassas öğrenme oranı
    margin = 1.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eğitim şu cihazda yapılacak: {device}")

    train_loader = get_train_loader(root_dir, batch_size)
    model = SignatureNet().to(device)

    criterion = nn.TripletMarginLoss(margin=2.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Eğitim başlıyor (Daha uzun sürecek)...")
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        running_pos_dist = 0.0
        running_neg_dist = 0.0
        
        for i, (anchor, positive, negative) in enumerate(train_loader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)

            loss = criterion(emb_anchor, emb_positive, emb_negative)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # --- TEŞHİS: Mesafeleri hesapla ve kaydet ---
            # Amacımız: Pos_Dist AZALSIN, Neg_Dist ARTSIN
            with torch.no_grad():
                pos_dist = F.pairwise_distance(emb_anchor, emb_positive).mean()
                neg_dist = F.pairwise_distance(emb_anchor, emb_negative).mean()
                running_pos_dist += pos_dist.item()
                running_neg_dist += neg_dist.item()

        # Ortalamaları hesapla
        avg_loss = running_loss / len(train_loader)
        avg_pos = running_pos_dist / len(train_loader)
        avg_neg = running_neg_dist / len(train_loader)

        # Her 10 epoch'ta bir detaylı bilgi ver
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] -> Loss: {avg_loss:.4f} | Pos Dist: {avg_pos:.4f} (Hedef: 0) | Neg Dist: {avg_neg:.4f} (Hedef: >1)")

    # --- KAYDETME ---
    if not os.path.exists('models'):
        os.makedirs('models')
    
    save_path = 'models/signature_cnn_augmented.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Eğitim bitti. Model kaydedildi: {save_path}")

if __name__ == "__main__":
    train()