import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import SignatureTripletDataset

def get_train_loader(root_dir, batch_size=32):
    # --- DATA AUGMENTATION EKLENDİ ---
    train_transforms = transforms.Compose([
        transforms.Resize((128, 224)),
        # Hafif döndürme (İmza atan kişinin el açısı değişebilir)
        transforms.RandomRotation(degrees=10), 
        # Hafif kaydırma ve büyüklük değişimi
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        # Kalem baskısı/Mürekkep farkı simülasyonu
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = SignatureTripletDataset(
        root_dir=root_dir,
        transform=train_transforms
    )

    # Windows'ta num_workers=0 olması daha güvenli
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return train_loader