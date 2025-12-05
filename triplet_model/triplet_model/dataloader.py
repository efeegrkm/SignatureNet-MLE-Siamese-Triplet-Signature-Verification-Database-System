import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import SignatureTripletDataset


def get_train_loader(root_dir, batch_size=32):

    train_transforms = transforms.Compose([
        transforms.Resize((128, 224)),                
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1)
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = SignatureTripletDataset(
        root_dir=root_dir,
        transform=train_transforms
    )

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    return loader
