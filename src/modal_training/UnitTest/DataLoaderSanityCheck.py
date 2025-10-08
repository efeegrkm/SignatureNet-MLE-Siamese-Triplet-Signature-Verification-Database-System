from modal_training.Data.SignaturePairDataset import SignaturePairDataset
from pathlib import Path
import matplotlib.pyplot as plt
import random

root = Path("C://Users//efegr//OneDrive//Belgeler//PythonProjects//SignatureAuthentication//data//final_data//train") 
dataset = SignaturePairDataset(root, writers=None, transform=None)

num_samples = 10

# Rastgele indexler seç
indices = random.sample(range(len(dataset)), num_samples)

for idx in indices:
    img1, img2, label = dataset[idx]

    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.imshow(img1.squeeze(), cmap='gray')
    plt.title("Image 1")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(img2.squeeze(), cmap='gray')
    plt.title(f"Image 2\nlabel={label}")
    plt.axis('off')

    plt.show()
