import os
import random
from PIL import Image
from torch.utils.data import Dataset

class SignatureTripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Klasörleri listele (Sadece gerçek kullanıcılar)
        self.users = [d for d in os.listdir(root_dir) 
                      if os.path.isdir(os.path.join(root_dir, d)) and not d.endswith('_forg')]
        
        # İmzaları önbelleğe al
        self.user_images = {}
        for user in self.users:
            user_path = os.path.join(self.root_dir, user)
            # Resim uzantılarını kontrol et
            images = [os.path.join(user_path, img) for img in os.listdir(user_path) 
                      if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.user_images[user] = images

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        # 1. Anchor ve Positive Seçimi
        user_name = self.users[index]
        real_imgs = self.user_images[user_name]
        
        if len(real_imgs) >= 2:
            anchor_path, positive_path = random.sample(real_imgs, 2)
        else:
            anchor_path = positive_path = real_imgs[0]

        # 2. Negative Seçimi
        should_get_hard_negative = random.random() < 0.8
        negative_path = None
        
        forg_folder_path = os.path.join(self.root_dir, user_name + '_forg')
        
        # Hard Negative (Sahte İmza)
        if should_get_hard_negative and os.path.exists(forg_folder_path):
            forg_imgs = [os.path.join(forg_folder_path, img) for img in os.listdir(forg_folder_path)
                         if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(forg_imgs) > 0:
                negative_path = random.choice(forg_imgs)
        
        # Random Negative (Başka Kişi)
        if negative_path is None:
            while True:
                other_user = random.choice(self.users)
                if other_user != user_name:
                    negative_path = random.choice(self.user_images[other_user])
                    break

        # 3. Yükleme
        anchor_img = Image.open(anchor_path).convert("L")
        positive_img = Image.open(positive_path).convert("L")
        negative_img = Image.open(negative_path).convert("L")

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img