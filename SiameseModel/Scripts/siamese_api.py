import os
from typing import Dict, Any

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from model import SignatureNet  # Aynı backbone

# Global model/device cache (her çağrıda modeli yeniden yüklememek için)
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model = None
_loaded_model_path = None


def _preprocess_image(image_path: str) -> torch.Tensor:
    """
    Tek bir imza resmini (grayscale) modele uygun tensöre çevirir.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    transform = transforms.Compose([
        transforms.Resize((128, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img = Image.open(image_path).convert('L')
    return transform(img).unsqueeze(0)  # [1, 1, 128, 224]


def _load_siamese_model(model_path: str) -> SignatureNet:
    """
    Modeli bir kez yükler, global cache'te tutar.
    """
    global _model, _loaded_model_path

    if _model is not None and _loaded_model_path == model_path:
        return _model

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = SignatureNet().to(_device)
    state = torch.load(model_path, map_location=_device)
    model.load_state_dict(state)
    model.eval()

    _model = model
    _loaded_model_path = model_path
    return _model


def compare_signatures(
    img1_path: str,
    img2_path: str,
    threshold: float = 0.3075,
    model_path: str = "C:/Users/efegr/OneDrive/Belgeler/PythonProjects/SignatureAuthentication/SiameseModel/models/signature_siamese_best.pth"
) -> Dict[str, Any]:
    """
    İki imza dosyasını (PNG/JPG) karşılaştırır.

    Args:
        img1_path: Birinci imza dosyası yolu.
        img2_path: İkinci imza dosyası yolu.
        threshold: Eşik değer (val setten bulduğumuz: ~0.3075).
        model_path: Eğitilmiş Siamese model ağırlıkları.

    Returns:
        {
            "distance": float,
            "same_writer": bool,
            "threshold": float,
            "decision": "same" | "different"
        }
    """
    model = _load_siamese_model(model_path)

    t1 = _preprocess_image(img1_path).to(_device)
    t2 = _preprocess_image(img2_path).to(_device)

    with torch.no_grad():
        emb1 = model(t1)
        emb2 = model(t2)
        dist = F.pairwise_distance(emb1, emb2).item()

    same = dist < threshold
    decision = "same" if same else "different"

    return {
        "distance": dist,
        "same_writer": same,
        "threshold": threshold,
        "decision": decision,
    }


if __name__ == "__main__":
    # Küçük bir manuel test örneği
    result = compare_signatures(
        "../sign_data/split/test/049/01_049.png",
        "../sign_data/split/test/049/02_049.png",
        threshold=0.3075
    )
    print(result)
