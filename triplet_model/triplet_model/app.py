import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st

# ===========================
#   MODEL ve DOSYA YOLLARI
# ===========================

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "models", "signature_cnn_augmented.pth")
EMB_DB_PATH = os.path.join(BASE_DIR, "embeddings.pt")
USERS_PATH = os.path.join(BASE_DIR, "users.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
#   MODELİ İÇE AKTAR
# ===========================

from model import SignatureNet   # ✔ senin gerçek modelin (dosyadan geliyor)

def load_model():
    """Modeli cache'leyerek 1 kez yükler."""
    if "model" in st.session_state:
        return st.session_state["model"]

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model bulunamadı: {MODEL_PATH}")
        st.stop()

    model = SignatureNet().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    st.session_state["model"] = model
    return model

model = load_model()

# ===========================
#   TRANSFORM (EĞİTİMLE AYNI)
# ===========================

transform = transforms.Compose([
    transforms.Resize((128, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def get_embedding(img: Image.Image):
    """Tek imzadan embedding üretir."""
    img = img.convert("L")
    t = transform(img)
    t = t.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(t)

    return emb.squeeze(0).cpu()  # (256,)

# ===========================
#   VERITABANI FONKSIYONLARI
# ===========================

def load_db():
    if os.path.exists(EMB_DB_PATH):
        emb_db = torch.load(EMB_DB_PATH, map_location="cpu")
    else:
        emb_db = {}

    if os.path.exists(USERS_PATH):
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            users = json.load(f)
    else:
        users = []

    return emb_db, users


def save_db(emb_db, users):
    torch.save(emb_db, EMB_DB_PATH)
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def compute_min_distance(emb, emb_list):
    if not emb_list:
        return None
    dists = [torch.norm(emb - e).item() for e in emb_list]
    return min(dists)

# ===========================
#   STREAMLIT UI
# ===========================

st.set_page_config(page_title="Signature Verification", layout="centered")
st.title("✒️ İmza Tanıma Sistemi")

emb_db, users = load_db()
id2name = {u["user_id"]: u["name"] for u in users}

mode = st.sidebar.selectbox(
    "Mod Seç",
    ["Yeni Kayıt", "Doğrulama (Bu kişi mi?)", "Veritabanında Ara"]
)

# ===========================
#   1) YENİ KİŞİ KAYDI
# ===========================

if mode == "Yeni Kayıt":
    st.header("Yeni Kullanıcı Kaydı")

    name = st.text_input("Ad Soyad")
    imgs = st.file_uploader(
        "En az 3 imza yükle:",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg"]
    )

    if st.button("Kaydı Oluştur"):
        if not name or len(imgs) < 3:
            st.error("3 adet imza ve isim gerekli!")
        else:
            new_id = f"u_{len(users)+1:04d}"
            st.success(f"Kullanıcı ID: {new_id}")

            emb_list = []
            for f in imgs:
                img = Image.open(f)
                emb = get_embedding(img)
                emb_list.append(emb)

            emb_db[new_id] = {"embeddings": emb_list}
            users.append({"user_id": new_id, "name": name})

            save_db(emb_db, users)
            st.success(f"✔ {name} başarıyla kaydedildi!")

# ===========================
#   2) DOĞRULAMA
# ===========================

elif mode == "Doğrulama (Bu kişi mi?)":
    st.header("İmza Doğrulama")

    if not users:
        st.warning("Veritabanı boş.")
    else:
        selected = st.selectbox("Kişi Seç", [u["name"] for u in users])
        uid = [u["user_id"] for u in users if u["name"] == selected][0]

        file = st.file_uploader("Test imzası yükle", type=["png", "jpg", "jpeg"])
        threshold = st.slider("Eşik Değer", 0.0, 5.0, 1.0, 0.1)

        if file and st.button("Doğrula"):
            img = Image.open(file)
            emb = get_embedding(img)

            emb_list = emb_db[uid]["embeddings"]
            d_min = compute_min_distance(emb, emb_list)

            st.write(f"Mesafe: **{d_min:.3f}**")

            if d_min < threshold:
                st.success("✔ Gerçek imza (aynı kişi)")
            else:
                st.error("✘ Sahte imza (farklı kişi)")

# ===========================
#   3) VERITABANINDA ARAMA
# ===========================

else:
    st.header("Veritabanında Arama")

    file = st.file_uploader("Test imzası yükle", type=["png", "jpg", "jpeg"])
    threshold = st.slider("Global Eşik", 0.0, 5.0, 1.0, 0.1)

    if file and st.button("Ara"):
        img = Image.open(file)
        emb = get_embedding(img)

        best_id = None
        best_dist = None

        for uid, data in emb_db.items():
            d = compute_min_distance(emb, data["embeddings"])
            if d is None:
                continue
            if best_dist is None or d < best_dist:
                best_dist = d
                best_id = uid

        if best_id is None:
            st.error("Veritabanı boş!")
        else:
            name = id2name[best_id]
            st.write(f"En yakın kişi: **{name}** (mesafe={best_dist:.3f})")

            if best_dist < threshold:
                st.success("✔ Bu imza veritabanındaki bir kişiye benziyor!")
            else:
                st.info("Bu imza kimseye benzemiyor. Yeni kişi olarak eklenebilir.")
