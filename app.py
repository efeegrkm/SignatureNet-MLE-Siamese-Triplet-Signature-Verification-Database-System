import os
import sys
import json
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
import streamlit as st

# =====================================================
# GENEL AYARLAR
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Buradan hangi modeli kullanacağını KODDAN seçiyorsun:
ACTIVE_MODEL_NAME = "Triplet"      # "Triplet" veya "Siamese"

USERS_PATH = BASE_DIR / "users.json"

MODEL_CONFIG = {
    "Triplet": {
        # İç içe klasör: triplet_model/triplet_model
        "code_dir": BASE_DIR / "triplet_model" / "triplet_model",
        "model_module": "model",
        "model_class": "SignatureNet",
        "weights": BASE_DIR / "triplet_model" / "triplet_model" / "models" / "signature_cnn_augmented.pth",
        "emb_path": BASE_DIR / "embeddings_triplet.pt",
        "th_min": 0.0,
        "th_max": 2.0,
        "th_default": 0.10,
    },
    "Siamese": {
        "code_dir": BASE_DIR / "SiameseModel",
        "model_module": "model",
        "model_class": "SignatureNet",      # Siamese modelde sınıf adı farklıysa değiştir
        "weights": BASE_DIR / "SiameseModel" / "siamese.pth",
        "emb_path": BASE_DIR / "embeddings_siamese.pt",
        "th_min": 0.0,
        "th_max": 2.0,
        "th_default": 0.50,
    },
}

CFG = MODEL_CONFIG[ACTIVE_MODEL_NAME]

# =====================================================
# GÖRSEL TEMA
# =====================================================
st.set_page_config(page_title="Signature Verification", layout="centered")

st.markdown("""
<style>
/* ARKA PLAN ve BAŞLIKLAR */
body {
    background: linear-gradient(135deg, #1e272e, #2f3640);
}

.main-title {
    text-align:center;
    font-size: 2.6rem;
    font-weight: 800;
    color: #000000;  /* düz siyah */
    margin-top: 0.5rem;
    margin-bottom: 0.2rem;
}

.subtitle {
    text-align:center;
    color:#f1f2f6;
    margin-bottom: 2rem;
    font-size: 0.95rem;
}

.main-card {
    background: rgba(255,255,255,0.06);
    border-radius: 1.5rem;
    padding: 2rem 2.5rem;
    box-shadow: 0 18px 45px rgba(0,0,0,0.45);
    backdrop-filter: blur(14px);
}

/* GENEL BUTON STİLİ */
.stButton > button {
    border: none;
    height: 5rem;
    font-size: 1.4rem;
    font-weight: 700;
    border-radius: 1.2rem;
    padding: 0.5rem 1rem;
    box-shadow: 0 10px 25px rgba(0,0,0,0.25);
    cursor: pointer;
    transition: 0.2s;
}

/* HOVER EFEKTİ */
.stButton > button:hover {
    transform: scale(1.04);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}

/* ANA EKRANDAKİ ÜÇ BÜYÜK BUTON */

/* DİKKAT: class doğrudan button’a ekleniyor, o yüzden selector:
   .stButton > button.big-btn-1 şeklinde olmalı
*/
.stButton > button.big-btn-1 {
    width: 100%;
    background: linear-gradient(135deg,#ff3b3b,#ff9f1a);
    color: white;
}

.stButton > button.big-btn-2 {
    width: 100%;
    background: linear-gradient(135deg,#00b894,#0984e3);
    color: white;
}

.stButton > button.big-btn-3 {
    width: 100%;
    background: linear-gradient(135deg,#6c5ce7,#d63031);
    color: white;
}

/* DİĞER (Kaydet, Doğrula, Geri Dön) BUTONLAR İÇİN (isteğe bağlı) */
.stButton > button.small-btn {
    background: #ffffff;
    color: #2f3640;
    height: 3.5rem;
    font-size: 1.1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title">✒️ Signature Verification</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="subtitle">Aktif model: <b>{ACTIVE_MODEL_NAME}</b> – yeni kullanıcı ekle, kayıtlı kullanıcıyı doğrula veya iki imzayı karşılaştır</div>',
    unsafe_allow_html=True,
)


# =====================================================
# TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.Resize((128, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# =====================================================
# YARDIMCI FONKSİYONLAR
# =====================================================
def load_users():
    if USERS_PATH.exists():
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_users(users):
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def load_embeddings():
    if CFG["emb_path"].exists():
        return torch.load(CFG["emb_path"], map_location="cpu")
    return {}


def save_embeddings(emb_db):
    torch.save(emb_db, CFG["emb_path"])


def load_model():
    if "model" in st.session_state:
        return st.session_state["model"]

    if not CFG["weights"].exists():
        st.error(f"Model ağırlığı bulunamadı:\n{CFG['weights']}")
        st.stop()

    sys.path.insert(0, str(CFG["code_dir"]))
    module = __import__(CFG["model_module"])
    ModelClass = getattr(module, CFG["model_class"])

    model = ModelClass().to(DEVICE)
    state = torch.load(CFG["weights"], map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    st.session_state["model"] = model
    return model


def get_embedding(img: Image.Image) -> torch.Tensor:
    model = load_model()
    img = img.convert("L")
    t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model(t)
    return emb.squeeze(0).cpu()


def compute_min_distance(emb, emb_list):
    if not emb_list:
        return None
    dists = [torch.norm(emb - e).item() for e in emb_list]
    return min(dists)


def get_or_create_user_id(users, name):
    for u in users:
        if u["name"] == name:
            return u["user_id"], users, False
    new_id = f"u_{len(users) + 1:04d}"
    users.append({"user_id": new_id, "name": name})
    return new_id, users, True

# =====================================================
# STATE
# =====================================================
if "mode" not in st.session_state:
    st.session_state["mode"] = None

users = load_users()
emb_db = load_embeddings()
id2name = {u["user_id"]: u["name"] for u in users}


# =====================================================
# ANA EKRAN: 3 BÜYÜK BUTON
# =====================================================
if st.session_state["mode"] is None:

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🆕 Yeni Kullanıcı", key="btn_new", use_container_width=True):
            st.session_state["mode"] = "new"

    with col2:
        if st.button("🖊️ Kayıtlı Kullanıcı", key="btn_existing", use_container_width=True):
            st.session_state["mode"] = "existing"

    with col3:
        if st.button("🔁 İki İmzayı Karşılaştır", key="btn_compare", use_container_width=True):
            st.session_state["mode"] = "compare"

    # BUTONLARA RENK SINIFI EKLEYEN SCRIPT
    st.markdown("""
    <script>
    const btns = window.parent.document.querySelectorAll('.stButton button');
    if (btns.length >= 3) {
        btns[0].classList.add('big-btn-1');
        btns[1].classList.add('big-btn-2');
        btns[2].classList.add('big-btn-3');
    }
    </script>
    """, unsafe_allow_html=True)




# =====================================================
# YENİ KULLANICI EKRANI
# =====================================================
if st.session_state["mode"] == "new":

    st.markdown("### 🆕 Yeni Kullanıcı Kaydı")

    # Kullanıcı adı
    name = st.text_input(
        "Kullanıcı Adı / ID",
        placeholder="Örn: Ali Yılmaz"
    )

    # İmza yükleme
    files = st.file_uploader(
        "En az 3 imza görseli yükle (JPG / PNG):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    st.caption("📌 Not: Daha güçlü model performansı için 3 veya daha fazla imza önerilir.")

    # Alt butonlar
    col_a, col_b = st.columns([2, 1])

    # KAYDET butonu
    with col_a:
        if st.button("💾 Kaydet", key="btn_save", use_container_width=True):
            if not name or len(files) < 3:
                st.error("⚠️ Lütfen bir isim gir ve **en az 3 imza** yükle.")
            else:
                uid, users, created = get_or_create_user_id(users, name)
                emb_list = emb_db.get(uid, {}).get("embeddings", [])

                with st.spinner("📡 İmzalar işleniyor..."):
                    for f in files:
                        img = Image.open(f)
                        emb = get_embedding(img)
                        emb_list.append(emb)

                emb_db[uid] = {"embeddings": emb_list}
                save_users(users)
                save_embeddings(emb_db)

                if created:
                    st.success(f"🎉 Yeni kullanıcı oluşturuldu: **{name}** (ID: {uid})")
                else:
                    st.success(f"✨ {name} için yeni imzalar başarıyla eklendi.")

    # GERİ DÖN butonu
    with col_b:
        if st.button("⬅️ Geri Dön", key="btn_back_new", use_container_width=True):
            st.session_state["mode"] = None


# =====================================================
# KAYITLI KULLANICI DOĞRULAMA EKRANI
# =====================================================
elif st.session_state["mode"] == "existing":
    st.subheader("🖊️ Kayıtlı Kullanıcı Doğrulama")

    if not users:
        st.warning("Hiç kullanıcı yok. Önce 'Yeni Kullanıcı' ile kayıt oluştur.")
        if st.button("⬅️ Geri Dön", key="btn_back_empty"):
            st.session_state["mode"] = None
    else:
        selected_name = st.selectbox("Kullanıcı seç", [u["name"] for u in users])
        uid = [u["user_id"] for u in users if u["name"] == selected_name][0]

        file = st.file_uploader(
            "Test edilecek imza (tek resim):",
            type=["jpg", "jpeg", "png"],
        )

        threshold = st.slider(
            "Eşik değer (mesafe)",
            min_value=float(CFG["th_min"]),
            max_value=float(CFG["th_max"]),
            value=float(CFG["th_default"]),
            step=0.01,
        )

        col_a, col_b = st.columns([2, 1])

        with col_a:
            if file and st.button("🔍 Doğrula", key="btn_verify"):
                img = Image.open(file)
                test_emb = get_embedding(img)

                emb_list = emb_db.get(uid, {}).get("embeddings", [])
                if not emb_list:
                    st.error("Bu kullanıcı için kayıtlı imza embedding'i yok.")
                else:
                    d_min = compute_min_distance(test_emb, emb_list)
                    st.metric("Mesafe", f"{d_min:.3f}")

                    if d_min < threshold:
                        st.success("✅ Aynı kişi (gerçek imza)")
                    else:
                        st.error("❌ Sahte imza / farklı kişi")

        with col_b:
            if st.button("⬅️ Geri Dön", key="btn_back_existing"):
                st.session_state["mode"] = None

# =====================================================
# İKİ İMZAYI KARŞILAŞTIRMA (DB'siz)
# =====================================================
elif st.session_state["mode"] == "compare":
    st.subheader("🔁 İki İmzayı Karşılaştır")

    col_left, col_right = st.columns(2)

    with col_left:
        file1 = st.file_uploader(
            "İmza 1",
            type=["jpg", "jpeg", "png"],
            key="cmp1"
        )
    with col_right:
        file2 = st.file_uploader(
            "İmza 2",
            type=["jpg", "jpeg", "png"],
            key="cmp2"
        )

    threshold = st.slider(
        "Eşik değer (mesafe)",
        min_value=float(CFG["th_min"]),
        max_value=float(CFG["th_max"]),
        value=float(CFG["th_default"]),
        step=0.01,
    )

    col_a, col_b = st.columns([2, 1])

    with col_a:
        if file1 and file2 and st.button("🔍 Karşılaştır", key="btn_cmp"):
            img1 = Image.open(file1)
            img2 = Image.open(file2)

            emb1 = get_embedding(img1)
            emb2 = get_embedding(img2)

            dist = torch.norm(emb1 - emb2).item()
            st.metric("Mesafe", f"{dist:.3f}")

            if dist < threshold:
                st.success("✅ Aynı kişi olma ihtimali yüksek (benzer imzalar)")
            else:
                st.error("❌ Farklı kişi olma ihtimali yüksek (benzemeyen imzalar)")

    with col_b:
        if st.button("⬅️ Geri Dön", key="btn_back_cmp"):
            st.session_state["mode"] = None

# kart divini kapat

# Tüm küçük butonlara stil ver (büyükler hariç)
# Yeni kullanıcı ekranı sonunda:
st.markdown("""
<script>
const btns = window.parent.document.querySelectorAll('.stButton button');

// Ana ekrandaki 3 büyük buton dışındakilere small-btn ver
btns.forEach((b, idx) => {
  if (!b.classList.contains('big-btn-1') &&
      !b.classList.contains('big-btn-2') &&
      !b.classList.contains('big-btn-3')) {
    b.classList.add('small-btn');
  }
});
</script>
""", unsafe_allow_html=True)

