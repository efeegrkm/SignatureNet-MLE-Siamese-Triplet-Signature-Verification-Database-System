import os
import sys
import json
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image, ImageOps
import streamlit as st

# =====================================================
# GENEL AYARLAR
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Kod tarafında varsayılan model (UI'da da default bu gelecek)
DEFAULT_MODEL_NAME = "Siamese"   # "Triplet" veya "Siamese"

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
        # SiameseModel klasörünün altında Scripts + models yapın var
        "code_dir": BASE_DIR / "SiameseModel" / "Scripts",
        "model_module": "model",
        "model_class": "SignatureNet",
        # Eğitimde kaydettiğin en iyi ağırlık:
        "weights": BASE_DIR / "SiameseModel" / "models" / "signature_siamese_best_0.92Accuracy.pth",
        "emb_path": BASE_DIR / "embeddings_siamese.pt",
        "th_min": 0.0,
        "th_max": 4.0,      # Siamese mesafelerin için biraz daha geniş aralık
        "th_default": 1.20, # Senin belirlediğin threshold ~1.2
    },
}

# =====================================================
# STREAMLIT TEMEL AYAR + UI MODEL SEÇİMİ
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
    color: #2ecc71;  /* yeşil */
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

/* ANA EKRANDAKİ BÜYÜK BUTONLAR */
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

/* 4. feature için yeni renk */
.stButton > button.big-btn-4 {
    width: 100%;
    background: linear-gradient(135deg,#00cec9,#0984e3);
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

# --- Model seçimini session_state ile yönet ---
if "model_name" not in st.session_state:
    st.session_state["model_name"] = DEFAULT_MODEL_NAME

model_choice = st.selectbox(
    "Kullanılacak modeli seç:",
    options=list(MODEL_CONFIG.keys()),
    index=list(MODEL_CONFIG.keys()).index(st.session_state["model_name"])
)

# Eğer seçim değişirse session_state güncelle
if model_choice != st.session_state["model_name"]:
    st.session_state["model_name"] = model_choice
    # Önceki cache'lenmiş modeli sil (farklı model için yeniden yüklenecek)
    model_key_old = f"model_Triplet"
    model_key_old2 = f"model_Siamese"
    if model_key_old in st.session_state:
        del st.session_state[model_key_old]
    if model_key_old2 in st.session_state:
        del st.session_state[model_key_old2]

ACTIVE_MODEL_NAME = st.session_state["model_name"]
CFG = MODEL_CONFIG[ACTIVE_MODEL_NAME]

st.markdown(
    f'<div class="subtitle">Aktif model: <b>{ACTIVE_MODEL_NAME}</b> – yeni kullanıcı ekle, kayıtlı kullanıcıyı doğrula veya iki imzayı karşılaştır</div>',
    unsafe_allow_html=True,
)

# =====================================================
# TRANSFORMLAR
# =====================================================

# Triplet modeli 128x224 ile eğitildi, aynen koruyoruz
TRIPLET_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Siamese için senin preprocess.py ile aynı mantığı PIL üzerinden uygulayalım
TARGET_SIZE = (400, 400)

def preprocess_for_siamese(img: Image.Image) -> torch.Tensor:
    """
    preprocess.py'deki preprocess_image(path) ile aynı mantık:
      - Griye çevir
      - Autocontrast
      - Oranı bozmadan 400x400 içine 'thumbnail'
      - 400x400 beyaz canvas ortasına yapıştır
      - ToTensor + Normalize
    """
    img = img.convert("L")
    img = ImageOps.autocontrast(img)

    # imzayı oranı bozulmadan 400x400 içine sığdır
    img.thumbnail(TARGET_SIZE, Image.LANCZOS)

    canvas = Image.new("L", TARGET_SIZE, 255)  # beyaz zemin
    offset_x = (TARGET_SIZE[0] - img.width) // 2
    offset_y = (TARGET_SIZE[1] - img.height) // 2
    canvas.paste(img, (offset_x, offset_y))

    t = transforms.ToTensor()(canvas)
    t = transforms.Normalize((0.5,), (0.5,))(t)
    return t

# Sadece önizleme için (tensor değil, PIL dönen versiyon)
def preprocess_for_siamese_preview(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img.thumbnail(TARGET_SIZE, Image.LANCZOS)
    canvas = Image.new("L", TARGET_SIZE, 255)
    offset_x = (TARGET_SIZE[0] - img.width) // 2
    offset_y = (TARGET_SIZE[1] - img.height) // 2
    canvas.paste(img, (offset_x, offset_y))
    return canvas

def preprocess_for_triplet_preview(img: Image.Image) -> Image.Image:
    # Triplet için sade: gri + 128x224 resize (normalize vs yok, sadece gösterim amaçlı)
    img = img.convert("L")
    img = img.resize((128, 224), Image.LANCZOS)
    return img

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
    """
    Her model için kendi .pt dosyasını yükler.
    Eski formatta sadece 'embeddings' varsa, mean_embedding'i geriye dönük hesaplayıp kaydeder.
    """
    if CFG["emb_path"].exists():
        emb_db = torch.load(CFG["emb_path"], map_location="cpu")
        changed = False

        for uid, rec in emb_db.items():
            if isinstance(rec, dict):
                embs = rec.get("embeddings", None)
                mean_emb = rec.get("mean_embedding", None)

                if mean_emb is None and embs is not None and len(embs) > 0:
                    # Eski kayıt -> mean_embedding'i hesapla
                    try:
                        stack = torch.stack(embs)
                        rec["mean_embedding"] = stack.mean(dim=0)
                        changed = True
                    except Exception:
                        # herhangi bir stacking hatasında sessiz geç
                        pass

        if changed:
            torch.save(emb_db, CFG["emb_path"])

        return emb_db
    return {}

def save_embeddings(emb_db):
    torch.save(emb_db, CFG["emb_path"])

def load_model():
    # Modeli model_Triplet / model_Siamese şeklinde cache'liyoruz
    model_key = f"model_{ACTIVE_MODEL_NAME}"
    if model_key in st.session_state:
        return st.session_state[model_key]

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

    st.session_state[model_key] = model
    return model

def get_embedding(img: Image.Image) -> torch.Tensor:
    """
    Aktif modele göre doğru preprocessing+transform'u uygula.
    Triplet: 128x224 transform
    Siamese: 400x400 canvas preprocess_for_siamese
    """
    model = load_model()

    if ACTIVE_MODEL_NAME == "Triplet":
        # Triplet için eski pipeline
        img = img.convert("L")
        t = TRIPLET_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    else:
        # Siamese için 400x400 canvas preprocessing
        t = preprocess_for_siamese(img).unsqueeze(0).to(DEVICE)

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

def clear_database_files():
    """
    Kullanıcı DB'sini sıfırlar:
      - users.json
      - embeddings_triplet.pt
      - embeddings_siamese.pt
    """
    paths = [
        USERS_PATH,
        MODEL_CONFIG["Triplet"]["emb_path"],
        MODEL_CONFIG["Siamese"]["emb_path"],
    ]
    for p in paths:
        try:
            p = Path(p)
            if p.exists():
                p.unlink()
        except Exception:
            pass

# =====================================================
# STATE
# =====================================================
if "mode" not in st.session_state:
    st.session_state["mode"] = None

if "confirm_clear_db" not in st.session_state:
    st.session_state["confirm_clear_db"] = False

users = load_users()
emb_db = load_embeddings()
id2name = {u["user_id"]: u["name"] for u in users}

# =====================================================
# ANA EKRAN: 4 FEATURE + DB SIFIRLA
# =====================================================
if st.session_state["mode"] is None:

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🆕 Yeni Kullanıcı", key="btn_new", use_container_width=True):
            st.session_state["mode"] = "new"

    with col2:
        if st.button("🖊️ Kayıtlı Kullanıcı", key="btn_existing", use_container_width=True):
            st.session_state["mode"] = "existing"

    with col3:
        if st.button("🔁 İki İmzayı Karşılaştır", key="btn_compare", use_container_width=True):
            st.session_state["mode"] = "compare"

    with col4:
        # YENİ: Tek imzadan kullanıcı bulma
        if st.button("🔎 İmzadan Kullanıcı Bul", key="btn_identify", use_container_width=True):
            st.session_state["mode"] = "identify"

    # BUTONLARA RENK SINIFI EKLEYEN SCRIPT (ilk 4 butona)
    st.markdown("""
    <script>
    const btns = window.parent.document.querySelectorAll('.stButton button');
    if (btns.length >= 4) {
        btns[0].classList.add('big-btn-1');
        btns[1].classList.add('big-btn-2');
        btns[2].classList.add('big-btn-3');
        btns[3].classList.add('big-btn-4');
    }
    </script>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # YENİ: Database'i sıfırlama butonu (onaylı)
    st.markdown("### 🗑️ Database Yönetimi")
    if not st.session_state["confirm_clear_db"]:
        if st.button("🗑️ Database Verilerini Sil (Sıfırla)", key="btn_clear_db", use_container_width=True):
            st.session_state["confirm_clear_db"] = True
    else:
        st.warning("Bu işlem tüm kullanıcıları ve embedding kayıtlarını silecektir. Emin misin?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Evet, hepsini sil", key="btn_clear_db_confirm", use_container_width=True):
                clear_database_files()
                st.session_state["confirm_clear_db"] = False
                st.session_state["mode"] = None
                # Sayfayı yenile
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
        with c2:
            if st.button("❌ Vazgeç", key="btn_clear_db_cancel", use_container_width=True):
                st.session_state["confirm_clear_db"] = False

# =====================================================
# YENİ KULLANICI EKRANI
# =====================================================
if st.session_state["mode"] == "new":

    st.markdown("### 🆕 Yeni Kullanıcı Kaydı")

    name = st.text_input(
        "Kullanıcı Adı / ID",
        placeholder="Örn: Ali Yılmaz"
    )

    files = st.file_uploader(
        "En az 3 imza görseli yükle (JPG / PNG):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    st.caption("📌 Not: Daha güçlü model performansı için 3 veya daha fazla imza önerilir.")

    col_a, col_b = st.columns([2, 1])

    with col_a:
        if st.button("💾 Kaydet", key="btn_save", use_container_width=True):
            if not name or len(files) < 3:
                st.error("⚠️ Lütfen bir isim gir ve **en az 3 imza** yükle.")
            else:
                uid, users, created = get_or_create_user_id(users, name)

                # Var olan kayıt varsa önce onu çek
                rec = emb_db.get(uid, {})
                emb_list = rec.get("embeddings", [])

                with st.spinner("📡 İmzalar işleniyor..."):
                    for f in files:
                        img = Image.open(f)
                        emb = get_embedding(img)
                        emb_list.append(emb)

                # Mean embedding hesapla
                if len(emb_list) > 0:
                    stack = torch.stack(emb_list)
                    mean_emb = stack.mean(dim=0)
                else:
                    mean_emb = None

                emb_db[uid] = {
                    "embeddings": emb_list,
                    "mean_embedding": mean_emb,
                }

                save_users(users)
                save_embeddings(emb_db)

                if created:
                    st.success(f"🎉 Yeni kullanıcı oluşturuldu: **{name}** (ID: {uid})")
                else:
                    st.success(f"✨ {name} için yeni imzalar başarıyla eklendi.")

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

                rec = emb_db.get(uid, {})
                emb_list = rec.get("embeddings", [])
                mean_emb = rec.get("mean_embedding", None)

                if not emb_list:
                    st.error("Bu kullanıcı için kayıtlı imza embedding'i yok.")
                else:
                    # Geriye dönük: mean_embedding yoksa hesapla ve kaydet
                    if mean_emb is None:
                        stack = torch.stack(emb_list)
                        mean_emb = stack.mean(dim=0)
                        rec["mean_embedding"] = mean_emb
                        emb_db[uid] = rec
                        save_embeddings(emb_db)

                    d_min = torch.norm(test_emb - mean_emb).item()
                    st.metric("Mesafe", f"{d_min:.3f}")

                    if d_min < threshold:
                        st.success("✅ Aynı kişi (gerçek imza)")
                    else:
                        st.error("❌ Sahte imza / farklı kişi")

        with col_b:
            if st.button("⬅️ Geri Dön", key="btn_back_existing", use_container_width=True):
                st.session_state["mode"] = None

# =====================================================
# YENİ FEATURE: TEK İMZADAN KULLANICI BUL (MEAN EMBEDDING)
# =====================================================
elif st.session_state["mode"] == "identify":
    st.subheader("🔎 İmzadan Kullanıcı Bul (Database Search)")

    if not users:
        st.warning("Hiç kullanıcı yok. Önce 'Yeni Kullanıcı' ile kayıt oluştur.")
        if st.button("⬅️ Geri Dön", key="btn_back_identify_empty", use_container_width=True):
            st.session_state["mode"] = None
    else:
        file = st.file_uploader(
            "Kime ait olduğunu bulmak istediğin imza (tek resim):",
            type=["jpg", "jpeg", "png"],
            key="identify_file"
        )

        # Önizleme: Orijinal + modele giden görüntü (senin compare ekranındaki mantıkla)
        if file:
            st.markdown("#### 📷 Yüklenen İmzanın Önizlemesi")
            img_orig = Image.open(file)
            c1, c2 = st.columns(2)
            with c1:
                st.image(img_orig, caption="Orijinal", use_column_width=True)
            with c2:
                if ACTIVE_MODEL_NAME == "Siamese":
                    img_proc = preprocess_for_siamese_preview(img_orig)
                    st.image(img_proc, caption="Model Girişi (400×400)", use_column_width=True)
                else:
                    img_proc = preprocess_for_triplet_preview(img_orig)
                    st.image(img_proc, caption="Model Girişi (128×224)", use_column_width=True)

        threshold = st.slider(
            "Eşik değer (mesafe)",
            min_value=float(CFG["th_min"]),
            max_value=float(CFG["th_max"]),
            value=float(CFG["th_default"]),
            step=0.01,
            key="identify_threshold"
        )

        col_a, col_b = st.columns([2, 1])

        with col_a:
            if file and st.button("🔍 Kullanıcıyı Bul", key="btn_identify_run", use_container_width=True):
                img = Image.open(file)
                test_emb = get_embedding(img)

                # Tüm kullanıcıların mean embedding'leri ile karşılaştır
                candidates = []
                changed = False

                for uid, rec in emb_db.items():
                    if not isinstance(rec, dict):
                        continue

                    mean_emb = rec.get("mean_embedding", None)
                    if mean_emb is None:
                        embs = rec.get("embeddings", [])
                        if embs:
                            try:
                                mean_emb = torch.stack(embs).mean(dim=0)
                                rec["mean_embedding"] = mean_emb
                                emb_db[uid] = rec
                                changed = True
                            except Exception:
                                continue

                    if mean_emb is None:
                        continue

                    dist = torch.norm(test_emb - mean_emb).item()
                    candidates.append((dist, uid))

                if changed:
                    save_embeddings(emb_db)

                if not candidates:
                    st.error("Database'de karşılaştırılabilir embedding kaydı bulunamadı.")
                else:
                    candidates.sort(key=lambda x: x[0])
                    best_dist, best_uid = candidates[0]
                    best_name = id2name.get(best_uid, best_uid)

                    st.metric("En Yakın Mesafe", f"{best_dist:.3f}")

                    if best_dist < threshold:
                        st.success(f"✅ Bu imza muhtemelen **{best_name}** (ID: {best_uid}) kullanıcısına ait.")
                    else:
                        st.warning("⚠️ Bu imza hiçbir kullanıcıyla eşik altında eşleşmedi (No matching user).")

                    # İstersen top-5 listeyi de göster (debug için faydalı)
                    top_k = candidates[:5]
                    rows = []
                    for d, uid in top_k:
                        rows.append({
                            "User": id2name.get(uid, uid),
                            "UserID": uid,
                            "Distance": round(d, 4)
                        })
                    st.markdown("#### 🔝 En Yakın 5 Aday")
                    st.dataframe(rows, use_container_width=True)

        with col_b:
            if st.button("⬅️ Geri Dön", key="btn_back_identify", use_container_width=True):
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

    # --- Seçilen imzaların önizlemesi (orijinal + model girişi) ---
    if file1 or file2:
        st.markdown("#### 📷 Seçilen İmzaların Önizlemesi")
        prev_col1, prev_col2 = st.columns(2)

        with prev_col1:
            if file1:
                img1_orig = Image.open(file1)
                st.image(img1_orig, caption="İmza 1 - Orijinal", use_column_width=True)

                if ACTIVE_MODEL_NAME == "Siamese":
                    img1_proc = preprocess_for_siamese_preview(img1_orig)
                    st.image(img1_proc, caption="İmza 1 - Model Girişi (400×400)", use_column_width=True)
                else:
                    img1_proc = preprocess_for_triplet_preview(img1_orig)
                    st.image(img1_proc, caption="İmza 1 - Model Girişi (128×224)", use_column_width=True)

        with prev_col2:
            if file2:
                img2_orig = Image.open(file2)
                st.image(img2_orig, caption="İmza 2 - Orijinal", use_column_width=True)

                if ACTIVE_MODEL_NAME == "Siamese":
                    img2_proc = preprocess_for_siamese_preview(img2_orig)
                    st.image(img2_proc, caption="İmza 2 - Model Girişi (400×400)", use_column_width=True)
                else:
                    img2_proc = preprocess_for_triplet_preview(img2_orig)
                    st.image(img2_proc, caption="İmza 2 - Model Girişi (128×224)", use_column_width=True)

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
        if st.button("⬅️ Geri Dön", key="btn_back_cmp", use_container_width=True):
            st.session_state["mode"] = None

# =====================================================
# BUTON STİL SCRIPTİ
# =====================================================
st.markdown("""
<script>
const btns = window.parent.document.querySelectorAll('.stButton button');

// Ana ekrandaki büyük butonlar haricindekilere small-btn ver
btns.forEach((b, idx) => {
  if (!b.classList.contains('big-btn-1') &&
      !b.classList.contains('big-btn-2') &&
      !b.classList.contains('big-btn-3') &&
      !b.classList.contains('big-btn-4')) {
    b.classList.add('small-btn');
  }
});
</script>
""", unsafe_allow_html=True)
