"""
PotatoScan — Potato Leaf Disease Classifier
A Streamlit app that loads a Keras CNN and diagnoses potato leaves
as Early Blight, Late Blight, or Healthy from an uploaded photo.
"""

import numpy as np
from PIL import Image
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL_PATH = "potato_disease_model.keras"
IMG_SIZE = (256, 256)  # model's expected input size

# IMPORTANT: This order must match the class order your model was trained on.
# For tf.keras image_dataset_from_directory the order is alphabetical:
#   Potato___Early_blight -> 0, Potato___Late_blight -> 1, Potato___healthy -> 2
# If your predictions look swapped, just reorder this list.
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Per-class metadata: status, accent color, plain-language guidance.
DISEASE_INFO = {
    "Early Blight": {
        "healthy": False,
        "color": "#E07A3F",
        "pathogen": "Alternaria solani (fungus)",
        "symptoms": "Dark brown spots with concentric rings (a target pattern), "
                    "usually starting on older, lower leaves with a yellow halo.",
        "action": "Remove affected leaves, avoid overhead watering, rotate crops, "
                  "and apply a recommended fungicide early if spread continues.",
    },
    "Late Blight": {
        "healthy": False,
        "color": "#C0392B",
        "pathogen": "Phytophthora infestans (water mold)",
        "symptoms": "Pale-green to brown water-soaked lesions that spread fast, "
                    "often with white fuzzy growth on the underside in humid weather.",
        "action": "Act quickly — this disease moves fast. Remove and destroy infected "
                  "plants, improve airflow, and apply protective fungicide promptly.",
    },
    "Healthy": {
        "healthy": True,
        "color": "#2D9A5B",
        "pathogen": "None detected",
        "symptoms": "Uniform green leaf surface with no necrotic spots, lesions, "
                    "or discoloration.",
        "action": "No treatment needed. Keep monitoring regularly and maintain good "
                  "field hygiene to prevent future infections.",
    },
}

# --------------------------------------------------------------------------- #
# Page setup + styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="PotatoScan — Leaf Disease Classifier",
    page_icon="🥔",
    layout="centered",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    /* ---- base ---- */
    .stApp {
        background:
            radial-gradient(1200px 500px at 15% -10%, #E7F0E6 0%, rgba(231,240,230,0) 60%),
            radial-gradient(1000px 600px at 110% 0%, #EAF3EE 0%, rgba(234,243,238,0) 55%),
            #F4F7F2;
    }
    .block-container { max-width: 760px; padding-top: 2.2rem; padding-bottom: 4rem; }

    /* hide default streamlit chrome */
    #MainMenu, header, footer { visibility: hidden; }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #1B2D24; }

    /* ---- hero ---- */
    .hero {
        position: relative;
        border-radius: 22px;
        padding: 34px 34px 30px;
        background: linear-gradient(135deg, #14402E 0%, #1E5B3F 55%, #2D6A4F 100%);
        overflow: hidden;
        box-shadow: 0 20px 45px -22px rgba(20,64,46,.55);
    }
    .hero::after {
        content: "";
        position: absolute; inset: 0;
        background-image:
            linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px),
            linear-gradient(180deg, rgba(255,255,255,.05) 1px, transparent 1px);
        background-size: 26px 26px;
        mask-image: radial-gradient(420px 220px at 85% 20%, #000 0%, transparent 70%);
        pointer-events: none;
    }
    .hero .eyebrow {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: .28em; text-transform: uppercase;
        font-size: .72rem; font-weight: 600; color: #9BE3B8; margin: 0 0 10px;
    }
    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.35rem; line-height: 1.05; font-weight: 700;
        color: #F5FBF6; margin: 0 0 12px; letter-spacing: -.01em;
    }
    .hero p { color: #C7E4D2; font-size: 1rem; max-width: 30rem; margin: 0; }
    .hero .leaf { position:absolute; right: 26px; top: 22px; opacity:.9; }

    /* ---- section labels ---- */
    .seclabel {
        font-family: 'Space Grotesk', sans-serif;
        font-size: .78rem; font-weight: 600; letter-spacing: .16em;
        text-transform: uppercase; color: #5C7568;
        margin: 30px 0 10px; display:flex; align-items:center; gap:10px;
    }
    .seclabel::before {
        content:""; width: 22px; height: 2px; background:#7DBE9B; border-radius:2px;
    }

    /* ---- uploader ---- */
    [data-testid="stFileUploaderDropzone"] {
        background: #FFFFFF;
        border: 1.5px dashed #BcD6C6;
        border-radius: 16px;
    }
    [data-testid="stFileUploaderDropzone"]:hover { border-color:#2D6A4F; }

    /* ---- diagnose button ---- */
    div.stButton > button {
        width: 100%;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600; font-size: 1.02rem; letter-spacing:.01em;
        border: none; border-radius: 14px; padding: .85rem 1rem;
        color: #F5FBF6;
        background: linear-gradient(135deg, #1E5B3F, #2D6A4F);
        box-shadow: 0 12px 24px -12px rgba(30,91,63,.7);
        transition: transform .12s ease, box-shadow .12s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 30px -12px rgba(30,91,63,.75);
        color:#fff;
    }
    div.stButton > button:active { transform: translateY(0); }

    /* ---- result card ---- */
    .result-card {
        background:#fff; border-radius: 20px; padding: 26px 28px;
        box-shadow: 0 18px 40px -24px rgba(20,64,46,.4);
        border: 1px solid #EAF0EB;
    }
    .status-pill {
        display:inline-flex; align-items:center; gap:8px;
        font-family:'Space Grotesk',sans-serif; font-weight:600;
        font-size:.78rem; letter-spacing:.06em; text-transform:uppercase;
        padding:7px 14px; border-radius:999px;
    }
    .dot { width:9px; height:9px; border-radius:50%; display:inline-block; }
    .diag-name {
        font-family:'Space Grotesk',sans-serif; font-weight:700;
        font-size:2rem; margin:16px 0 2px; letter-spacing:-.01em;
    }
    .diag-conf { color:#5C7568; font-size:.95rem; margin:0; }

    /* ---- confidence bars ---- */
    .bars { margin-top: 22px; }
    .bar-row { margin: 12px 0; }
    .bar-head {
        display:flex; justify-content:space-between;
        font-size:.86rem; font-weight:500; margin-bottom:6px; color:#324a3d;
    }
    .bar-track {
        height: 11px; background:#EEF3EF; border-radius:999px; overflow:hidden;
    }
    .bar-fill {
        height:100%; border-radius:999px; transform-origin:left;
        animation: grow .9s cubic-bezier(.22,1,.36,1) both;
    }
    @keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }

    /* ---- info card ---- */
    .info-card {
        background:#FBFDFB; border:1px solid #EAF0EB; border-left:4px solid #2D6A4F;
        border-radius:14px; padding:18px 20px; margin-top:18px;
    }
    .info-card h4 {
        font-family:'Space Grotesk',sans-serif; font-size:.82rem; font-weight:600;
        letter-spacing:.1em; text-transform:uppercase; color:#2D6A4F; margin:0 0 4px;
    }
    .info-card p { margin:0 0 14px; font-size:.94rem; color:#33473c; line-height:1.5; }
    .info-card p:last-child { margin-bottom:0; }

    .footnote { text-align:center; color:#8AA095; font-size:.82rem; margin-top:30px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Model loading (cached so it loads once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


def predict(image: Image.Image):
    """Resize, batch, and run inference. Model rescales 1/255 internally."""
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype="float32")      # raw [0,255]
    batch = np.expand_dims(arr, axis=0)         # (1, 256, 256, 3)
    model = load_model()
    probs = model.predict(batch, verbose=0)[0]
    return probs


# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero">
      <svg class="leaf" width="70" height="70" viewBox="0 0 24 24" fill="none"
           stroke="#9BE3B8" stroke-width="1.3" stroke-linecap="round">
        <path d="M5 21c0-9 5-16 14-18-1 12-7 17-14 18z"/>
        <path d="M5 21C8 15 12 10 18 6"/>
      </svg>
      <p class="eyebrow">Plant Pathology · CNN Diagnostics</p>
      <h1>PotatoScan</h1>
      <p>Upload a photo of a potato leaf and get an instant diagnosis —
         Early Blight, Late Blight, or Healthy — with a confidence breakdown.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #
st.markdown('<div class="seclabel">Upload a leaf image</div>', unsafe_allow_html=True)
file = st.file_uploader(
    "Drag and drop a leaf photo, or browse",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if file is not None:
    image = Image.open(file)
    st.image(image, caption="Selected leaf", use_container_width=True)

    diagnose = st.button("Diagnose leaf")

    if diagnose:
        with st.spinner("Analyzing leaf tissue…"):
            try:
                probs = predict(image)
            except Exception as e:  # noqa: BLE001
                st.error(
                    "Couldn't run the model. Make sure "
                    f"`{MODEL_PATH}` is in the same folder as this app.\n\n"
                    f"Details: {e}"
                )
                st.stop()

        top_idx = int(np.argmax(probs))
        top_label = CLASS_NAMES[top_idx]
        top_conf = float(probs[top_idx]) * 100
        info = DISEASE_INFO[top_label]
        is_healthy = info["healthy"]

        status_text = "No disease detected" if is_healthy else "Disease detected"
        status_color = "#2D9A5B" if is_healthy else "#C0392B"
        status_bg = "#E8F6EC" if is_healthy else "#FBEAE8"

        # ---- result card ----
        st.markdown('<div class="seclabel">Diagnosis</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="result-card">
              <span class="status-pill" style="background:{status_bg}; color:{status_color};">
                <span class="dot" style="background:{status_color};"></span>{status_text}
              </span>
              <div class="diag-name" style="color:{info['color']};">{top_label}</div>
              <p class="diag-conf">Model confidence · {top_conf:.1f}%</p>

              <div class="bars">
            """,
            unsafe_allow_html=True,
        )

        # ---- per-class confidence bars ----
        order = np.argsort(probs)[::-1]
        bars_html = ""
        for i in order:
            label = CLASS_NAMES[i]
            pct = float(probs[i]) * 100
            color = DISEASE_INFO[label]["color"]
            bars_html += f"""
              <div class="bar-row">
                <div class="bar-head"><span>{label}</span><span>{pct:.1f}%</span></div>
                <div class="bar-track">
                  <div class="bar-fill" style="width:{pct:.2f}%; background:{color};"></div>
                </div>
              </div>
            """
        st.markdown(bars_html + "</div></div>", unsafe_allow_html=True)

        # ---- guidance ----
        st.markdown(
            f"""
            <div class="info-card" style="border-left-color:{info['color']};">
              <h4>Likely cause</h4>
              <p>{info['pathogen']}</p>
              <h4>What it looks like</h4>
              <p>{info['symptoms']}</p>
              <h4>What to do</h4>
              <p>{info['action']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p class="footnote">This tool assists diagnosis and is not a substitute '
            'for professional agronomic advice.</p>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        '<p class="footnote">Supported formats: JPG, PNG, WEBP · '
        'For best results use a clear, well-lit photo of a single leaf.</p>',
        unsafe_allow_html=True,
    )
