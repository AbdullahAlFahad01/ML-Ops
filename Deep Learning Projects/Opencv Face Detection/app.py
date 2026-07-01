import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="FaceScope AI — Face Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS — dark "viewfinder" theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg:        #0d1117;
    --panel:     #161b26;
    --border:    #2a3244;
    --amber:     #f5a623;
    --amber-dim: rgba(245, 166, 35, 0.12);
    --text:      #e6e9f0;
    --muted:     #8b94a7;
    --green:     #3ddc84;
}

.stApp {
    background: linear-gradient(180deg, #0d1117 0%, #10151f 100%);
    font-family: 'Space Grotesk', sans-serif;
}

/* ---------- Hero header ---------- */
.hero {
    position: relative;
    padding: 2.2rem 2.4rem;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    margin-bottom: 1.6rem;
    overflow: hidden;
}
.hero::before, .hero::after {
    content: '';
    position: absolute;
    width: 26px; height: 26px;
    border: 2px solid var(--amber);
}
.hero::before { top: 12px; left: 12px; border-right: none; border-bottom: none; }
.hero::after  { bottom: 12px; right: 12px; border-left: none; border-top: none; }

.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 0.5rem;
}
.hero-title {
    font-size: 2.3rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.1;
    margin: 0;
}
.hero-title span { color: var(--amber); }
.hero-sub {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.6rem;
    max-width: 560px;
}

/* ---------- Metric cards ---------- */
.metric-row { display: flex; gap: 1rem; margin: 1.2rem 0; }
.metric-card {
    flex: 1;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
}
.metric-card.accent { border-color: var(--amber); background: var(--amber-dim); }
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text);
    margin-top: 0.2rem;
}
.metric-card.accent .metric-value { color: var(--amber); }

/* ---------- Status pill ---------- */
.status-pill {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    border: 1px solid var(--green);
    color: var(--green);
    background: rgba(61, 220, 132, 0.08);
    margin-bottom: 0.8rem;
}
.status-pill.none {
    border-color: #e05555;
    color: #e05555;
    background: rgba(224, 85, 85, 0.08);
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 6px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    color: var(--muted);
    border-radius: 8px;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: var(--amber-dim) !important;
    color: var(--amber) !important;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--amber);
}

/* ---------- Uploader & camera ---------- */
[data-testid="stFileUploader"] {
    background: var(--panel);
    border: 1px dashed var(--border);
    border-radius: 12px;
    padding: 0.8rem;
}

/* ---------- Buttons ---------- */
.stDownloadButton button, .stButton button {
    background: var(--amber);
    color: #10151f;
    font-weight: 700;
    border: none;
    border-radius: 8px;
}
.stDownloadButton button:hover, .stButton button:hover {
    background: #ffbe45;
    color: #10151f;
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border);
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Cached model loader
# ============================================================
@st.cache_resource
def load_cascades():
    face = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )
    return face, eye


face_cascade, eye_cascade = load_cascades()


# ============================================================
# Detection logic
# ============================================================
def detect_faces(img_array, scale_factor, min_neighbors, min_size,
                 detect_eyes=False, box_color=(245, 166, 35)):
    """Detect faces (and optionally eyes) and draw viewfinder boxes."""
    output = img_array.copy()
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)  # better results in uneven lighting

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
    )

    for i, (x, y, w, h) in enumerate(faces, start=1):
        # Corner-bracket "viewfinder" box instead of a plain rectangle
        L = max(int(w * 0.18), 10)
        t = max(int(w * 0.02), 2)
        c = box_color
        for (px, py, dx1, dy1, dx2, dy2) in [
            (x, y, L, 0, 0, L),                     # top-left
            (x + w, y, -L, 0, 0, L),                # top-right
            (x, y + h, L, 0, 0, -L),                # bottom-left
            (x + w, y + h, -L, 0, 0, -L),           # bottom-right
        ]:
            cv2.line(output, (px, py), (px + dx1, py + dy1), c, t)
            cv2.line(output, (px, py), (px + dx2, py + dy2), c, t)

        # Label
        label = f"FACE {i:02d}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(output, (x, y - th - 12), (x + tw + 10, y - 2), c, -1)
        cv2.putText(output, label, (x + 5, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (13, 17, 23), 2)

        # Optional eye detection inside each face region
        if detect_eyes:
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
            for (ex, ey, ew, eh) in eyes:
                cv2.circle(
                    output,
                    (x + ex + ew // 2, y + ey + eh // 2),
                    max(ew // 3, 6), (61, 220, 132), 2,
                )

    return output, len(faces)


def to_png_bytes(img_array):
    buf = BytesIO()
    Image.fromarray(img_array).save(buf, format="PNG")
    return buf.getvalue()


def render_metrics(total_faces, img_shape):
    h, w = img_shape[:2]
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card accent">
            <div class="metric-label">Faces detected</div>
            <div class="metric-value">{total_faces}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Resolution</div>
            <div class="metric-value">{w}×{h}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value" style="font-size:1.1rem; padding-top:0.5rem;">Haar Cascade</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_status(total_faces):
    if total_faces > 0:
        st.markdown(
            f'<div class="status-pill">● DETECTION COMPLETE — {total_faces} FACE(S) FOUND</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill none">● NO FACES DETECTED — TRY ADJUSTING SENSITIVITY</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# Sidebar — detection settings
# ============================================================
with st.sidebar:
    st.markdown("### ⚙ Detection Settings")
    scale_factor = st.slider(
        "Scale factor", 1.05, 1.5, 1.3, 0.05,
        help="Lower = more thorough but slower. 1.1–1.3 works for most images.",
    )
    min_neighbors = st.slider(
        "Min neighbors", 1, 10, 5,
        help="Higher = fewer false positives, but may miss faces.",
    )
    min_size = st.slider(
        "Min face size (px)", 20, 200, 30, 10,
        help="Ignore faces smaller than this.",
    )
    detect_eyes = st.toggle("Detect eyes", value=False)

    st.markdown("---")
    st.markdown("### ℹ About")
    st.caption(
        "FaceScope AI uses OpenCV Haar Cascade classifiers to locate "
        "frontal faces in images. All processing happens on the server — "
        "no images are stored."
    )


# ============================================================
# Hero header
# ============================================================
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Computer Vision · OpenCV · Streamlit</div>
    <p class="hero-title">FaceScope <span>AI</span></p>
    <p class="hero-sub">Professional face detection from your camera or an uploaded
    image. Tune the detector in the sidebar, then download the annotated result.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Tabs: Upload | Camera
# ============================================================
tab_upload, tab_camera = st.tabs(["📁  UPLOAD IMAGE", "📷  LIVE CAMERA"])

# ---------- Upload tab ----------
with tab_upload:
    uploaded_image = st.file_uploader(
        "Drop an image here (JPG / JPEG / PNG / WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_image is not None:
        img = Image.open(uploaded_image).convert("RGB")
        img_array = np.array(img)

        with st.spinner("Scanning for faces…"):
            detected_img, total_faces = detect_faces(
                img_array, scale_factor, min_neighbors, min_size, detect_eyes
            )

        render_status(total_faces)
        render_metrics(total_faces, img_array.shape)

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown('<div class="metric-label" style="margin-bottom:6px;">ORIGINAL</div>',
                        unsafe_allow_html=True)
            st.image(img_array, use_container_width=True)
        with col2:
            st.markdown('<div class="metric-label" style="margin-bottom:6px;">DETECTED</div>',
                        unsafe_allow_html=True)
            st.image(detected_img, use_container_width=True)

        st.download_button(
            "⬇ Download annotated image",
            data=to_png_bytes(detected_img),
            file_name="facescope_result.png",
            mime="image/png",
        )
    else:
        st.info("Upload an image to begin detection.")

# ---------- Camera tab ----------
with tab_camera:
    st.caption(
        "Uses your browser's camera — works locally **and** on Streamlit "
        "Community Cloud. Allow camera access when prompted."
    )
    camera_photo = st.camera_input("Take a photo")

    if camera_photo is not None:
        img = Image.open(camera_photo).convert("RGB")
        img_array = np.array(img)

        with st.spinner("Scanning for faces…"):
            detected_img, total_faces = detect_faces(
                img_array, scale_factor, min_neighbors, min_size, detect_eyes
            )

        render_status(total_faces)
        render_metrics(total_faces, img_array.shape)

        st.image(detected_img, use_container_width=True)

        st.download_button(
            "⬇ Download annotated image",
            data=to_png_bytes(detected_img),
            file_name="facescope_camera_result.png",
            mime="image/png",
        )


# ============================================================
# Footer
# ============================================================
st.markdown(
    '<div class="footer">FACESCOPE AI · OPENCV HAAR CASCADE · BUILT WITH STREAMLIT</div>',
    unsafe_allow_html=True,
)
