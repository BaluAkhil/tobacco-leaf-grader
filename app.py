import streamlit as st
import cv2
import numpy as np
import joblib
import os
import glob
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

st.set_page_config(
    page_title="Tobacco Leaf Grader",
    page_icon="🌿",
    layout="centered"
)

MODEL_PATH   = "tobacco_grader.pkl"
DATASET_PATH = r"C:\Users\BALU LOHITH REDDY\Downloads\archive\tobacco leaves"

# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container { padding-top: 1.8rem; padding-bottom: 2rem; max-width: 700px; }

html, body, [class*="css"] { font-family: 'Segoe UI', Arial, sans-serif; }

/* ── Sidebar ─────────────────────────── */
[data-testid="stSidebar"] { background: #1c3d20; }
[data-testid="stSidebar"] * { color: #dceedd !important; }
[data-testid="stSidebar"] hr { border-color: #2e5c34 !important; }

/* ── Mode selector row ───────────────── */
div[data-testid="stHorizontalBlock"] .stButton button {
    border-radius: 6px !important;
    border: 1px solid #ccc !important;
    background: #fff !important;
    color: #333 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 0.45rem 0.6rem !important;
}
div[data-testid="stHorizontalBlock"] .stButton button:hover {
    border-color: #2e7d32 !important;
    color: #2e7d32 !important;
    background: #f4fbf4 !important;
}

/* ── Primary button ─────────────────── */
.stButton > button[kind="primary"] {
    background: #2e7d32 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 7px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.2rem !important;
    width: 100%;
}
.stButton > button[kind="primary"]:hover { background: #1b5e20 !important; }

/* ── Result card ─────────────────────── */
.result-card {
    border-radius: 10px;
    padding: 20px 18px;
    margin: 14px 0 8px;
    text-align: center;
}
.card-a { background: #fff8e1; border: 1.5px solid #f9a825; }
.card-b { background: #e8f5e9; border: 1.5px solid #43a047; }
.card-c { background: #fbe9e7; border: 1.5px solid #e53935; }
.grade-name { font-size: 24px; font-weight: 700; margin: 2px 0; }
.grade-type { font-size: 13px; color: #666; }
.card-a .grade-name { color: #e65100; }
.card-b .grade-name { color: #1b5e20; }
.card-c .grade-name { color: #b71c1c; }

/* ── Mode badge ──────────────────────── */
.mode-badge {
    display: inline-block;
    background: #e8f5e9;
    color: #2e7d32;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 4px;
    border: 1px solid #c8e6c9;
    margin-bottom: 10px;
}

/* ── Section labels ──────────────────── */
.sec-label {
    font-size: 11px;
    font-weight: 600;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin: 14px 0 5px;
}

/* ── Tip box ─────────────────────────── */
.tip-box {
    background: #f5f5f5;
    border-left: 3px solid #2e7d32;
    border-radius: 0 6px 6px 0;
    padding: 9px 12px;
    font-size: 13px;
    color: #444;
    margin-top: 10px;
}

/* ── Upload area ─────────────────────── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #ccc !important;
    border-radius: 8px !important;
    background: #fafafa !important;
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION
#  *** Must match exactly what was used in train_model.py ***
# ─────────────────────────────────────────────────────────────

def remove_background(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    bg  = cv2.inRange(hsv, np.array([0, 0, 190]), np.array([180, 40, 255]))
    k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bg  = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, k)
    bg  = cv2.morphologyEx(bg, cv2.MORPH_OPEN,  k)
    mask = cv2.bitwise_not(bg)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.uint8(labels == largest) * 255
    return cv2.bitwise_and(img_bgr, img_bgr, mask=mask), mask


def color_features(img_bgr, mask):
    feats    = []
    has_mask = mask is not None and np.any(mask > 0)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = (hsv[mask > 0, 0] if has_mask else hsv[:, :, 0].flatten()).astype(np.float32)
    s = (hsv[mask > 0, 1] if has_mask else hsv[:, :, 1].flatten()).astype(np.float32)
    v = (hsv[mask > 0, 2] if has_mask else hsv[:, :, 2].flatten()).astype(np.float32)

    h_hist, _ = np.histogram(h, bins=18, range=(0, 180), density=True)
    s_hist, _ = np.histogram(s, bins=8,  range=(0, 255), density=True)
    v_hist, _ = np.histogram(v, bins=8,  range=(0, 255), density=True)
    feats.extend(h_hist); feats.extend(s_hist); feats.extend(v_hist)   # 34

    feats += [np.mean(h), np.std(h), np.median(h)]  # 3
    feats += [np.mean(s), np.std(s)]                 # 2
    feats += [np.mean(v), np.std(v)]                 # 2

    for p in [10, 25, 50, 75, 90]:                   # 5
        feats.append(float(np.percentile(v, p)) / 255.0)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab_px = lab[mask > 0] if has_mask else lab.reshape(-1, 3)
    for ch in range(3):                               # 6
        feats.append(float(np.mean(lab_px[:, ch])))
        feats.append(float(np.std(lab_px[:, ch])))

    px = (img_bgr[mask > 0] if has_mask else img_bgr.reshape(-1, 3)).astype(float)
    b_ch, g_ch, r_ch = px[:, 0], px[:, 1], px[:, 2]
    total = np.mean(r_ch) + np.mean(g_ch) + np.mean(b_ch) + 1e-6
    feats += [np.mean(r_ch)/total, np.mean(g_ch)/total, np.mean(b_ch)/total]  # 3
    feats += [np.mean(r_ch) - np.mean(b_ch), np.mean(g_ch) - np.mean(b_ch)]  # 2

    return np.array(feats, dtype=np.float32)   # 57 total


def texture_features(img_bgr, mask):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))
    feats = []

    glcm = graycomatrix(gray, distances=[1, 3, 5],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)

    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        vals = graycoprops(glcm, prop).flatten()      # 12 values each
        feats += [float(np.mean(vals)), float(np.std(vals)),
                  float(np.min(vals)),  float(np.max(vals))]   # 16 per prop = 64

    eps     = 1e-10
    entropy = -np.sum(glcm * np.log2(glcm + eps), axis=(0, 1)).flatten()
    feats  += [float(np.mean(entropy)), float(np.std(entropy))]   # 2

    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gm = np.sqrt(sx**2 + sy**2).flatten()
    feats += [float(np.mean(gm)), float(np.std(gm)), float(np.percentile(gm, 75))]  # 3

    feats.append(float(np.var(cv2.Laplacian(gray, cv2.CV_64F))))  # 1

    return np.array(feats, dtype=np.float32)   # 70 total


def get_features(img_bgr):
    """
    Extract features — MUST match train_model.py exactly.
    Total = color(57) + texture(70) = 127 features.

    If you get a feature mismatch error, retrain your model using
    train_model.py (not the notebook) so both use the same code.
    """
    img = cv2.resize(img_bgr, (224, 224))
    img_m, mask = remove_background(img)
    if np.sum(mask > 0) < 224 * 224 * 0.05:
        img_m, mask = img, None
    cf = color_features(img_m, mask)
    tf = texture_features(img_m, mask)
    return np.concatenate([cf, tf])


@st.cache_resource
def load_model(path):
    return joblib.load(path) if os.path.exists(path) else None


# ─────────────────────────────────────────────────────────────
GRADES = {
    0: ("Grade A", "Gold",            "card-a", "#f9a825"),
    1: ("Grade B", "Yellowish Green", "card-b", "#43a047"),
    2: ("Grade C", "Dark Brown",      "card-c", "#e53935"),
}
TIPS = {
    0: "Premium quality — suitable for high-value tobacco products.",
    1: "Medium quality — acceptable for standard blends.",
    2: "Lower grade — may be sold at a reduced price.",
}

model_data = load_model(MODEL_PATH)


# ─────────────────────────────────────────────────────────────
#  SIDEBAR  — kept minimal, no AI-looking clutter
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    # Simple text-based header — no excessive SVG or emoji
    st.markdown("""
    <div style="padding:16px 4px 12px">
        <div style="font-size:18px;font-weight:700;color:#e8f5e9">
            Tobacco Grader
        </div>
        <div style="font-size:12px;color:#81c784;margin-top:3px">
            Leaf quality classifier
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Grade reference only — clean table style
    st.markdown(
        "<div style='font-size:12px;font-weight:600;color:#a5d6a7;"
        "text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px'>"
        "Grade reference</div>",
        unsafe_allow_html=True
    )

    grade_rows = [
        ("#f9a825", "Grade A", "Gold leaf"),
        ("#43a047", "Grade B", "Yellowish green"),
        ("#e53935", "Grade C", "Dark brown"),
    ]
    for col, lbl, desc in grade_rows:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;
                    padding:7px 0;border-bottom:1px solid #2e5c34">
            <div style="width:10px;height:10px;border-radius:2px;
                        background:{col};flex-shrink:0"></div>
            <div>
                <div style="font-size:13px;font-weight:600;color:#e8f5e9">{lbl}</div>
                <div style="font-size:11px;color:#81c784">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Single practical note
    st.markdown(
        "<div style='font-size:11px;color:#81c784;line-height:1.6'>"
        "For best results, place the leaf flat on white paper before scanning."
        "</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

st.markdown("## Tobacco Leaf Grader")
st.markdown(
    "<p style='color:#888;font-size:13px;margin-top:-10px;margin-bottom:18px'>"
    "Upload, photograph, or scan a tobacco leaf to identify its grade."
    "</p>",
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["Classify", "Batch", "History"])


# ═══════════════════════
#  TAB 1 — Classify
# ═══════════════════════
with tab1:

    # Session state init
    for k, v in [("mode","upload"),("img_pil",None),
                 ("pred",None),("proba",None),("last_fname",None)]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ── 3 mode buttons — plain text, no emoji overload ────────
    st.markdown(
        "<div class='sec-label'>Select input method</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Upload file", use_container_width=True):
            st.session_state.update(mode="upload", img_pil=None, pred=None, proba=None)
    with c2:
        if st.button("Take photo", use_container_width=True):
            st.session_state.update(mode="camera", img_pil=None, pred=None, proba=None)
    with c3:
        if st.button("Scan leaf", use_container_width=True):
            st.session_state.update(mode="scan", img_pil=None, pred=None, proba=None)

    # Active mode badge
    badge_map = {"upload": "Upload file", "camera": "Take photo", "scan": "Scan leaf"}
    st.markdown(
        f"<div class='mode-badge'>{badge_map[st.session_state.mode]}</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ── Input widget ──────────────────────────────────────────
    img_pil = None
    mode    = st.session_state.mode

    if mode == "upload":
        st.markdown("<div class='sec-label'>Upload image</div>", unsafe_allow_html=True)
        f = st.file_uploader(
            "Choose file",
            type=["jpg","jpeg","png"],
            label_visibility="collapsed",
            key="file_up"
        )
        if f:
            if st.session_state.last_fname != f.name:
                st.session_state.pred = None
                st.session_state.proba = None
                st.session_state.last_fname = f.name
            img_pil = Image.open(f)

    elif mode == "camera":
        st.markdown("<div class='sec-label'>Camera</div>", unsafe_allow_html=True)
        st.caption("Point at the leaf and press the capture button.")
        cam = st.camera_input("Camera", label_visibility="collapsed", key="cam_cap")
        if cam:
            img_pil = Image.open(cam)
            if st.session_state.last_fname != "cam":
                st.session_state.pred = None
                st.session_state.proba = None
                st.session_state.last_fname = "cam"

    elif mode == "scan":
        st.markdown("<div class='sec-label'>Scan</div>", unsafe_allow_html=True)
        st.caption("Place the leaf flat on white paper. Hold the camera directly above and capture.")
        sc = st.camera_input("Scan", label_visibility="collapsed", key="scan_cap")
        if sc:
            img_pil = Image.open(sc)
            if st.session_state.last_fname != "scan":
                st.session_state.pred = None
                st.session_state.proba = None
                st.session_state.last_fname = "scan"

    if img_pil:
        st.session_state.img_pil = img_pil

    # ── Preview + predict ─────────────────────────────────────
    if st.session_state.img_pil:
        pil = st.session_state.img_pil
        st.markdown("<div class='sec-label'>Preview</div>", unsafe_allow_html=True)
        st.image(pil, use_container_width=True)

        btn_col, clr_col = st.columns([4, 1])
        with btn_col:
            go = st.button("Predict grade", type="primary", use_container_width=True)
        with clr_col:
            if st.button("Clear", use_container_width=True):
                st.session_state.update(img_pil=None, pred=None, proba=None, last_fname=None)
                st.rerun()

        if go:
            if not model_data:
                st.error("Model not loaded. Place tobacco_grader.pkl in the same folder as this file and restart.")
            else:
                with st.spinner("Analysing..."):
                    try:
                        arr     = np.array(pil.convert("RGB"))
                        bgr     = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                        feats   = get_features(bgr).reshape(1, -1)
                        pred    = int(model_data["model"].predict(feats)[0])
                        proba   = model_data["model"].predict_proba(feats)[0].tolist()
                        st.session_state.pred  = pred
                        st.session_state.proba = proba
                    except ValueError as e:
                        st.error(
                            f"Feature size mismatch: {e}\n\n"
                            "This means the app's feature extraction does not match "
                            "what was used during training.\n\n"
                            "Fix: retrain using **train_model.py** (not the notebook), "
                            "then copy the new tobacco_grader.pkl here."
                        )

                if st.session_state.pred is not None:
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    pi = st.session_state.pred
                    pb = st.session_state.proba
                    st.session_state.history.append({
                        "File":       st.session_state.last_fname or mode,
                        "Method":     badge_map[mode],
                        "Grade":      GRADES[pi][0],
                        "Type":       GRADES[pi][1],
                        "Confidence": f"{pb[pi]*100:.1f}%",
                    })

    else:
        st.markdown("""
        <div style="border:1.5px dashed #ddd;border-radius:10px;
                    padding:50px 20px;text-align:center;background:#fafafa">
            <div style="font-size:14px;color:#bbb">
                No image selected. Choose a method above.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Result display ────────────────────────────────────────
    if st.session_state.pred is not None and st.session_state.proba is not None:
        pi    = st.session_state.pred
        pb    = st.session_state.proba
        label, typ, card_cls, col = GRADES[pi]
        conf  = pb[pi] * 100

        st.markdown("---")
        st.markdown("<div class='sec-label'>Result</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card {card_cls}">
            <div class="grade-name">{label}</div>
            <div class="grade-type">{typ} &nbsp;|&nbsp; {conf:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(conf))

        st.markdown("<div class='sec-label' style='margin-top:12px'>Breakdown</div>",
                    unsafe_allow_html=True)

        for i, (lbl, t, _, c) in GRADES.items():
            p  = pb[i] * 100
            wt = "600" if i == pi else "400"
            r1, r2, r3 = st.columns([2, 5, 1])
            r1.markdown(
                f"<div style='font-size:13px;font-weight:{wt};padding-top:3px'>{lbl}</div>",
                unsafe_allow_html=True
            )
            r2.progress(int(p))
            r3.markdown(
                f"<div style='font-size:13px;font-weight:{wt};text-align:right;padding-top:3px'>{p:.1f}%</div>",
                unsafe_allow_html=True
            )

        st.markdown(f"<div class='tip-box'>{TIPS[pi]}</div>", unsafe_allow_html=True)


# ═══════════════════════
#  TAB 2 — Batch
# ═══════════════════════
with tab2:

    st.markdown(
        "<p style='color:#888;font-size:13px'>Run predictions on all images in a folder.</p>",
        unsafe_allow_html=True
    )

    folder = st.text_input(
        "Folder path",
        os.path.join(DATASET_PATH, "gold"),
        key="b_folder"
    )

    if st.button("Run batch", type="primary", key="b_run"):
        if not model_data:
            st.error("Model not loaded.")
        elif not os.path.isdir(folder):
            st.error(f"Folder not found: {folder}")
        else:
            files = (glob.glob(os.path.join(folder, "*.jpg"))  +
                     glob.glob(os.path.join(folder, "*.jpeg")) +
                     glob.glob(os.path.join(folder, "*.png")))
            if not files:
                st.warning("No images found.")
            else:
                bar  = st.progress(0, text=f"0 / {len(files)}")
                rows = []
                for i, f in enumerate(files):
                    try:
                        img   = cv2.imread(f)
                        feats = get_features(img).reshape(1, -1)
                        pred  = model_data["model"].predict(feats)[0]
                        prob  = model_data["model"].predict_proba(feats)[0]
                        conf  = round(prob[pred] * 100, 1)
                        rows.append({
                            "File":       os.path.basename(f),
                            "Grade":      GRADES[pred][0],
                            "Type":       GRADES[pred][1],
                            "Confidence": f"{conf}%",
                            "A%": f"{prob[0]*100:.1f}",
                            "B%": f"{prob[1]*100:.1f}",
                            "C%": f"{prob[2]*100:.1f}",
                        })
                    except Exception:
                        pass
                    bar.progress((i+1)/len(files), text=f"{i+1} / {len(files)}")
                bar.empty()

                df = pd.DataFrame(rows)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total",   len(df))
                c2.metric("Grade A", len(df[df["Grade"]=="Grade A"]))
                c3.metric("Grade B", len(df[df["Grade"]=="Grade B"]))
                c4.metric("Grade C", len(df[df["Grade"]=="Grade C"]))

                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download CSV",
                    df.to_csv(index=False).encode(),
                    "batch_results.csv",
                    "text/csv"
                )


# ═══════════════════════
#  TAB 3 — History
# ═══════════════════════
with tab3:

    hist = st.session_state.get("history", [])
    if not hist:
        st.info("No predictions yet.")
    else:
        df_h = pd.DataFrame(hist)
        st.write(f"{len(df_h)} prediction(s) this session")
        st.dataframe(df_h, use_container_width=True, hide_index=True)

        dl, cl = st.columns([1, 4])
        with dl:
            st.download_button(
                "Download",
                df_h.to_csv(index=False).encode(),
                "history.csv",
                "text/csv"
            )
        with cl:
            if st.button("Clear"):
                st.session_state.history = []
                st.rerun()