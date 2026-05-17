import streamlit as st
import cv2
import numpy as np
import joblib
import os
import glob
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

st.set_page_config(page_title="Tobacco Leaf Grader", page_icon="🌿", layout="centered")

MODEL_PATH     = "tobacco_grader.pkl"
VALIDATOR_PATH = "tobacco_validator.pkl"
DATASET_PATH   = r"C:\Users\BALU LOHITH REDDY\Downloads\archive\tobacco leaves"
IMG_SIZE       = (224, 224)

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container { padding-top: 1.8rem; padding-bottom: 2rem; max-width: 700px; }
html, body, [class*="css"] { font-family: 'Segoe UI', Arial, sans-serif; }

[data-testid="stSidebar"] { background: #1c3d20; }
[data-testid="stSidebar"] * { color: #dceedd !important; }
[data-testid="stSidebar"] hr { border-color: #2e5c34 !important; }

.stButton > button[kind="primary"] {
    background: #2e7d32 !important; color: #fff !important;
    border: none !important; border-radius: 7px !important;
    font-size: 14px !important; font-weight: 600 !important;
    padding: 0.55rem 1.2rem !important; width: 100%;
}
.stButton > button[kind="primary"]:hover { background: #1b5e20 !important; }

div[data-testid="stHorizontalBlock"] .stButton button {
    border-radius: 6px !important; border: 1px solid #ccc !important;
    background: #fff !important; color: #333 !important;
    font-size: 13px !important; font-weight: 500 !important;
}
div[data-testid="stHorizontalBlock"] .stButton button:hover {
    border-color: #2e7d32 !important; color: #2e7d32 !important;
    background: #f4fbf4 !important;
}

.result-card { border-radius: 10px; padding: 20px 18px; margin: 14px 0 8px; text-align: center; }
.card-a { background: #fff8e1; border: 1.5px solid #f9a825; }
.card-b { background: #e8f5e9; border: 1.5px solid #43a047; }
.card-c { background: #fbe9e7; border: 1.5px solid #e53935; }
.grade-name { font-size: 24px; font-weight: 700; margin: 2px 0; }
.grade-type { font-size: 13px; color: #666; }
.card-a .grade-name { color: #e65100; }
.card-b .grade-name { color: #1b5e20; }
.card-c .grade-name { color: #b71c1c; }

.warn-box {
    background: #fff3e0; border: 1.5px solid #ff9800;
    border-radius: 10px; padding: 16px 18px; margin: 14px 0;
    font-size: 14px; color: #e65100;
}
.mode-badge {
    display: inline-block; background: #e8f5e9; color: #2e7d32;
    font-size: 11px; font-weight: 600; padding: 2px 10px;
    border-radius: 4px; border: 1px solid #c8e6c9; margin-bottom: 10px;
}
.sec-label {
    font-size: 11px; font-weight: 600; color: #aaa;
    text-transform: uppercase; letter-spacing: 0.07em; margin: 14px 0 5px;
}
.tip-box {
    background: #f5f5f5; border-left: 3px solid #2e7d32;
    border-radius: 0 6px 6px 0; padding: 9px 12px;
    font-size: 13px; color: #444; margin-top: 10px;
}
[data-testid="stFileUploader"] {
    border: 1.5px dashed #ccc !important;
    border-radius: 8px !important; background: #fafafa !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  EXACT FEATURE EXTRACTION FROM notebook  (187 features)
#  Do NOT change these functions — they must match the notebook
# ─────────────────────────────────────────────────────────────

def remove_background(img_bgr):
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    bg   = cv2.inRange(hsv, np.array([0, 0, 190]), np.array([180, 40, 255]))
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bg   = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, k)
    bg   = cv2.morphologyEx(bg, cv2.MORPH_OPEN,  k)
    mask = cv2.bitwise_not(bg)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask    = np.uint8(labels == largest) * 255
    return cv2.bitwise_and(img_bgr, img_bgr, mask=mask), mask


def color_features(img_bgr, mask):
    feats    = []
    has_mask = mask is not None and np.any(mask > 0)
    hsv      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = (hsv[mask>0,0] if has_mask else hsv[:,:,0].flatten()).astype(np.float32)
    s = (hsv[mask>0,1] if has_mask else hsv[:,:,1].flatten()).astype(np.float32)
    v = (hsv[mask>0,2] if has_mask else hsv[:,:,2].flatten()).astype(np.float32)
    if len(h) == 0:
        return np.zeros(57, dtype=np.float32)
    h_hist,_ = np.histogram(h, bins=18, range=(0,180), density=True)
    s_hist,_ = np.histogram(s, bins=8,  range=(0,255), density=True)
    v_hist,_ = np.histogram(v, bins=8,  range=(0,255), density=True)
    feats.extend(h_hist); feats.extend(s_hist); feats.extend(v_hist)
    feats += [np.mean(h), np.std(h), np.median(h)]
    feats += [np.mean(s), np.std(s)]
    feats += [np.mean(v), np.std(v)]
    for p in [10,25,50,75,90]:
        feats.append(float(np.percentile(v,p)) / 255.0)
    lab    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab_px = lab[mask>0] if has_mask else lab.reshape(-1,3)
    for ch in range(3):
        feats.append(float(np.mean(lab_px[:,ch])))
        feats.append(float(np.std(lab_px[:,ch])))
    px     = (img_bgr[mask>0] if has_mask else img_bgr.reshape(-1,3)).astype(float)
    b_ch,g_ch,r_ch = px[:,0], px[:,1], px[:,2]
    total  = np.mean(r_ch)+np.mean(g_ch)+np.mean(b_ch)+1e-6
    feats += [np.mean(r_ch)/total, np.mean(g_ch)/total, np.mean(b_ch)/total]
    feats += [np.mean(r_ch)-np.mean(b_ch), np.mean(g_ch)-np.mean(b_ch)]
    return np.array(feats, dtype=np.float32)


def texture_features(img_bgr, mask):
    # NOTE: matches the notebook's second texture_features that trained the model.
    # Each GLCM property: 12 raw values + 4 stats = 16 per prop x 4 props = 64
    # + entropy(2) + sobel(3) + laplacian(1) = 70 total  [NOT 22]
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray  = cv2.resize(gray, (256, 256))
    feats = []
    glcm  = graycomatrix(gray, distances=[1,3,5],
                         angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                         levels=256, symmetric=True, normed=True)
    for prop in ['contrast','correlation','energy','homogeneity']:
        vals = graycoprops(glcm, prop).flatten()   # 12 values (3 dist x 4 angles)
        feats.extend([float(v) for v in vals])     # +12 raw values
        feats += [float(np.mean(vals)), float(np.std(vals)),
                  float(np.min(vals)),  float(np.max(vals))]  # +4 stats
    # 4 props x 16 = 64
    eps     = 1e-10
    entropy = -np.sum(glcm*np.log2(glcm+eps), axis=(0,1)).flatten()
    feats  += [float(np.mean(entropy)), float(np.std(entropy))]   # +2
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gm = np.sqrt(sx**2+sy**2).flatten()
    feats += [float(np.mean(gm)), float(np.std(gm)), float(np.percentile(gm,75))]  # +3
    feats.append(float(np.var(cv2.Laplacian(gray, cv2.CV_64F))))   # +1
    return np.array(feats, dtype=np.float32)  # total = 70


def shape_features(mask):
    if mask is None:
        return np.zeros(10, dtype=np.float32)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(10, dtype=np.float32)
    cnt  = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 200:
        return np.zeros(10, dtype=np.float32)
    x,y,w,h     = cv2.boundingRect(cnt)
    aspect_ratio = float(max(w,h)) / (min(w,h) + 1e-6)
    extent       = area / (w*h + 1e-6)
    hull      = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity  = area / (hull_area + 1e-6)
    perim            = cv2.arcLength(cnt, True)
    perim_area_ratio = (perim**2) / (area + 1e-6)
    equiv_diam       = np.sqrt(4*area/np.pi)
    eccentricity = 0.0
    if len(cnt) >= 5:
        el = cv2.fitEllipse(cnt)
        ma,MA = el[1]
        eccentricity = np.sqrt(1-(min(ma,MA)/(max(ma,MA)+1e-6))**2)
    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    try:
        defects = cv2.convexityDefects(cnt, hull_idx)
        n_def   = len(defects) if defects is not None else 0
        max_def = float(np.max(defects[:,0,3])/256.0) if defects is not None else 0.0
    except:
        n_def, max_def = 0, 0.0
    img_area  = mask.shape[0] * mask.shape[1]
    norm_area = area / img_area
    return np.array([
        aspect_ratio, extent, solidity, perim_area_ratio,
        equiv_diam / 224.0, eccentricity, float(n_def), max_def,
        norm_area, hull_area / (img_area + 1e-6),
    ], dtype=np.float32)


def vein_features(img_bgr, mask):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256,256)).astype(np.float32)
    m_r  = cv2.resize(mask, (256,256)) if mask is not None else None
    feats = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        kernel   = cv2.getGaborKernel(
            ksize=(21,21), sigma=4.0, theta=theta,
            lambd=10.0, gamma=0.5, psi=0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        resp     = filtered[m_r>0] if (m_r is not None and np.any(m_r>0)) else filtered.flatten()
        feats.append(float(np.mean(np.abs(resp))))
        feats.append(float(np.std(resp)))
        feats.append(float(np.max(np.abs(resp))))
        feats.append(float(np.percentile(np.abs(resp), 75)))
    return np.array(feats, dtype=np.float32)


def damage_features(img_bgr, mask):
    feats    = []
    has_mask = mask is not None and np.any(mask > 0)
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray  = cv2.resize(gray, (224,224))
    m_r   = cv2.resize(mask,(224,224)) if has_mask else np.ones((224,224),np.uint8)*255
    leaf_px   = gray[m_r>0]
    leaf_area = np.sum(m_r>0) + 1e-6
    feats.append(float(np.sum(leaf_px < 80) / leaf_area))
    feats.append(float(np.sum(leaf_px < 50) / leaf_area))
    contours,_ = cv2.findContours(m_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt      = max(contours, key=cv2.contourArea)
        hull_img = np.zeros_like(m_r)
        cv2.drawContours(hull_img, [cv2.convexHull(cnt)], -1, 255, -1)
        hull_a   = np.sum(hull_img>0) + 1e-6
        feats.append(float(np.clip(1.0 - leaf_area/hull_a, 0, 1)))
    else:
        feats.append(0.0)
    sx  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gm  = np.sqrt(sx**2+sy**2)
    bnd = cv2.dilate(m_r, np.ones((5,5),np.uint8)) - m_r
    if np.any(bnd > 0):
        ev = gm[bnd > 0]
        feats += [float(np.mean(ev)), float(np.std(ev))]
    else:
        feats += [0.0, 0.0]
    img_r = cv2.resize(img_bgr, (224,224))
    hsv   = cv2.cvtColor(img_r, cv2.COLOR_BGR2HSV)
    feats.append(float(np.std(hsv[:,:,0][m_r>0])))
    feats.append(float(np.std(hsv[:,:,1][m_r>0])))
    feats.append(float(np.std(hsv[:,:,2][m_r>0])))
    return np.array(feats, dtype=np.float32)


def lbp_features(img_bgr, mask):
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray     = cv2.resize(gray, (224,224))
    radius   = 3
    n_points = 8 * radius
    lbp      = local_binary_pattern(gray, n_points, radius, method='uniform')
    if mask is not None and np.any(mask > 0):
        m_r  = cv2.resize(mask, (224,224))
        vals = lbp[m_r > 0]
    else:
        vals = lbp.flatten()
    hist,_ = np.histogram(vals, bins=n_points+2, range=(0, n_points+2), density=True)
    return hist.astype(np.float32)


def extract_features(img_bgr):
    """187 features — must match notebook exactly."""
    img      = cv2.resize(img_bgr, IMG_SIZE)
    img_m, mask = remove_background(img)
    if np.sum(mask > 0) < IMG_SIZE[0]*IMG_SIZE[1]*0.05:
        img_m, mask = img, None
    cf = color_features(img_m, mask)    # 57
    tf = texture_features(img_m, mask)  # 70
    sf = shape_features(mask)           # 10
    vf = vein_features(img_m, mask)     # 16
    df = damage_features(img_m, mask)   # 8
    lf = lbp_features(img_m, mask)      # 26
    return np.concatenate([cf, tf, sf, vf, df, lf])  # 187


# ─────────────────────────────────────────────────────────────
#  BACKGROUND REMOVAL FOR DISPLAY
#  Produces a clean white-background preview shown to the user.
#  Uses HSV threshold first, GrabCut as fallback for non-white
#  backgrounds (outdoor / coloured table / coloured paper).
# ─────────────────────────────────────────────────────────────
def remove_bg_for_display(pil_img):
    """
    Returns (original_pil, bg_removed_pil, leaf_coverage_pct)
    """
    img_rgb  = np.array(pil_img.convert("RGB"))
    img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Resize for display only (keep aspect ratio, max 600px wide)
    h, w   = img_bgr.shape[:2]
    scale  = min(600/w, 800/h, 1.0)
    dw, dh = int(w*scale), int(h*scale)
    disp   = cv2.resize(img_bgr, (dw, dh))

    # ── Method 1: White-paper HSV threshold ──────────────────
    hsv  = cv2.cvtColor(disp, cv2.COLOR_BGR2HSV)
    bg   = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([180, 45, 255]))
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    bg   = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, k)
    bg   = cv2.morphologyEx(bg, cv2.MORPH_OPEN,  k)
    mask = cv2.bitwise_not(bg)

    # Keep largest connected component (the leaf)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask    = np.uint8(labels == largest) * 255

    fg_ratio = np.sum(mask > 0) / (dw * dh)

    # ── Method 2: GrabCut fallback (non-white background) ────
    if fg_ratio < 0.08 or fg_ratio > 0.92:
        gc_mask   = np.zeros((dh, dw), dtype=np.uint8)
        margin    = max(10, int(min(dh, dw) * 0.05))
        rect      = (margin, margin, dw - 2*margin, dh - 2*margin)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(disp, gc_mask, rect, bgd_model, fgd_model,
                        5, cv2.GC_INIT_WITH_RECT)
            gc_mask2 = np.where(
                (gc_mask == 2) | (gc_mask == 0), 0, 255
            ).astype(np.uint8)
            if np.sum(gc_mask2 > 0) > np.sum(mask > 0):
                mask = gc_mask2
        except Exception:
            pass

    # ── Smooth mask edges ─────────────────────────────────────
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # ── Composite onto white canvas ───────────────────────────
    alpha  = mask.astype(float) / 255.0
    alpha3 = np.stack([alpha]*3, axis=-1)
    white  = np.ones_like(disp) * 255
    result = (disp * alpha3 + white * (1 - alpha3)).astype(np.uint8)

    orig_pil   = Image.fromarray(cv2.cvtColor(disp,   cv2.COLOR_BGR2RGB))
    result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    coverage   = round(np.sum(mask > 0) / (dw * dh) * 100, 1)

    return orig_pil, result_pil, coverage


# ─────────────────────────────────────────────────────────────
#  RULE-BASED VALIDATOR (from notebook Cell 10)
# ─────────────────────────────────────────────────────────────
def rule_based_check(img_bgr):
    """Returns (is_valid: bool, reason: str)"""
    img = cv2.resize(img_bgr, IMG_SIZE)
    _, mask = remove_background(img)
    fg_ratio = np.sum(mask > 0) / (IMG_SIZE[0] * IMG_SIZE[1])

    if fg_ratio < 0.08:
        return False, "No leaf detected. Please place a tobacco leaf on white paper and scan again."
    if fg_ratio > 0.95:
        return False, "The image looks like a blank surface. Please scan the leaf on white paper only."

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if np.mean(hsv[:,:,1]) < 18:
        return False, "Image looks like a blank white or grey sheet, not a leaf."

    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    kernel = cv2.getGaborKernel((21,21), 4.0, 0, 10.0, 0.5, 0, cv2.CV_32F)
    resp   = cv2.filter2D(gray, cv2.CV_32F, kernel)
    vscore = float(np.mean(np.abs(resp[mask>0]))) if np.any(mask>0) else 0.0
    if vscore < 1.5:
        return False, "No vein pattern detected. This does not appear to be a leaf."

    return True, "ok"


# ─────────────────────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    grader    = joblib.load(MODEL_PATH)     if os.path.exists(MODEL_PATH)     else None
    validator = joblib.load(VALIDATOR_PATH) if os.path.exists(VALIDATOR_PATH) else None
    return grader, validator

grader_data, validator_data = load_models()


# ─────────────────────────────────────────────────────────────
#  FULL TWO-STAGE PREDICTION
# ─────────────────────────────────────────────────────────────
def predict(img_bgr):
    """
    Stage 1 : Rule-based check (blank paper / no vein / no saturation)
    Stage 2 : Grade classifier  ->  A / B / C

    NOTE: One-Class SVM validator removed from hard-reject path.
    It caused false rejections of real tobacco leaves (over-fitted to
    training image conditions). Rule-based checks are sufficient and
    more reliable in practice.
    """
    # Stage 1 — rule-based only (fast, reliable)
    valid, reason = rule_based_check(img_bgr)
    if not valid:
        return {"is_tobacco": False, "warning": reason,
                "grade": None, "proba": None}

    # Stage 2 — grade classifier
    feats = extract_features(img_bgr)
    model = grader_data["model"]
    pred  = model.predict([feats])[0]
    proba = model.predict_proba([feats])[0]
    return {"is_tobacco": True, "warning": None,
            "grade": pred, "proba": proba.tolist()}


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


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 4px 10px">
        <div style="font-size:18px;font-weight:700;color:#e8f5e9">Tobacco Grader</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Model status
    col_g, col_v = st.columns(2)
    col_g.markdown(
        f"<div style='font-size:11px;background:{'#2d5a32' if grader_data else '#5a2d2d'};"
        f"border-radius:5px;padding:5px 8px;text-align:center'>"
        f"{'✅' if grader_data else '❌'} Grader</div>",
        unsafe_allow_html=True
    )
    col_v.markdown(
        f"<div style='font-size:11px;background:{'#2d5a32' if validator_data else '#5a4a2d'};"
        f"border-radius:5px;padding:5px 8px;text-align:center'>"
        f"{'✅' if validator_data else '⚠️'} Validator</div>",
        unsafe_allow_html=True
    )
    if not grader_data:
        st.caption("Run model.ipynb to generate tobacco_grader.pkl")
    if not validator_data:
        st.caption("tobacco_validator.pkl missing — validator disabled")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:12px;font-weight:600;color:#a5d6a7;"
        "text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px'>Grade reference</div>",
        unsafe_allow_html=True
    )
    for _, (label, typ, _, col) in GRADES.items():
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;"
            f"padding:6px 0;border-bottom:1px solid #2e5c34'>"
            f"<div style='width:10px;height:10px;border-radius:2px;"
            f"background:{col};flex-shrink:0'></div>"
            f"<div><div style='font-size:13px;font-weight:600'>{label}</div>"
            f"<div style='font-size:11px;color:#81c784'>{typ}</div></div></div>",
            unsafe_allow_html=True
        )

    dataset_input = DATASET_PATH


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
st.markdown("## Tobacco Leaf Grader")
st.markdown(
    "<p style='color:#888;font-size:13px;margin-top:-8px;margin-bottom:18px'>"
    "Upload a photo or use your camera to identify the leaf grade instantly."
    "</p>", unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["Classify leaf", "Batch test", "History"])


# ═══════════════════════
#  TAB 1 — Classify
# ═══════════════════════
with tab1:

    for k, v in [("mode","upload"),("img_pil",None),
                 ("pred_result",None),("last_fname",None)]:
        if k not in st.session_state:
            st.session_state[k] = v
    # Reset scan mode (removed) to upload
    if st.session_state.mode == "scan":
        st.session_state.mode = "upload"

    # ── Mode buttons ──────────────────────────────────────────
    st.markdown("<div class='sec-label'>Select input method</div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Upload file", use_container_width=True):
            st.session_state.update(mode="upload", img_pil=None,
                                    pred_result=None)
    with c2:
        if st.button("Take photo", use_container_width=True):
            st.session_state.update(mode="camera", img_pil=None,
                                    pred_result=None)

    badge_map = {"upload":"Upload file","camera":"Take photo"}
    st.markdown(
        f"<div class='mode-badge'>{badge_map[st.session_state.mode]}</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ── Input widget ──────────────────────────────────────────
    img_pil = None
    mode    = st.session_state.mode

    if mode == "upload":
        st.markdown("<div class='sec-label'>Upload image</div>",
                    unsafe_allow_html=True)
        f = st.file_uploader("Choose leaf photo", type=["jpg","jpeg","png"],
                              label_visibility="collapsed", key="file_up")
        if f:
            if st.session_state.last_fname != f.name:
                st.session_state.pred_result = None
                st.session_state.last_fname  = f.name
            img_pil = Image.open(f)

    elif mode == "camera":
        st.markdown("<div class='sec-label'>Camera</div>", unsafe_allow_html=True)
        st.caption("Point at the leaf and press the capture button.")
        cam = st.camera_input("Camera", label_visibility="collapsed", key="cam_cap")
        if cam:
            img_pil = Image.open(cam)
            if st.session_state.last_fname != "cam":
                st.session_state.pred_result = None
                st.session_state.last_fname  = "cam"


    if img_pil:
        st.session_state.img_pil = img_pil

    # ── Preview + background removal + predict button ─────────
    if st.session_state.get("img_pil"):
        pil = st.session_state.img_pil

        # Run background removal for display
        with st.spinner("Removing background..."):
            orig_pil, clean_pil, coverage = remove_bg_for_display(pil)

        # Show before / after side by side
        st.markdown("<div class='sec-label'>Preview</div>", unsafe_allow_html=True)
        col_orig, col_clean = st.columns(2)
        with col_orig:
            st.markdown(
                "<div style='font-size:11px;color:#aaa;text-align:center;"
                "margin-bottom:4px'>Original</div>",
                unsafe_allow_html=True
            )
            st.image(orig_pil, use_container_width=True)

        with col_clean:
            st.markdown(
                f"<div style='font-size:11px;color:#2e7d32;text-align:center;"
                f"margin-bottom:4px'>Background removed ({coverage}% leaf)</div>",
                unsafe_allow_html=True
            )
            st.image(clean_pil, use_container_width=True)

        # Coverage warning
        if coverage < 8:
            st.warning("Very little leaf detected in the image. "
                       "Make sure the leaf is visible against the background.")
        elif coverage > 92:
            st.warning("Background could not be fully separated. "
                       "Try placing the leaf on plain white paper.")

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        btn_col, clr_col = st.columns([4, 1])
        with btn_col:
            go = st.button("Predict grade", type="primary",
                           use_container_width=True, key="go_btn")
        with clr_col:
            if st.button("Clear", use_container_width=True, key="clr_btn"):
                st.session_state.update(img_pil=None, pred_result=None,
                                        last_fname=None)
                st.rerun()

        if go:
            if not grader_data:
                st.error("Grader model not loaded. Run model.ipynb first.")
            else:
                # Predict on the CLEAN (background-removed) image
                # so the model sees the same image as the display
                with st.spinner("Analysing..."):
                    try:
                        # Use the bg-removed image for prediction
                        clean_arr = np.array(clean_pil.convert("RGB"))
                        bgr       = cv2.cvtColor(clean_arr, cv2.COLOR_RGB2BGR)
                        result    = predict(bgr)
                        st.session_state.pred_result = result
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")

                # Save to history
                r = st.session_state.pred_result
                if r and r["is_tobacco"]:
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    pi = r["grade"]
                    pb = r["proba"]
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
        </div>""", unsafe_allow_html=True)

    # ── Result display ────────────────────────────────────────
    r = st.session_state.get("pred_result")
    if r is not None:
        st.markdown("---")
        st.markdown("<div class='sec-label'>Result</div>", unsafe_allow_html=True)

        if not r["is_tobacco"]:
            # Warning — not a tobacco leaf
            st.markdown(f"""
            <div class="warn-box">
                <div style="font-size:16px;font-weight:700;margin-bottom:4px">
                    Not a tobacco leaf
                </div>
                <div>{r['warning']}</div>
            </div>""", unsafe_allow_html=True)

        else:
            pi    = r["grade"]
            pb    = r["proba"]
            label, typ, card_cls, col = GRADES[pi]
            conf  = pb[pi] * 100

            st.markdown(f"""
            <div class="result-card {card_cls}">
                <div class="grade-name">{label}</div>
                <div class="grade-type">{typ} leaf &nbsp;|&nbsp; {conf:.1f}% confidence</div>
            </div>""", unsafe_allow_html=True)

            st.progress(int(conf))

            st.markdown("<div class='sec-label' style='margin-top:12px'>Breakdown</div>",
                        unsafe_allow_html=True)
            for i, (lbl, t, _, c) in GRADES.items():
                p  = pb[i] * 100
                wt = "600" if i == pi else "400"
                r1, r2, r3 = st.columns([2, 5, 1])
                r1.markdown(
                    f"<div style='font-size:13px;font-weight:{wt};padding-top:3px'>{lbl}</div>",
                    unsafe_allow_html=True)
                r2.progress(int(p))
                r3.markdown(
                    f"<div style='font-size:13px;font-weight:{wt};text-align:right;padding-top:3px'>{p:.1f}%</div>",
                    unsafe_allow_html=True)

            st.markdown(f"<div class='tip-box'>{TIPS[pi]}</div>", unsafe_allow_html=True)


# ═══════════════════════
#  TAB 2 — Batch
# ═══════════════════════
with tab2:
    st.markdown("<p style='color:#888;font-size:13px'>Run predictions on all images in a folder.</p>",
                unsafe_allow_html=True)

    folder = st.text_input("Folder path",
                            os.path.join(dataset_input, "gold"),
                            key="b_folder")

    if st.button("Run batch", type="primary", key="b_run"):
        if not grader_data:
            st.error("Grader model not loaded.")
        elif not os.path.isdir(folder):
            st.error(f"Folder not found: {folder}")
        else:
            files = (glob.glob(os.path.join(folder,"*.jpg"))  +
                     glob.glob(os.path.join(folder,"*.jpeg")) +
                     glob.glob(os.path.join(folder,"*.png")))
            if not files:
                st.warning("No images found.")
            else:
                bar  = st.progress(0, text=f"0 / {len(files)}")
                rows = []
                for i, f in enumerate(files):
                    try:
                        img    = cv2.imread(f)
                        result = predict(img)
                        if result["is_tobacco"]:
                            pi   = result["grade"]
                            pb   = result["proba"]
                            conf = round(pb[pi]*100, 1)
                            rows.append({
                                "File":       os.path.basename(f),
                                "Is Tobacco": "Yes",
                                "Grade":      GRADES[pi][0],
                                "Type":       GRADES[pi][1],
                                "Confidence": f"{conf}%",
                                "A%": f"{pb[0]*100:.1f}",
                                "B%": f"{pb[1]*100:.1f}",
                                "C%": f"{pb[2]*100:.1f}",
                            })
                        else:
                            rows.append({
                                "File": os.path.basename(f), "Is Tobacco": "No",
                                "Grade":"N/A","Type":"N/A",
                                "Confidence":"N/A",
                                "A%":"","B%":"","C%":"",
                            })
                    except Exception:
                        pass
                    bar.progress((i+1)/len(files), text=f"{i+1} / {len(files)}")
                bar.empty()

                df = pd.DataFrame(rows)
                accepted = len(df[df["Is Tobacco"]=="Yes"])
                rejected = len(df[df["Is Tobacco"]=="No"])

                c1,c2,c3,c4,c5 = st.columns(5)
                c1.metric("Total",    len(df))
                c2.metric("Accepted", accepted)
                c3.metric("Grade A",  len(df[df["Grade"]=="Grade A"]))
                c4.metric("Grade B",  len(df[df["Grade"]=="Grade B"]))
                c5.metric("Grade C",  len(df[df["Grade"]=="Grade C"]))

                if rejected:
                    st.warning(f"{rejected} image(s) rejected — not tobacco leaves.")

                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button("Download CSV",
                                   df.to_csv(index=False).encode(),
                                   "batch_results.csv", "text/csv")


# ═══════════════════════
#  TAB 3 — History
# ═══════════════════════
with tab3:
    hist = st.session_state.get("history", [])
    if not hist:
        st.info("No predictions yet — classify a leaf in the first tab.")
    else:
        df_h = pd.DataFrame(hist)
        st.write(f"{len(df_h)} prediction(s) this session")
        st.dataframe(df_h, use_container_width=True, hide_index=True)
        dl, cl = st.columns([1, 4])
        with dl:
            st.download_button("Download",
                               df_h.to_csv(index=False).encode(),
                               "history.csv", "text/csv")
        with cl:
            if st.button("Clear history"):
                st.session_state.history = []
                st.rerun()