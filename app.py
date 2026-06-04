import streamlit as st
import cv2
import numpy as np
import joblib
import os
import glob
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

st.set_page_config(
    page_title="LeafGrade — Tobacco Classifier",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
#  PATHS & CONSTANTS  (must match notebook exactly)
# ─────────────────────────────────────────────────────────────
MODEL_PATH     = "tobacco_grader.pkl"
VALIDATOR_PATH = "tobacco_validator.pkl"
IMG_SIZE       = (224, 224)

GRADES = {
    0: ("Grade A", "Gold",            "grade-a", "#e8a020"),
    1: ("Grade B", "Yellowish-Green", "grade-b", "#6dc44a"),
    2: ("Grade C", "Dark Brown",      "grade-c", "#d45050"),
}
GRADE_TIPS = {
    0: "Premium quality — ideal for high-value tobacco products and export.",
    1: "Standard quality — well-suited for mainstream blends.",
    2: "Lower grade — consider blending or reduced-price markets.",
}

# ─────────────────────────────────────────────────────────────
#  THEME / GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main { background: #0d1208 !important; }
[data-testid="stMainBlockContainer"] {
    max-width: 780px !important;
    padding: 0 1.5rem 4rem !important;
}
section[data-testid="stMain"] > div { padding-top: 0 !important; }
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: #c9d4c2 !important;
}
.hero-wrap {
    padding: 3.5rem 0 2.5rem;
    text-align: center;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: .25em;
    text-transform: uppercase;
    color: #5a8a3c;
    margin-bottom: 14px;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.4rem, 6vw, 3.6rem);
    color: #e8f0df;
    line-height: 1.1;
    margin: 0 0 12px;
    letter-spacing: -0.02em;
}
.hero-title em { color: #7dc354; font-style: italic; }
.hero-sub {
    font-size: 14px;
    color: #7a9070;
    max-width: 380px;
    margin: 0 auto;
    line-height: 1.6;
}
.hero-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 2rem 0;
}
.hero-divider::before, .hero-divider::after {
    content: '';
    flex: 1;
    height: 0.5px;
    background: #1e2d18;
}
.hero-divider span {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: #3a5030;
    letter-spacing: .15em;
}
.card {
    background: #111a0c;
    border: 0.5px solid #1e3018;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.sec-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: #3d6030;
    margin: 1.6rem 0 .6rem;
}
[data-testid="stFileUploader"] > label { display: none; }
[data-testid="stFileUploader"] > div {
    background: #0a0f07 !important;
    border: 1.5px dashed #1e3018 !important;
    border-radius: 12px !important;
    padding: 2.5rem 1rem !important;
}
[data-testid="stFileUploader"] > div:hover { border-color: #3a6028 !important; }
[data-testid="stFileUploader"] span { color: #4a6840 !important; font-size: 13px !important; }
[data-testid="stFileUploader"] small { color: #2e4525 !important; font-size: 11px !important; }
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    transition: all .18s ease !important;
}
.stButton > button[kind="primary"] {
    background: #2a5c1a !important;
    color: #b8e896 !important;
    border: 0.5px solid #3a7422 !important;
    font-size: 14px !important;
    padding: .65rem 1.4rem !important;
    width: 100% !important;
}
.stButton > button[kind="primary"]:hover {
    background: #336e1f !important;
    color: #cdf0a0 !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"],
.stButton > button:not([kind]) {
    background: #0d1208 !important;
    color: #5a7850 !important;
    border: 0.5px solid #1e3018 !important;
    font-size: 13px !important;
    padding: .55rem 1rem !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #0a0f07 !important;
    border: 0.5px solid #1a2814 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    margin-bottom: 1.4rem !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 9px !important;
    color: #3d5535 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #1e3a14 !important;
    color: #8dd65e !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }
.result-shell {
    border-radius: 16px;
    padding: 2rem 1.5rem;
    margin: 1rem 0;
    text-align: center;
}
.grade-a-bg  { background: #1a1400; border: 0.5px solid #4a3800; }
.grade-b-bg  { background: #0d1a0a; border: 0.5px solid #1e4014; }
.grade-c-bg  { background: #1a0c0c; border: 0.5px solid #4a1818; }
.rejected-bg { background: #0f0f12; border: 0.5px solid #2a2040; }
.grade-letter {
    font-family: 'DM Serif Display', serif;
    font-size: 72px;
    line-height: 1;
    margin-bottom: 4px;
    letter-spacing: -0.04em;
}
.grade-a-txt   { color: #e8a020; }
.grade-b-txt   { color: #6dc44a; }
.grade-c-txt   { color: #d45050; }
.rejected-txt  { color: #6060a0; }
.grade-label   { font-size: 15px; font-weight: 600; margin-bottom: 2px; }
.grade-desc    { font-size: 12px; color: #7a9070; }
.conf-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 6px;
    margin-top: 10px;
    letter-spacing: .05em;
}
.conf-a { background: #2a1e00; color: #c89030; border: 0.5px solid #4a3010; }
.conf-b { background: #0e2008; color: #5aac38; border: 0.5px solid #1e4014; }
.conf-c { background: #200e0e; color: #c04040; border: 0.5px solid #3c1010; }
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
    border-bottom: 0.5px solid #141d0f;
}
.prob-row:last-child { border-bottom: none; }
.prob-label { font-size: 12px; font-weight: 500; min-width: 64px; color: #7a9070; }
.prob-bar-track { flex: 1; height: 4px; background: #141d0f; border-radius: 2px; overflow: hidden; }
.prob-bar-fill  { height: 100%; border-radius: 2px; }
.prob-bar-a { background: #c08020; }
.prob-bar-b { background: #5aac38; }
.prob-bar-c { background: #c04040; }
.prob-pct { font-family: 'DM Mono', monospace; font-size: 11px; min-width: 40px; text-align: right; color: #5a7850; }
.prob-pct.winner { color: #c9d4c2; font-weight: 500; }
.info-pill {
    background: #0e1c0a;
    border: 0.5px solid #1e3818;
    border-left: 2px solid #3a7020;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 13px;
    color: #7a9c68;
    margin-top: 10px;
    line-height: 1.5;
}
.warn-pill {
    background: #120e00;
    border: 0.5px solid #3a2c00;
    border-left: 2px solid #8a6000;
    border-radius: 0 8px 8px 0;
    padding: 12px 14px;
    font-size: 13px;
    color: #c09040;
    margin: 10px 0;
}
.error-pill {
    background: #120808;
    border: 0.5px solid #3a1414;
    border-left: 2px solid #8a2020;
    border-radius: 0 8px 8px 0;
    padding: 12px 14px;
    font-size: 13px;
    color: #c06060;
    margin: 10px 0;
}
[data-testid="stMetric"] {
    background: #0e160a !important;
    border: 0.5px solid #1a2c14 !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: .12em !important;
    color: #3d6030 !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif !important;
    font-size: 28px !important;
    color: #b4d898 !important;
}
[data-testid="stDataFrame"] {
    border: 0.5px solid #1a2814 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-testid="stProgressBar"] > div { background: #0a0f07 !important; border-radius: 4px !important; }
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #2a5c1a, #5aac38) !important;
    border-radius: 4px !important;
}
[data-testid="stTextInput"] input {
    background: #0a0f07 !important;
    border: 0.5px solid #1e3018 !important;
    border-radius: 10px !important;
    color: #c9d4c2 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    padding: 10px 14px !important;
}
[data-testid="stTextInput"] label { display: none !important; }
[data-testid="stCameraInput"] > div {
    background: #0a0f07 !important;
    border: 0.5px solid #1e3018 !important;
    border-radius: 12px !important;
}
[data-testid="stCameraInput"] label { display: none !important; }
[data-testid="stSidebar"] {
    background: #080e05 !important;
    border-right: 0.5px solid #1a2814 !important;
}
[data-testid="stSidebar"] * { color: #7a9070 !important; }
[data-testid="stDownloadButton"] button {
    background: #0d1208 !important;
    color: #5a8840 !important;
    border: 0.5px solid #1e3018 !important;
    border-radius: 10px !important;
    font-size: 12px !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stImage"] img { border-radius: 12px !important; border: 0.5px solid #1e3018 !important; }
.drop-zone {
    background: #0a0f07;
    border: 1.5px dashed #1e3018;
    border-radius: 14px;
    padding: 60px 20px;
    text-align: center;
}
.drop-icon { font-size: 36px; margin-bottom: 10px; opacity: .5; }
.drop-text  { font-size: 13px; color: #3a5530; line-height: 1.6; }
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.dot-ok   { background: #4aaa28; box-shadow: 0 0 6px rgba(74,170,40,.5); }
.dot-err  { background: #aa3030; }
.legend-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
    border-bottom: 0.5px solid #111a0c;
    font-size: 13px;
}
.legend-row:last-child { border-bottom: none; }
.legend-dot { width: 8px; height: 8px; border-radius: 2px; flex-shrink: 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION  — identical to model.ipynb (187 features)
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
    return np.array(feats, dtype=np.float32)   # 57


def texture_features(img_bgr, mask):
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray  = cv2.resize(gray, (256, 256))
    feats = []
    glcm  = graycomatrix(gray, distances=[1,3,5],
                         angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                         levels=256, symmetric=True, normed=True)
    for prop in ['contrast','correlation','energy','homogeneity']:
        vals = graycoprops(glcm, prop).flatten()
        feats.extend([float(v) for v in vals])
        feats += [float(np.mean(vals)), float(np.std(vals)),
                  float(np.min(vals)),  float(np.max(vals))]
    eps     = 1e-10
    entropy = -np.sum(glcm*np.log2(glcm+eps), axis=(0,1)).flatten()
    feats  += [float(np.mean(entropy)), float(np.std(entropy))]
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gm = np.sqrt(sx**2+sy**2).flatten()
    feats += [float(np.mean(gm)), float(np.std(gm)), float(np.percentile(gm,75))]
    feats.append(float(np.var(cv2.Laplacian(gray, cv2.CV_64F))))
    return np.array(feats, dtype=np.float32)   # 70


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
    hull         = cv2.convexHull(cnt)
    hull_area    = cv2.contourArea(hull)
    solidity     = area / (hull_area + 1e-6)
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
    ], dtype=np.float32)   # 10


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
    return np.array(feats, dtype=np.float32)   # 16


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
    return np.array(feats, dtype=np.float32)   # 8


def lbp_features(img_bgr, mask):
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray     = cv2.resize(gray, (224,224))
    radius   = 3
    n_points = 8 * radius   # 24
    lbp      = local_binary_pattern(gray, n_points, radius, method='uniform')
    if mask is not None and np.any(mask > 0):
        m_r  = cv2.resize(mask, (224,224))
        vals = lbp[m_r > 0]
    else:
        vals = lbp.flatten()
    hist,_ = np.histogram(vals, bins=n_points+2, range=(0, n_points+2), density=True)
    return hist.astype(np.float32)   # 26


def extract_features(img_bgr):
    """Returns 187-feature vector — identical order to notebook."""
    img         = cv2.resize(img_bgr, IMG_SIZE)
    img_m, mask = remove_background(img)
    if np.sum(mask > 0) < IMG_SIZE[0]*IMG_SIZE[1]*0.05:
        img_m, mask = img, None
    cf = color_features(img_m, mask)    # 57
    tf = texture_features(img_m, mask)  # 70
    sf = shape_features(mask)           # 10
    vf = vein_features(img_m, mask)     # 16
    df = damage_features(img_m, mask)   # 8
    lf = lbp_features(img_m, mask)      # 26
    return np.concatenate([cf, tf, sf, vf, df, lf])   # 187


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
#  PREDICTION  — mirrors predict_full_pipeline() from notebook
#
#  Stage 1 : Rule-based checks  (fast: coverage + saturation)
#  Stage 2 : One-Class SVM      (tobacco_validator.pkl)
#  Stage 3 : Gradient Boosting  (tobacco_grader.pkl)
# ─────────────────────────────────────────────────────────────
def rule_based_check(img_bgr):
    """Lightweight pre-filter. Returns (is_valid, reason)."""
    img      = cv2.resize(img_bgr, IMG_SIZE)
    _, mask  = remove_background(img)
    fg_ratio = np.sum(mask > 0) / (IMG_SIZE[0] * IMG_SIZE[1])

    if fg_ratio < 0.08:
        return False, ("No leaf detected. "
                       "Place the tobacco leaf on a plain white background and try again.")
    if fg_ratio > 0.95:
        return False, ("The image looks like a blank surface. "
                       "Please scan the leaf on white paper only.")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if np.mean(hsv[:,:,1]) < 18:
        return False, "Image looks like a blank white or grey sheet — no leaf detected."

    return True, "ok"


def predict(img_bgr):
    """
    Full two-stage pipeline exactly as in the notebook.

    Returns dict:
        is_tobacco  : bool
        warning     : str or None
        grade       : int index (0/1/2) or None
        grade_name  : str or None
        confidence  : float or None
        proba       : list [pA, pB, pC] or None
    """
    # ── Stage 1 : Rule-based ──────────────────────────────────
    valid, reason = rule_based_check(img_bgr)
    if not valid:
        return {"is_tobacco": False, "warning": reason,
                "grade": None, "grade_name": None,
                "confidence": None, "proba": None}

    # ── Extract features once (shared by stages 2 & 3) ───────
    feats = extract_features(img_bgr)

    # ── Stage 2 : One-Class SVM validator ────────────────────
    if validator_data is not None:
        val = validator_data["validator"]
        sc  = validator_data["scaler"]
        fs  = sc.transform([feats])
        if val.predict(fs)[0] != 1:
            return {"is_tobacco": False,
                    "warning": ("This does not look like a tobacco leaf. "
                                "Please scan a tobacco leaf only."),
                    "grade": None, "grade_name": None,
                    "confidence": None, "proba": None}

    # ── Stage 3 : Gradient Boosting grader ───────────────────
    model = grader_data["model"]          # sklearn Pipeline (scaler + GB)
    pred  = int(model.predict([feats])[0])
    proba = model.predict_proba([feats])[0].tolist()
    conf  = round(proba[pred] * 100, 1)

    return {
        "is_tobacco":  True,
        "warning":     None,
        "grade":       pred,
        "grade_name":  GRADES[pred][0],
        "confidence":  conf,
        "proba":       proba,
    }


# ─────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">Computer vision · Leaf analysis</div>
    <h1 class="hero-title">Tobacco<br><em>Leaf Grader</em></h1>
    <p class="hero-sub">Upload or capture a leaf — the model grades it instantly.</p>
</div>
<div class="hero-divider"><span>● ● ●</span></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  MODEL STATUS
# ─────────────────────────────────────────────────────────────
if not grader_data:
    st.markdown("""
    <div class="error-pill">
        <strong>Grader model not found.</strong><br>
        Run <code>model.ipynb</code> to generate <code>tobacco_grader.pkl</code>
        and <code>tobacco_validator.pkl</code>, then place them in the same folder as this file.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not validator_data:
    st.markdown("""
    <div class="warn-pill">
        <strong>Validator model not found.</strong><br>
        <code>tobacco_validator.pkl</code> is missing. Non-leaf images may not be
        rejected correctly. Re-run <code>model.ipynb</code> to regenerate it.
    </div>
    """, unsafe_allow_html=True)
else:
    model_name = grader_data.get("model_name", "Gradient Boosting")
    test_acc   = grader_data.get("test_acc", None)
    acc_str    = f" · {test_acc*100:.1f}% test accuracy" if test_acc else ""
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;
                font-family:'DM Mono',monospace;font-size:11px;
                color:#3d6030;margin-bottom:1rem">
        <span class="status-dot dot-ok"></span>
        {model_name} loaded{acc_str} &nbsp;·&nbsp; Validator ready
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🌿  Single Leaf", "📂  Batch", "📋  History"])


# ═══════════════════════════════════════════════════
#  TAB 1 — Single leaf
# ═══════════════════════════════════════════════════
with tab1:

    # ── Input mode ─────────────────────────────────
    st.markdown("<div class='sec-label'>Input method</div>", unsafe_allow_html=True)
    mode = st.radio("mode", ["Upload image", "Use camera"],
                    horizontal=True, label_visibility="collapsed")

    img_bgr = None

    if mode == "Upload image":
        uploaded = st.file_uploader(
            "upload", type=["jpg","jpeg","png"],
            label_visibility="collapsed")
        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            st.image(pil_img, use_container_width=True)
    else:
        cam = st.camera_input("camera", label_visibility="collapsed")
        if cam:
            pil_img = Image.open(cam).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    col_btn, col_clr = st.columns([3, 1])
    with col_btn:
        analyse = st.button("Analyse leaf →", type="primary",
                            disabled=(img_bgr is None), key="s_analyse")
    with col_clr:
        if st.button("✕", key="s_clear"):
            st.session_state.pop("pred_result", None)
            st.rerun()

    if analyse and img_bgr is not None:
        with st.spinner("Analysing…"):
            result = predict(img_bgr)

        st.session_state["pred_result"] = result

        # Append to history
        if "history" not in st.session_state:
            st.session_state["history"] = []
        if result["is_tobacco"]:
            pi = result["grade"]
            pb = result["proba"]
            st.session_state["history"].append({
                "Grade":      GRADES[pi][0],
                "Type":       GRADES[pi][1],
                "Confidence": f"{result['confidence']}%",
                "A%":         f"{pb[0]*100:.1f}",
                "B%":         f"{pb[1]*100:.1f}",
                "C%":         f"{pb[2]*100:.1f}",
            })
        else:
            st.session_state["history"].append({
                "Grade": "Rejected", "Type": "—", "Confidence": "—",
                "A%": "", "B%": "", "C%": "",
            })

    # ── Result ─────────────────────────────────────
    r = st.session_state.get("pred_result")
    if r is not None:
        st.markdown("<div class='sec-label' style='margin-top:1.4rem'>Result</div>",
                    unsafe_allow_html=True)

        if not r["is_tobacco"]:
            st.markdown(f"""
            <div class="result-shell rejected-bg">
                <div class="grade-letter rejected-txt">✗</div>
                <div class="grade-label" style="color:#8080c0">Not a tobacco leaf</div>
                <div class="grade-desc" style="margin-top:6px">{r['warning']}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            pi    = r["grade"]
            pb    = r["proba"]
            label, typ, slug, col = GRADES[pi]
            letter = label.split()[-1]   # "A", "B", or "C"

            st.markdown(f"""
            <div class="result-shell {slug}-bg">
                <div class="grade-letter {slug}-txt">{letter}</div>
                <div class="grade-label" style="color:{col}">{label} — {typ}</div>
                <div class="grade-desc">Detected with</div>
                <div class="conf-badge conf-{letter.lower()}">{r['confidence']}% confidence</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='sec-label' style='margin-top:1rem'>Grade probabilities</div>",
                        unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            for i, (lbl, t, s, c) in GRADES.items():
                pct    = pb[i] * 100
                is_win = (i == pi)
                ltr    = lbl.split()[-1]
                st.markdown(f"""
                <div class="prob-row">
                    <span class="prob-label" style="color:{'#c9d4c2' if is_win else ''}">{lbl}</span>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill prob-bar-{ltr.lower()}"
                             style="width:{pct:.1f}%"></div>
                    </div>
                    <span class="prob-pct {'winner' if is_win else ''}">{pct:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="info-pill">{GRADE_TIPS[pi]}</div>',
                        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
#  TAB 2 — Batch
# ═══════════════════════════════════════════════════
with tab2:
    st.markdown(
        "<p style='color:#4a6840;font-size:13px;margin-top:-.4rem;margin-bottom:1.2rem'>"
        "Run predictions on every image in a folder and download the results as CSV.</p>",
        unsafe_allow_html=True)

    st.markdown("<div class='sec-label'>Folder path</div>", unsafe_allow_html=True)
    folder = st.text_input("folder", value="", placeholder="/path/to/leaf/images",
                            label_visibility="collapsed", key="b_folder")

    if st.button("Run batch prediction →", type="primary", key="b_run"):
        if not folder or not os.path.isdir(folder):
            st.markdown('<div class="warn-pill">Folder not found — check the path above.</div>',
                        unsafe_allow_html=True)
        else:
            files = (glob.glob(os.path.join(folder,"*.jpg"))  +
                     glob.glob(os.path.join(folder,"*.jpeg")) +
                     glob.glob(os.path.join(folder,"*.png")))
            if not files:
                st.markdown('<div class="warn-pill">No JPG/PNG images found in that folder.</div>',
                            unsafe_allow_html=True)
            else:
                bar  = st.progress(0, text=f"0 / {len(files)}")
                rows = []
                for i, fp in enumerate(files):
                    try:
                        img = cv2.imread(fp)
                        r   = predict(img)
                        if r["is_tobacco"]:
                            pi = r["grade"]
                            pb = r["proba"]
                            rows.append({
                                "File":       os.path.basename(fp),
                                "Tobacco":    "Yes",
                                "Grade":      GRADES[pi][0],
                                "Type":       GRADES[pi][1],
                                "Confidence": f"{r['confidence']}%",
                                "A%":         f"{pb[0]*100:.1f}",
                                "B%":         f"{pb[1]*100:.1f}",
                                "C%":         f"{pb[2]*100:.1f}",
                            })
                        else:
                            rows.append({
                                "File":       os.path.basename(fp),
                                "Tobacco":    "No",
                                "Grade":      "—", "Type": "—",
                                "Confidence": "—",
                                "A%": "", "B%": "", "C%": "",
                            })
                    except Exception:
                        pass
                    bar.progress((i+1)/len(files), text=f"Processing {i+1} / {len(files)}")
                bar.empty()

                df       = pd.DataFrame(rows)
                accepted = len(df[df["Tobacco"]=="Yes"])
                rejected = len(df) - accepted
                a_count  = len(df[df["Grade"]=="Grade A"])
                b_count  = len(df[df["Grade"]=="Grade B"])
                c_count  = len(df[df["Grade"]=="Grade C"])

                m1,m2,m3,m4,m5,m6 = st.columns(6)
                m1.metric("Total",    len(df))
                m2.metric("Accepted", accepted)
                m3.metric("Rejected", rejected)
                m4.metric("Grade A",  a_count)
                m5.metric("Grade B",  b_count)
                m6.metric("Grade C",  c_count)

                if rejected:
                    st.markdown(
                        f'<div class="warn-pill">{rejected} image(s) rejected — '
                        f'not identified as tobacco leaves.</div>',
                        unsafe_allow_html=True)

                st.markdown("<div class='sec-label'>Results</div>", unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button("↓  Download CSV",
                                   df.to_csv(index=False).encode(),
                                   "batch_results.csv", "text/csv")


# ═══════════════════════════════════════════════════
#  TAB 3 — History
# ═══════════════════════════════════════════════════
with tab3:
    hist = st.session_state.get("history", [])
    if not hist:
        st.markdown("""
        <div class="drop-zone">
            <div class="drop-icon">📋</div>
            <div class="drop-text">No predictions yet.<br>
            Classify a leaf in the first tab to see history here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            f"<p style='color:#4a6840;font-size:13px;margin-bottom:1rem'>"
            f"{len(hist)} prediction(s) this session</p>",
            unsafe_allow_html=True)
        df_h = pd.DataFrame(hist)
        st.dataframe(df_h, use_container_width=True, hide_index=True)
        dl_col, cl_col, _ = st.columns([1, 1, 3])
        with dl_col:
            st.download_button("↓  Download",
                               df_h.to_csv(index=False).encode(),
                               "history.csv", "text/csv")
        with cl_col:
            if st.button("Clear", key="clr_hist"):
                st.session_state.history = []
                st.rerun()


# ─────────────────────────────────────────────────────────────
#  SIDEBAR — Grade reference
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 .8rem;font-family:'DM Serif Display',serif;
                font-size:18px;color:#a0c880">Grade Reference</div>
    """, unsafe_allow_html=True)

    for _, (label, typ, slug, col) in GRADES.items():
        st.markdown(f"""
        <div class="legend-row">
            <div class="legend-dot" style="background:{col}"></div>
            <div>
                <div style="font-size:13px;font-weight:600;color:#c9d4c2">{label}</div>
                <div style="font-size:11px;color:#4a6840">{typ}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.2rem;font-family:'DM Mono',monospace;font-size:10px;
                color:#2a4020;letter-spacing:.12em;text-transform:uppercase">
        Features
    </div>
    <div style="font-size:12px;color:#3a5830;line-height:1.8;margin-top:.5rem">
        Color · 57<br>Texture · 70<br>Shape · 10<br>
        Vein · 16<br>Damage · 8<br>LBP · 26<br>
        <span style="color:#2a4020">─────────</span><br>
        Total · 187
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px;color:#2a4020;line-height:1.7">
        Place leaf on a plain white or light background for best results.
    </div>
    """, unsafe_allow_html=True)