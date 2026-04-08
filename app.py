"""
A Predictive Model for Optimizing CNNs Using Pruning, Quantization, and SVD
Streamlit App — ResNet18 on CIFAR-10
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle, os, copy, random
import gdown
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ─────────────────────────────────────────────────────────────
# DOWNLOAD BASELINE FROM GOOGLE DRIVE
# ─────────────────────────────────────────────────────────────
def download_baseline():
    if not os.path.exists("baseline.pth"):
        file_id = "1wxux74QrYPjTTJ7QZ3QElmKFhm2nEerc"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, "baseline.pth", quiet=False)

download_baseline()

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CNN Compression Predictor",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS — professional dark-accent theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── font & background ── */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
}
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
section[data-testid="stSidebar"] .stSlider > div > div > div { background: #4fc3f7; }

/* ── metric cards ── */
div[data-testid="metric-container"] {
    background: #f0f4ff;
    border: 1px solid #d0d8f0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

/* ── tabs ── */
button[data-baseweb="tab"] {
    font-size: 15px;
    font-weight: 600;
    padding: 10px 24px;
}

/* ── project title banner ── */
.project-banner {
    background: linear-gradient(135deg, #1a237e 0%, #283593 40%, #1565c0 100%);
    border-radius: 16px;
    padding: 32px 40px;
    color: white;
    margin-bottom: 24px;
}
.project-banner h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 8px 0;
    color: white;
}
.project-banner p  { font-size: 1.05rem; margin: 0; opacity: 0.88; }

/* ── info cards ── */
.info-card {
    background: #ffffff;
    border-left: 5px solid #1565c0;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.07);
}
.info-card h4 { margin: 0 0 6px 0; color: #1565c0; font-size: 1rem; }
.info-card p  { margin: 0; color: #444; font-size: 0.93rem; line-height: 1.5; }

/* ── technique cards ── */
.tech-card {
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 10px;
    color: white;
}
.tech-pruning      { background: linear-gradient(135deg, #1565c0, #1976d2); }
.tech-quantization { background: linear-gradient(135deg, #6a1b9a, #8e24aa); }
.tech-svd          { background: linear-gradient(135deg, #00695c, #00897b); }
.tech-card h3 { margin: 0 0 8px 0; font-size: 1.1rem; }
.tech-card p  { margin: 0; font-size: 0.9rem; opacity: 0.93; line-height: 1.5; }

/* ── divider ── */
hr { border-color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]

BASELINE = {
    "accuracy":       87.28,
    "nonzero_params": 11_181_642,
    "model_size_mb":  42.7311,
    "gflops":         3.6271,
}

QUANT_MAP = {"None": 0, "FP16": 1, "INT8": 2}

# Normalization used during baseline training — must match exactly
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD  = (0.5, 0.5, 0.5)

# Resize used during baseline training
INFERENCE_SIZE = 224

# ─────────────────────────────────────────────────────────────
# LOAD ASSETS  (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_predictive_models():
    with open("compression_models.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_baseline_model():
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    ckpt = torch.load("baseline.pth", map_location="cpu")
    if isinstance(ckpt, dict):
        state = (ckpt.get("model_state_dict")
                 or ckpt.get("state_dict")
                 or ckpt)
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

@st.cache_data
def load_cifar10_test():
    # Load raw — NO normalization here; we normalise inside classify_image
    tf = transforms.Compose([transforms.ToTensor()])
    return torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=tf
    )

@st.cache_data
def load_results_df():
    return pd.read_csv("compression_data.csv")

# ─────────────────────────────────────────────────────────────
# COMPRESSION FUNCTIONS  (unchanged from working version)
# ─────────────────────────────────────────────────────────────
def apply_pruning(model, sparsity):
    pruned = copy.deepcopy(model)
    layers = [(m, "weight") for m in pruned.modules()
              if isinstance(m, (nn.Conv2d, nn.Linear))]
    prune.global_unstructured(layers,
                              pruning_method=prune.L1Unstructured,
                              amount=sparsity)
    for m in pruned.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(m, "weight")
            except ValueError:
                pass
    return pruned


def apply_svd(model, rank_ratio):
    m = copy.deepcopy(model)
    m.eval()
    W    = m.fc.weight.data.float()
    bias = m.fc.bias.data.float() if m.fc.bias is not None else None
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    k = max(1, int(rank_ratio * min(W.shape)))
    fc_V = nn.Linear(W.shape[1], k, bias=False)
    fc_U = nn.Linear(k, W.shape[0], bias=(bias is not None))
    with torch.no_grad():
        fc_V.weight.copy_(Vh[:k, :])
        fc_U.weight.copy_(U[:, :k] * S[:k].unsqueeze(0))
        if bias is not None:
            fc_U.bias.copy_(bias)
    m.fc = nn.Sequential(fc_V, fc_U)
    return m


def apply_quantization(model, quant_mode):
    q = copy.deepcopy(model)
    q.eval()
    if quant_mode == "INT8":
        if isinstance(q.fc, nn.Sequential):
            new_layers = []
            for layer in q.fc:
                if isinstance(layer, nn.Linear):
                    nl = nn.Linear(layer.in_features, layer.out_features,
                                   bias=(layer.bias is not None))
                    with torch.no_grad():
                        nl.weight.copy_(layer.weight.float())
                        if layer.bias is not None:
                            nl.bias.copy_(layer.bias.float())
                    new_layers.append(nl)
                else:
                    new_layers.append(layer)
            q.fc = nn.Sequential(*new_layers)
        else:
            old = q.fc
            nl  = nn.Linear(old.in_features, old.out_features,
                             bias=(old.bias is not None))
            with torch.no_grad():
                nl.weight.copy_(old.weight.float())
                if old.bias is not None:
                    nl.bias.copy_(old.bias.float())
            q.fc = nl
        q = torch.quantization.quantize_dynamic(q, {nn.Linear}, dtype=torch.qint8)
    elif quant_mode == "FP16":
        q = q.half()
    return q


def build_compressed_model(baseline_model, pruning, quant, svd):
    model = copy.deepcopy(baseline_model)
    model.eval()
    if pruning > 0.0:
        model = apply_pruning(model, pruning)
    if svd > 0.0:
        model = apply_svd(model, svd)
    if quant != "None":
        model = apply_quantization(model, quant)
    return model


# ─────────────────────────────────────────────────────────────
# CLASSIFY IMAGE  — normalization fixed to match training
# ─────────────────────────────────────────────────────────────
def classify_image(model, img_tensor, quant_mode):
    """
    img_tensor : (C, H, W) float32 tensor in [0, 1]  — raw ToTensor output
    Pipeline   : Resize 32→224  →  Normalize(0.5, 0.5, 0.5)
    This exactly mirrors the test transform used during baseline training:
        transforms.Resize(224)
        transforms.ToTensor()
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    """
    # Step 1: resize raw 32×32 tensor to 224×224
    resize_tf = transforms.Resize(INFERENCE_SIZE,
                                  interpolation=transforms.InterpolationMode.BILINEAR)
    img_resized = resize_tf(img_tensor)           # (3, 224, 224)

    # Step 2: normalize with the exact same values used at training time
    norm_tf = transforms.Normalize(NORM_MEAN, NORM_STD)
    inp = norm_tf(img_resized).unsqueeze(0)       # (1, 3, 224, 224)

    # Step 3: move to correct dtype / device
    model = model.cpu()
    if quant_mode == "INT8":
        inp = inp.cpu().contiguous()
    elif quant_mode == "FP16":
        inp = inp.half().cpu()
    else:
        inp = inp.cpu()

    model.eval()
    with torch.no_grad():
        logits = model(inp)
        probs  = torch.softmax(logits.float(), dim=1)[0]
        pred   = int(probs.argmax().item())
        conf   = float(probs[pred].item()) * 100

    top5_probs, top5_idx = torch.topk(probs, 5)
    top5 = [(CIFAR10_CLASSES[i], float(p)*100)
            for i, p in zip(top5_idx.tolist(), top5_probs.tolist())]
    return pred, conf, top5


# ─────────────────────────────────────────────────────────────
# PREDICT METRICS
# ─────────────────────────────────────────────────────────────
def predict_metrics(pred_models, pruning, quant, svd):
    quant_num = QUANT_MAP[quant]
    X = np.array([[pruning, quant_num, svd]])
    return {
        "accuracy":       round(float(pred_models["accuracy"].predict(X)[0]), 2),
        "nonzero_params": int(pred_models["nonzero_params"].predict(X)[0]),
        "model_size_mb":  round(float(pred_models["model_size_mb"].predict(X)[0]), 4),
        "gflops":         BASELINE["gflops"],
    }


# ─────────────────────────────────────────────────────────────
# UI HELPER
# ─────────────────────────────────────────────────────────────
def metric_delta(label, value, baseline_value, lower_is_better=False, unit=""):
    delta = value - baseline_value
    delta_str = f"{delta:+.4g}{unit}"
    color = ("normal" if delta <= 0 else "inverse") if lower_is_better \
            else ("normal" if delta >= 0 else "inverse")
    st.metric(label=label,
              value=f"{value:,.4g}{unit}",
              delta=delta_str,
              delta_color=color)


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():

    # ── Project Banner ────────────────────────────────────────
    st.markdown("""
    <div class="project-banner">
        <h1>🧠 A Predictive Model for Optimizing Convolutional Neural Networks</h1>
        <p>Using Pruning, Quantization, and Matrix Factorization (SVD) &nbsp;|&nbsp;
           ResNet18 on CIFAR-10 &nbsp;|&nbsp; BK Birla College, Kalyan</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load assets ───────────────────────────────────────────
    try:
        pred_models    = load_predictive_models()
        baseline_model = load_baseline_model()
        dataset        = load_cifar10_test()
        df_results     = load_results_df()
        assets_ok      = True
    except Exception as e:
        st.error(f"❌ Failed to load assets: {e}")
        st.info("Make sure baseline.pth, compression_models.pkl, "
                "and compression_data.csv are present.")
        return

    # ── Sidebar ───────────────────────────────────────────────
    st.sidebar.markdown("## ⚙️ Compression Settings")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### ✂️ Pruning")
    use_pruning  = st.sidebar.checkbox("Apply Pruning", value=False)
    pruning_rate = st.sidebar.slider("Sparsity Level", 0.3, 0.7, 0.5, 0.1,
                                     disabled=not use_pruning,
                                     help="Fraction of weights set to zero")
    pruning_val  = pruning_rate if use_pruning else 0.0

    st.sidebar.markdown("### 🔢 Quantization")
    quant_mode = st.sidebar.selectbox("Precision Mode",
                                      ["None", "FP16", "INT8"],
                                      help="FP16 = half precision | INT8 = 8-bit integer")

    st.sidebar.markdown("### 🔬 SVD")
    use_svd   = st.sidebar.checkbox("Apply SVD", value=False)
    svd_ratio = st.sidebar.slider("Rank Ratio", 0.3, 0.9, 0.7, 0.1,
                                  disabled=not use_svd,
                                  help="Higher = more information retained")
    svd_val   = svd_ratio if use_svd else 0.0

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🖼️ Image Selection")
    img_index = st.sidebar.number_input(
        "Test Image Index (0 – 9999)",
        min_value=0, max_value=9999, value=42, step=1
    )
    if st.sidebar.button("🎲 Pick Random Image"):
        img_index = random.randint(0, 9999)
        st.sidebar.success(f"Selected index: {img_index}")

    st.sidebar.markdown("---")
    run_btn = st.sidebar.button("🚀 Run Prediction & Classify",
                                type="primary", use_container_width=True)

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🏠 Project Overview",
        "📊 Prediction & Classification",
        "📋 Experiment Results & Techniques"
    ])

    # ═════════════════════════════════════════════════════════
    # TAB 1 — PROJECT OVERVIEW
    # ═════════════════════════════════════════════════════════
    with tab1:
        st.markdown("## 📌 About This Project")
        st.markdown("""
        This research project investigates how **Convolutional Neural Networks (CNNs)**
        can be made smaller and faster for deployment on edge devices — mobile phones,
        microcontrollers, and embedded systems — without significantly sacrificing accuracy.
        """)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("""
            <div class="info-card">
                <h4>🎯 Objective</h4>
                <p>Apply three compression techniques — Pruning, Quantization, and SVD —
                to a ResNet18 model trained on CIFAR-10, and build a regression-based
                predictive model that can forecast post-compression accuracy without
                re-running expensive experiments.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-card">
                <h4>🏗️ Model & Dataset</h4>
                <p><b>Model:</b> ResNet18 — a deep residual network with 11.17M parameters.<br>
                <b>Dataset:</b> CIFAR-10 — 60,000 images across 10 classes (airplane, car,
                bird, cat, deer, dog, frog, horse, ship, truck), resized to 224×224.<br>
                <b>Baseline Accuracy:</b> 87.28%</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-card">
                <h4>📐 Predictive Framework</h4>
                <p>A Gradient Boosting Regressor was trained on 29 compression experiments
                to predict accuracy, model size, non-zero parameters, and GFLOPs from user-selected
                compression settings — enabling intelligent compression planning without
                exhaustive re-experimentation.</p>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("""
            <div class="info-card">
                <h4>🔬 What This App Does</h4>
                <p>
                • <b>Tab 2:</b> Select any combination of Pruning, Quantization, and SVD
                using the sidebar → instantly get predicted accuracy, model size, GFLOPs,
                and non-zero parameters. Also classifies any CIFAR-10 test image using the
                compressed model in real time.<br><br>
                • <b>Tab 3:</b> Explore all 29 experiment results, understand each compression
                technique, and see the accuracy trends across all methods.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-card">
                <h4>📊 Experiments Conducted</h4>
                <p>
                • <b>Individual:</b> Pruning (0.3–0.7), Quantization (FP16, INT8), SVD (0.3–0.9)<br>
                • <b>Combined:</b> Pruning + Quantization (6), SVD + Quantization (6),
                Pruning + SVD (3)<br>
                • <b>Triple:</b> Pruning + Quantization + SVD (2)<br>
                • <b>Total: 29 experiments</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-card">
                <h4>💡 Key Finding</h4>
                <p>Pruning and SVD — when combined with fine-tuning — actually
                <b>improve accuracy above the 87.28% baseline</b> (up to 91.06%)
                due to regularization effects. Quantization preserves baseline accuracy
                while halving model size (FP16) with no retraining required.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Baseline metrics at a glance
        st.markdown("### 📈 Baseline Model at a Glance")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("🎯 Accuracy",       "87.28%")
        b2.metric("💾 Model Size",      "42.73 MB")
        b3.metric("⚡ GFLOPs",          "3.6271")
        b4.metric("🔢 Parameters",      "11.18 M")

    # ═════════════════════════════════════════════════════════
    # TAB 2 — PREDICTION & CLASSIFICATION
    # ═════════════════════════════════════════════════════════
    with tab2:

        # ── Active compression badge ──────────────────────────
        active = []
        if use_pruning:          active.append(f"Pruning (sparsity={pruning_val})")
        if quant_mode != "None": active.append(f"Quantization ({quant_mode})")
        if use_svd:              active.append(f"SVD (rank={svd_val})")

        if not active:
            st.info("ℹ️ No compression selected — showing baseline metrics. "
                    "Use the sidebar to apply compression techniques.")
            metrics = BASELINE.copy()
        else:
            st.success(f"✅ Active compression: **{' + '.join(active)}**")
            metrics = predict_metrics(pred_models, pruning_val, quant_mode, svd_val)

        # ── Metric cards ─────────────────────────────────────
        st.markdown("### 🔮 Predicted Metrics vs Baseline")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_delta("🎯 Accuracy (%)",
                         metrics["accuracy"], BASELINE["accuracy"],
                         lower_is_better=False, unit="%")
        with col2:
            metric_delta("⚡ GFLOPs",
                         metrics["gflops"], BASELINE["gflops"],
                         lower_is_better=True)
        with col3:
            metric_delta("💾 Model Size (MB)",
                         metrics["model_size_mb"], BASELINE["model_size_mb"],
                         lower_is_better=True, unit=" MB")
        with col4:
            nz_m     = metrics["nonzero_params"] / 1e6
            bl_m     = BASELINE["nonzero_params"] / 1e6
            delta_nz = nz_m - bl_m
            st.metric("🔢 Non-zero Params",
                      value=f"{nz_m:.3f} M",
                      delta=f"{delta_nz:+.3f} M",
                      delta_color="normal" if delta_nz >= 0 else "inverse")

        st.markdown("---")

        # ── Bar chart ─────────────────────────────────────────
        st.markdown("### 📊 Visual Comparison vs Baseline")
        compare_df = pd.DataFrame({
            "Metric":     ["Accuracy (%)", "Model Size (MB)",
                           "GFLOPs", "Non-zero Params (M)"],
            "Baseline":   [BASELINE["accuracy"], BASELINE["model_size_mb"],
                           BASELINE["gflops"], BASELINE["nonzero_params"]/1e6],
            "Compressed": [metrics["accuracy"], metrics["model_size_mb"],
                           metrics["gflops"], metrics["nonzero_params"]/1e6],
        })

        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        fig.patch.set_facecolor("#f8f9ff")
        colors = ["#1565c0", "#e53935"]
        for ax, (_, row) in zip(axes, compare_df.iterrows()):
            bars = ax.bar(["Baseline", "Compressed"],
                          [row["Baseline"], row["Compressed"]],
                          color=colors, width=0.45,
                          edgecolor="white", linewidth=1.5)
            ax.set_title(row["Metric"], fontsize=10, fontweight="bold", pad=8)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() * 1.02,
                        f"{bar.get_height():.3g}",
                        ha="center", va="bottom", fontsize=9, fontweight="600")
            ax.spines[["top","right","left"]].set_visible(False)
            ax.set_ylim(0, row[["Baseline","Compressed"]].max() * 1.25)
            ax.tick_params(left=False)
            ax.set_facecolor("#f8f9ff")
        plt.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close()

        st.markdown("---")

        # ── Image Classification ───────────────────────────────
        st.markdown("### 🖼️ Image Classification with Compressed Model")
        st.markdown(
            "The compressed model (built with your sidebar settings) classifies "
            "the CIFAR-10 test image selected below."
        )

        img_tensor, true_label = dataset[img_index]

        # Display: upscale 32×32 → 224×224 using NEAREST for pixel-art clarity
        img_pil   = transforms.ToPILImage()(img_tensor)
        img_large = img_pil.resize((224, 224), Image.NEAREST)

        col_img, col_result = st.columns([1, 2])

        with col_img:
            st.markdown(f"**Selected Image #{img_index}**")
            st.image(img_large, width=220)
            st.markdown(
                f"<div style='text-align:center; background:#e8f0fe; border-radius:8px;"
                f"padding:8px; font-weight:600; color:#1565c0;'>"
                f"True Label: {CIFAR10_CLASSES[true_label].upper()}</div>",
                unsafe_allow_html=True
            )

        with col_result:
            if run_btn:
                with st.spinner("⏳ Building compressed model and classifying..."):
                    try:
                        compressed = build_compressed_model(
                            baseline_model, pruning_val, quant_mode, svd_val
                        )
                        pred_idx, conf, top5 = classify_image(
                            compressed, img_tensor, quant_mode
                        )
                        pred_label = CIFAR10_CLASSES[pred_idx]
                        is_correct = (pred_idx == true_label)

                        st.markdown("**🤖 Model Prediction**")
                        if is_correct:
                            st.success(
                                f"✅ **{pred_label.upper()}** — "
                                f"{conf:.1f}% confidence  |  Correct!"
                            )
                        else:
                            st.error(
                                f"❌ Predicted: **{pred_label.upper()}** "
                                f"({conf:.1f}% confidence) | "
                                f"True: **{CIFAR10_CLASSES[true_label].upper()}**"
                            )

                        st.markdown("**📊 Top-5 Class Probabilities**")
                        fig2, ax2 = plt.subplots(figsize=(6, 3))
                        fig2.patch.set_facecolor("#f8f9ff")
                        bar_colors = [
                            "#2e7d32" if c == CIFAR10_CLASSES[true_label]
                            else "#1565c0" for c, _ in top5
                        ]
                        bars2 = ax2.barh(
                            [c.capitalize() for c, _ in top5],
                            [p for _, p in top5],
                            color=bar_colors, edgecolor="white",
                            linewidth=1.2, height=0.55
                        )
                        ax2.set_xlabel("Probability (%)", fontsize=9)
                        ax2.invert_yaxis()
                        ax2.spines[["top","right","left"]].set_visible(False)
                        ax2.tick_params(left=False)
                        ax2.set_facecolor("#f8f9ff")
                        for i, (_, p) in enumerate(top5):
                            ax2.text(p + 0.8, i, f"{p:.1f}%",
                                     va="center", fontsize=9, fontweight="600")
                        ax2.set_xlim(0, max(p for _, p in top5) * 1.2)
                        plt.tight_layout()
                        st.pyplot(fig2)
                        plt.close()

                        st.caption(
                            "🟢 Green bar = true class  |  "
                            "🔵 Blue bars = other classes"
                        )

                    except Exception as e:
                        st.error(f"Classification failed: {e}")
                        st.info("Note: INT8 quantization runs on CPU and may be slow.")
            else:
                st.markdown("")
                st.info(
                    "👈 Select your compression settings in the sidebar, "
                    "then press **🚀 Run Prediction & Classify** to see results."
                )

        st.markdown("---")

        # 10 random sample images for reference
        st.markdown("### 🔍 Sample Test Images for Reference")
        st.caption("Click on any index number and enter it in the sidebar to classify that image.")
        sample_indices = random.sample(range(len(dataset)), 10)
        cols = st.columns(10)
        for col, idx in zip(cols, sample_indices):
            t, lbl = dataset[idx]
            pil    = transforms.ToPILImage()(t)
            big    = pil.resize((80, 80), Image.NEAREST)
            col.image(big, caption=f"#{idx}\n{CIFAR10_CLASSES[lbl]}", width=80)

    # ═════════════════════════════════════════════════════════
    # TAB 3 — RESULTS TABLE + TECHNIQUE EXPLANATIONS
    # ═════════════════════════════════════════════════════════
    with tab3:

        # ── Technique explanations ────────────────────────────
        st.markdown("## 🔬 Understanding the Compression Techniques")
        st.markdown(
            "Before exploring the results, here is a plain-language explanation "
            "of the three techniques used in this research."
        )

        tc1, tc2, tc3 = st.columns(3)

        with tc1:
            st.markdown("""
            <div class="tech-card tech-pruning">
                <h3>✂️ Pruning</h3>
                <p>
                Pruning removes unimportant weights from the neural network.
                Weights with the smallest absolute values (L1-norm) contribute
                least to predictions and are set to zero — like trimming dead
                branches from a tree.<br><br>
                <b>In this project:</b> Global unstructured L1 pruning was applied
                at sparsity levels of 30%–70%, followed by 3 epochs of fine-tuning
                to recover accuracy. Higher sparsity = fewer active weights.<br><br>
                <b>Effect:</b> Reduces non-zero parameters significantly while
                keeping model file size the same (zeros still stored in memory).
                </p>
            </div>
            """, unsafe_allow_html=True)

        with tc2:
            st.markdown("""
            <div class="tech-card tech-quantization">
                <h3>🔢 Quantization</h3>
                <p>
                Quantization reduces the numerical precision of the model weights.
                Instead of storing each weight as a 32-bit float (FP32),
                they are stored at lower precision — 16-bit (FP16) or
                8-bit integer (INT8).<br><br>
                <b>In this project:</b> Dynamic quantization was applied to the
                linear (fully connected) layer without any fine-tuning, using
                PyTorch's quantize_dynamic API.<br><br>
                <b>Effect:</b> FP16 halves the model size (44.8 MB → 21.3 MB)
                with no accuracy loss. INT8 targets faster inference on
                compatible hardware.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with tc3:
            st.markdown("""
            <div class="tech-card tech-svd">
                <h3>🔬 SVD (Matrix Factorization)</h3>
                <p>
                Singular Value Decomposition (SVD) decomposes the weight matrix
                of the final classification layer into three smaller matrices,
                keeping only the most important components (top-k singular values).
                This approximates the original layer with fewer parameters.<br><br>
                <b>In this project:</b> SVD was applied to the fc layer at rank
                ratios of 0.3–0.9, replacing the single layer with two smaller
                sequential layers, followed by 3 epochs of fine-tuning.<br><br>
                <b>Effect:</b> Reduces fc layer parameters while the fine-tuning
                step often pushes accuracy above baseline.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Results table ─────────────────────────────────────
        st.markdown("## 📋 All 29 Compression Experiment Results")

        method_filter = st.multiselect(
            "Filter by compression method:",
            options=df_results["method"].unique().tolist(),
            default=df_results["method"].unique().tolist(),
        )

        filtered = df_results[df_results["method"].isin(method_filter)].copy()

        # ── Build 'level' column defensively — works whether or not
        #    the CSV already has it ─────────────────────────────────
        def make_level(row):
            parts = []
            try:
                if float(row.get("pruning", 0)) > 0:
                    parts.append(f'p={row["pruning"]}')
            except (TypeError, ValueError):
                pass
            q = str(row.get("quant", "none")).strip().lower()
            if q not in ("none", "", "nan"):
                parts.append(q.upper())
            try:
                if float(row.get("svd", 0)) > 0:
                    parts.append(f's={row["svd"]}')
            except (TypeError, ValueError):
                pass
            return " | ".join(parts) if parts else "baseline"

        filtered["level"]          = filtered.apply(make_level, axis=1)
        filtered["nonzero_params"] = filtered["nonzero_params"].apply(
            lambda x: f"{int(x):,}"
        )

        st.dataframe(
            filtered[["method", "level", "accuracy", "nonzero_params",
                       "model_size_mb", "gflops"]].rename(columns={
                "method":         "Method",
                "level":          "Level",
                "accuracy":       "Accuracy (%)",
                "nonzero_params": "Non-zero Params",
                "model_size_mb":  "Size (MB)",
                "gflops":         "GFLOPs",
            }),
            use_container_width=True,
            height=480,
        )

        st.markdown("---")

        # ── Accuracy scatter ──────────────────────────────────
        st.markdown("## 📈 Accuracy Across All Experiments")
        st.caption(
            "Each dot is one experiment. The dashed line is the baseline (87.28%). "
            "Points above the line indicate compression improved accuracy."
        )

        fig3, ax3 = plt.subplots(figsize=(13, 5))
        fig3.patch.set_facecolor("#f8f9ff")
        ax3.set_facecolor("#f8f9ff")

        method_colors = {
            "baseline":                 "#607d8b",
            "pruning":                  "#1565c0",
            "quantization":             "#6a1b9a",
            "svd":                      "#00695c",
            "pruning+quantization":     "#e53935",
            "pruning+svd":              "#f57f17",
            "svd+quantization":         "#37474f",
            "pruning+quantization+svd": "#ad1457",
        }
        method_labels = {
            "baseline":                 "Baseline",
            "pruning":                  "Pruning",
            "quantization":             "Quantization",
            "svd":                      "SVD",
            "pruning+quantization":     "Pruning + Quantization",
            "pruning+svd":              "Pruning + SVD",
            "svd+quantization":         "SVD + Quantization",
            "pruning+quantization+svd": "Triple Combination",
        }
        for method, grp in df_results.groupby("method"):
            ax3.scatter(
                range(len(grp)),
                grp["accuracy"],
                label=method_labels.get(method, method),
                color=method_colors.get(method, "gray"),
                s=90, zorder=3, edgecolors="white", linewidth=0.8
            )
        ax3.axhline(87.28, color="#b71c1c", linestyle="--",
                    linewidth=1.5, label="Baseline (87.28%)", zorder=2)
        ax3.set_ylabel("Test Accuracy (%)", fontsize=11)
        ax3.set_xlabel("Experiment Index", fontsize=11)
        ax3.legend(loc="lower right", fontsize=8, ncol=2,
                   framealpha=0.9, edgecolor="#cccccc")
        ax3.spines[["top","right"]].set_visible(False)
        ax3.set_ylim(78, 93)
        ax3.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()


if __name__ == "__main__":
    main()
