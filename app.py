"""
CNN Compression Predictor + CIFAR-10 Image Classifier
Streamlit App — ResNet18 on CIFAR-10
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle, json, os, copy, random
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
        file_id = "1wxux74QrYPjTTJ7QZ3QElmKFhm2nEerc"   # ← replace with your file ID
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
    tf = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=tf
    )
    return dataset

@st.cache_data
def load_results_df():
    return pd.read_csv("compression_data.csv")

# ─────────────────────────────────────────────────────────────
# COMPRESSION FUNCTIONS
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
# CLASSIFY IMAGE
# ─────────────────────────────────────────────────────────────
def classify_image(model, img_tensor, quant_mode):
    norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    inp  = norm(img_tensor).unsqueeze(0)
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
    if lower_is_better:
        color = "normal" if delta <= 0 else "inverse"
    else:
        color = "normal" if delta >= 0 else "inverse"
    st.metric(label=label,
              value=f"{value:,.4g}{unit}",
              delta=delta_str,
              delta_color=color)


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():
    st.title("🧠 CNN Compression Predictor")
    st.markdown(
        "**ResNet18 on CIFAR-10** — Select compression techniques to predict "
        "model metrics and classify a CIFAR-10 image with the compressed model."
    )
    st.divider()

    # ── Load assets ──────────────────────────────────────────
    try:
        pred_models    = load_predictive_models()
        baseline_model = load_baseline_model()
        dataset        = load_cifar10_test()
        df_results     = load_results_df()
        assets_ok      = True
    except Exception as e:
        st.error(f"❌ Failed to load assets: {e}")
        st.info("Make sure baseline.pth, compression_models.pkl, "
                "and compression_data.csv are in the same folder as app.py")
        assets_ok = False

    if not assets_ok:
        return

    # ── Sidebar ───────────────────────────────────────────────
    st.sidebar.header("⚙️ Compression Settings")

    st.sidebar.subheader("1. Pruning")
    use_pruning  = st.sidebar.checkbox("Apply Pruning", value=False)
    pruning_rate = st.sidebar.slider("Sparsity", 0.3, 0.7, 0.5, 0.1,
                                     disabled=not use_pruning)
    pruning_val  = pruning_rate if use_pruning else 0.0

    st.sidebar.subheader("2. Quantization")
    quant_mode = st.sidebar.selectbox("Precision", ["None", "FP16", "INT8"])

    st.sidebar.subheader("3. SVD")
    use_svd   = st.sidebar.checkbox("Apply SVD", value=False)
    svd_ratio = st.sidebar.slider("Rank Ratio", 0.3, 0.9, 0.7, 0.1,
                                  disabled=not use_svd)
    svd_val   = svd_ratio if use_svd else 0.0

    st.sidebar.divider()
    st.sidebar.subheader("🖼️ Image Selection")
    img_index  = st.sidebar.number_input(
        "CIFAR-10 Test Image Index (0–9999)",
        min_value=0, max_value=9999, value=42, step=1
    )
    if st.sidebar.button("🎲 Pick Random Image"):
        img_index = random.randint(0, 9999)
        st.sidebar.success(f"Random index: {img_index}")

    run_btn = st.sidebar.button("🚀 Run Prediction", type="primary",
                                use_container_width=True)

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📊 Metric Prediction",
        "🖼️ Image Classification",
        "📋 Full Results Table"
    ])

    # ─────────────────────────────────────────────────────────
    # TAB 1 — METRIC PREDICTION
    # ─────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Predicted Compression Metrics")

        active = []
        if use_pruning:          active.append(f"Pruning ({pruning_val})")
        if quant_mode != "None": active.append(f"Quantization ({quant_mode})")
        if use_svd:              active.append(f"SVD ({svd_val})")

        if not active:
            st.info("No compression selected — showing baseline metrics.")
            metrics = BASELINE.copy()
        else:
            st.markdown(f"**Active:** {' + '.join(active)}")
            metrics = predict_metrics(pred_models, pruning_val, quant_mode, svd_val)

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
                      value=f"{nz_m:.3f}M",
                      delta=f"{delta_nz:+.3f}M",
                      delta_color="normal" if delta_nz >= 0 else "inverse")

        st.divider()

        st.subheader("📊 Comparison vs Baseline")
        compare_df = pd.DataFrame({
            "Metric":     ["Accuracy (%)", "Model Size (MB)", "GFLOPs",
                           "Non-zero Params (M)"],
            "Baseline":   [BASELINE["accuracy"], BASELINE["model_size_mb"],
                           BASELINE["gflops"], BASELINE["nonzero_params"]/1e6],
            "Compressed": [metrics["accuracy"], metrics["model_size_mb"],
                           metrics["gflops"], metrics["nonzero_params"]/1e6],
        })

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        colors = ["#4C72B0", "#DD8452"]
        for ax, (_, row) in zip(axes, compare_df.iterrows()):
            bars = ax.bar(["Baseline", "Compressed"],
                          [row["Baseline"], row["Compressed"]],
                          color=colors, width=0.5, edgecolor="white")
            ax.set_title(row["Metric"], fontsize=10, fontweight="bold")
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() * 1.01,
                        f"{bar.get_height():.3g}",
                        ha="center", va="bottom", fontsize=9)
            ax.spines[["top","right"]].set_visible(False)
            ax.set_ylim(0, row[["Baseline","Compressed"]].max() * 1.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ─────────────────────────────────────────────────────────
    # TAB 2 — IMAGE CLASSIFICATION
    # ─────────────────────────────────────────────────────────
    with tab2:
        st.subheader("CIFAR-10 Image Classification")
        st.markdown(
            "The compressed model (with your selected settings) classifies "
            "the chosen CIFAR-10 test image."
        )

        img_tensor, true_label = dataset[img_index]
        img_pil   = transforms.ToPILImage()(img_tensor)
        img_large = img_pil.resize((224, 224), Image.NEAREST)

        col_img, col_result = st.columns([1, 2])

        with col_img:
            st.image(img_large,
                     caption=f"Image #{img_index} | True: {CIFAR10_CLASSES[true_label]}",
                     width=224)

        with col_result:
            if run_btn:
                with st.spinner("Building compressed model and classifying..."):
                    try:
                        compressed = build_compressed_model(
                            baseline_model, pruning_val, quant_mode, svd_val
                        )
                        pred_idx, conf, top5 = classify_image(
                            compressed, img_tensor, quant_mode
                        )
                        pred_label = CIFAR10_CLASSES[pred_idx]
                        is_correct = (pred_idx == true_label)

                        if is_correct:
                            st.success(
                                f"✅ Predicted: **{pred_label}** "
                                f"({conf:.1f}% confidence)"
                            )
                        else:
                            st.error(
                                f"❌ Predicted: **{pred_label}** "
                                f"({conf:.1f}% confidence) | "
                                f"True: **{CIFAR10_CLASSES[true_label]}**"
                            )

                        st.markdown("**Top-5 Class Probabilities:**")
                        fig2, ax2 = plt.subplots(figsize=(6, 3))
                        bar_colors = [
                            "#2ecc71" if c == CIFAR10_CLASSES[true_label]
                            else "#4C72B0" for c, _ in top5
                        ]
                        ax2.barh([c for c, _ in top5],
                                 [p for _, p in top5],
                                 color=bar_colors, edgecolor="white")
                        ax2.set_xlabel("Probability (%)")
                        ax2.invert_yaxis()
                        ax2.spines[["top","right"]].set_visible(False)
                        for i, (_, p) in enumerate(top5):
                            ax2.text(p + 0.5, i, f"{p:.1f}%",
                                     va="center", fontsize=9)
                        plt.tight_layout()
                        st.pyplot(fig2)
                        plt.close()

                    except Exception as e:
                        st.error(f"Classification failed: {e}")
                        st.info("Note: INT8 quantization runs on CPU and may be slow.")
            else:
                st.info(
                    "👈 Configure compression settings and press "
                    "**Run Prediction** to classify the image."
                )

        st.divider()

        st.subheader("🔍 Sample CIFAR-10 Test Images")
        sample_indices = random.sample(range(len(dataset)), 10)
        cols = st.columns(10)
        for col, idx in zip(cols, sample_indices):
            t, lbl = dataset[idx]
            pil    = transforms.ToPILImage()(t)
            big    = pil.resize((80, 80), Image.NEAREST)
            col.image(big, caption=f"#{idx}\n{CIFAR10_CLASSES[lbl]}", width=80)

    # ─────────────────────────────────────────────────────────
    # TAB 3 — FULL RESULTS TABLE
    # ─────────────────────────────────────────────────────────
    with tab3:
        st.subheader("📋 All Compression Experiment Results")

        method_filter = st.multiselect(
            "Filter by method:",
            options=df_results["method"].unique().tolist(),
            default=df_results["method"].unique().tolist(),
        )

        filtered = df_results[df_results["method"].isin(method_filter)].copy()
        filtered["nonzero_params"] = filtered["nonzero_params"].apply(
            lambda x: f"{int(x):,}"
        )

        # ── safely select only columns that exist in the CSV ──
        desired_cols = ["method", "accuracy", "nonzero_params",
                        "model_size_mb", "gflops"]
        rename_map = {
            "method":         "Method",
            "accuracy":       "Accuracy (%)",
            "nonzero_params": "Non-zero Params",
            "model_size_mb":  "Size (MB)",
            "gflops":         "GFLOPs",
        }
        # Only keep columns that are actually present in the dataframe
        available_cols = [c for c in desired_cols if c in filtered.columns]
        available_rename = {k: v for k, v in rename_map.items() if k in available_cols}

        st.dataframe(
            filtered[available_cols].rename(columns=available_rename),
            use_container_width=True,
            height=500,
        )

        st.subheader("Accuracy Across All Experiments")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        method_colors = {
            "baseline":                 "#888888",
            "pruning":                  "#4C72B0",
            "quantization":             "#DD8452",
            "svd":                      "#55A868",
            "pruning+quantization":     "#C44E52",
            "pruning+svd":              "#8172B3",
            "svd+quantization":         "#937860",
            "pruning+quantization+svd": "#DA8BC3",
        }
        for method, grp in df_results.groupby("method"):
            ax3.scatter(
                range(len(grp)),
                grp["accuracy"],        # ← correct column name
                label=method,
                color=method_colors.get(method, "gray"),
                s=80, zorder=3
            )
        ax3.axhline(87.28, color="black", linestyle="--",
                    linewidth=1, label="Baseline (87.28%)")
        ax3.set_ylabel("Accuracy (%)")
        ax3.set_xlabel("Experiment")
        ax3.legend(loc="lower right", fontsize=8, ncol=2)
        ax3.spines[["top","right"]].set_visible(False)
        ax3.set_ylim(78, 93)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()


if __name__ == "__main__":
    main()

