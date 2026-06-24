import streamlit as st
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO
import time
import pandas as pd
import os

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Waste Classification System",
    layout="wide"
)

classes = ['glass', 'metal', 'plastic']
device = torch.device("cpu")

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

vit_path = os.path.join(BASE_DIR, "vit_finetuned.pth")
yolo_path = os.path.join(BASE_DIR, "best.pt")

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_vit():
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=3
    )

    model.load_state_dict(
        torch.load(vit_path, map_location=device)
    )

    model.eval()

    return model


@st.cache_resource
def load_yolo():
    return YOLO(yolo_path)


vit_model = load_vit()
yolo_model = load_yolo()

# =========================
# TITLE
# =========================
st.title("♻ AI Waste Classification System")
st.markdown("### Comparison between ViT and YOLO")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Model Statistics")

stats = pd.DataFrame({
    "Model": ["ViT", "YOLO"],
    "Accuracy": ["90%", "93%"],
    "F1-score": ["0.89", "0.92"],
    "Latency": ["0.05 sec", "0.02 sec"]
})

st.sidebar.dataframe(stats)

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["Inference", "Comparison"])

# ==================================================
# TAB 1
# ==================================================
with tab1:

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"]
    )

    model_choice = st.selectbox(
        "Choose model",
        ["ViT", "YOLO", "Both"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Input image",
            use_container_width=True
        )

        img = transform(image).unsqueeze(0)

        col1, col2 = st.columns(2)

        # ================= VIT =================
        if model_choice in ["ViT", "Both"]:
            start = time.time()
            with torch.no_grad():
                output = vit_model(img)
                probabilities = torch.softmax(output, dim=1)
            vit_time = time.time() - start
            pred_idx = pred.item()
            vit_conf = conf.item()
            label_vit = classes[pred_idx]
            col1.subheader("🧠 ViT Prediction")
            if vit_conf < 0.5:
                label_vit = "Unknown object"
            col1.success(label_vit)
            col1.metric("Confidence",f"{vit_conf * 100:.2f}%")
            col1.progress(vit_conf)
            col1.info(f"Inference time : {vit_time:.4f} sec")
            top3_probs, top3_idx = torch.topk(probabilities, k=3, dim=1)
            col1.write("### Top 3 Predictions")
            for i in range(3):
                cls = classes[top3_idx[0, i].item()]
                score = top3_probs[0, i].item()
                col1.write(f"• {cls} : {score * 100:.2f}%")
        
        # ================= YOLO =================
        if model_choice in ["YOLO", "Both"]:

            start = time.time()

            results = yolo_model(image)

            probs = results[0].probs

            top1 = int(probs.top1)

            conf = float(probs.top1conf)

            yolo_time = time.time() - start

            if conf < 0.5:
                label_yolo = "Unknown object"
            else:
                label_yolo = classes[top1]

            col2.subheader("YOLO Prediction")

            col2.success(label_yolo)

            col2.info(
                f"Confidence : {conf:.3f}"
            )

            col2.info(
                f"Inference time : {yolo_time:.4f} sec"
            )

# ==================================================
# TAB 2
# ==================================================
with tab2:

    st.subheader("Model Comparison")

    comparison = pd.DataFrame({
        "Model": ["ViT", "YOLO"],
        "Accuracy": [0.90, 0.93],
        "Precision": [0.90, 0.93],
        "Recall": [0.88, 0.92],
        "F1-score": [0.89, 0.92],
        "Latency (sec)": [0.05, 0.02]
    })

    st.dataframe(comparison)

    st.markdown("---")

    st.subheader("Analysis")

    st.write(
        """
        • YOLO is faster and suitable for real-time applications.

        • ViT provides powerful feature extraction.

        • YOLO gives the best speed/accuracy trade-off.

        • ViT is more suitable when classification quality is prioritized.
        """
    )

    st.success(
        "Best overall model : YOLO"
    )
