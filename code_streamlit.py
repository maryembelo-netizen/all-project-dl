import streamlit as st
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO
import time
import pandas as pd
import os
st.set_page_config(
    page_title="AI Waste Classification System",
    layout="wide"
)
classes = ['glass', 'metal', 'plastic']
device = torch.device("cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
vit_path = os.path.join(BASE_DIR, "vit_finetuned.pth")
yolo_path = os.path.join(BASE_DIR, "best.pt")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])

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

 
st.title("♻ AI Waste Classification System (glass,plastic,metal)")
st.markdown("### Comparison between ViT and YOLO")

st.sidebar.header("Model Statistics")

stats = pd.DataFrame({
    "Model": ["ViT", "YOLO"],
    "Accuracy": ["90%", "93%"],
    "F1-score": ["0.89", "0.92"],
    "Latency": ["0.05 sec", "0.02 sec"]
})

st.sidebar.dataframe(stats)

tab1, tab2 = st.tabs(["Inference", "Comparison"])


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
                # Application de Softmax pour extraire les probabilités de confiance
                probabilities = torch.nn.functional.softmax(output, dim=1)
                conf_vit, pred = torch.max(probabilities, 1)

            vit_time = time.time() - start
            label_vit = classes[pred.item()]
            confidence_score_vit = conf_vit.item()

            col1.subheader("ViT Prediction")
            col1.success(label_vit)

            # Affichage de la confiance pour ViT aligné sur YOLO
            col1.info(
                f"Confidence : {confidence_score_vit:.3f}"
            )
            col1.info(
                f"Inference time : {vit_time:.4f} sec"
            )

            # ================= YOLO =================
        if model_choice in ["YOLO", "Both"]:

            start = time.time()
            results = yolo_model(image)
            probs = results[0].probs
            top1 = int(probs.top1)
            conf_yolo = float(probs.top1conf)
            yolo_time = time.time() - start

            if conf_yolo < 0.5:
                label_yolo = "Unknown object"
            else:
                label_yolo = classes[top1]

            col2.subheader("YOLO Prediction")
            col2.success(label_yolo)
            col2.info(
                f"Confidence : {conf_yolo:.3f}"
            )
            col2.info(
                f"Inference time : {yolo_time:.4f} sec"
            )

   
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

