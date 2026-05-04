
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from src.models.model import build_model

# ── Charger le modèle ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model()
checkpoint = torch.load("output/exp_01.pt", map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

print(f"Modèle chargé sur {device}")

# ── Transform ─────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Fonction de prédiction ────────────────────────────────────
def predict(image):
    if image is None:
        return "⚠️ Veuillez uploader une image", "", 0.0

    img    = Image.fromarray(image).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor).squeeze()
        prob  = torch.sigmoid(logit).item()

    confiance = prob if prob >= 0.5 else 1 - prob
    est_fake  = prob < 0.5

    if est_fake:
        label   = "🚨 FAKE — Image générée par IA (StyleGAN)"
        couleur = "red"
        details = f"""
### Résultat : FAKE
- **Confiance** : {confiance*100:.2f}%
- **Probabilité REAL** : {prob*100:.2f}%
- **Probabilité FAKE** : {(1-prob)*100:.2f}%
- **Type détecté** : StyleGAN
- **Verdict** : Cette image est générée par une IA
        """
    else:
        label   = "✅ REAL — Image authentique"
        couleur = "green"
        details = f"""
### Résultat : REAL
- **Confiance** : {confiance*100:.2f}%
- **Probabilité REAL** : {prob*100:.2f}%
- **Probabilité FAKE** : {(1-prob)*100:.2f}%
- **Verdict** : Cette image semble authentique
        """

    return label, details, float(confiance)


# ── Interface Gradio ──────────────────────────────────────────
with gr.Blocks(title="Détecteur IA — FaceDetectorCNN", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🔍 Détecteur de Visages Générés par IA
    ### Modèle : FaceDetectorCNN — Accuracy : 99.31% sur 20 000 images
    Uploadez une image de visage pour savoir si elle est **réelle** ou **générée par une IA (StyleGAN)**.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📁 Upload une image de visage",
                type="numpy",
                height=350
            )
            btn = gr.Button("🔍 Analyser", variant="primary", size="lg")

        with gr.Column(scale=1):
            label_output = gr.Label(
                label="Prédiction",
                num_top_classes=1
            )
            confiance_output = gr.Slider(
                minimum=0,
                maximum=1,
                label="Confiance du modèle",
                interactive=False
            )
            details_output = gr.Markdown(label="Détails")

    gr.Markdown("---")

    with gr.Row():
        gr.Markdown("""
        ### ℹ️ À propos du modèle
        - **Architecture** : CNN custom (FaceDetectorCNN)
        - **Dataset** : 140 000 images (100K train / 20K valid / 20K test)
        - **Images FAKE** : Générées par StyleGAN uniquement
        - **Accuracy** : 99.31% sur le test set
        - **Limitation** : Optimisé pour StyleGAN — autres générateurs IA moins bien détectés
        """)

    btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[label_output, details_output, confiance_output]
    )

    image_input.change(
        fn=predict,
        inputs=image_input,
        outputs=[label_output, details_output, confiance_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)