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

# ── Prédiction ────────────────────────────────────────────────
def predict(image):
    if image is None:
        return "", "", 0.0

    img    = Image.fromarray(image).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor).squeeze()
        prob  = torch.sigmoid(logit).item()

    confiance = prob if prob >= 0.5 else 1 - prob
    est_fake  = prob < 0.5

    if est_fake:
        badge = f"""
<div style='
    background: linear-gradient(135deg, #fff5f5, #ffe0e0);
    border: 2px solid #ff4d4d;
    border-radius: 16px;
    padding: 24px 32px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(255,77,77,0.15);
'>
    <div style='font-size: 48px; margin-bottom: 8px;'>🚨</div>
    <div style='font-size: 28px; font-weight: 700; color: #cc0000; letter-spacing: 2px;'>FAKE</div>
    <div style='font-size: 14px; color: #888; margin-top: 6px;'>Image générée par StyleGAN</div>
    <div style='
        margin-top: 16px;
        background: #ff4d4d;
        color: white;
        border-radius: 50px;
        padding: 8px 24px;
        font-size: 18px;
        font-weight: 600;
        display: inline-block;
    '>Confiance : {confiance*100:.1f}%</div>
</div>
"""
        details = f"""
<div style='font-family: sans-serif; padding: 8px;'>
    <table style='width:100%; border-collapse: collapse; font-size: 14px;'>
        <tr style='border-bottom: 1px solid #eee;'>
            <td style='padding: 10px; color: #666;'>Probabilité FAKE</td>
            <td style='padding: 10px; font-weight: 600; color: #cc0000;'>{(1-prob)*100:.2f}%</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
            <td style='padding: 10px; color: #666;'>Probabilité REAL</td>
            <td style='padding: 10px; font-weight: 600; color: #333;'>{prob*100:.2f}%</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
            <td style='padding: 10px; color: #666;'>Type détecté</td>
            <td style='padding: 10px; font-weight: 600; color: #333;'>StyleGAN</td>
        </tr>
        <tr>
            <td style='padding: 10px; color: #666;'>Modèle utilisé</td>
            <td style='padding: 10px; font-weight: 600; color: #333;'>FaceDetectorCNN</td>
        </tr>
    </table>
</div>
"""
    else:
        badge = f"""
<div style='
    background: linear-gradient(135deg, #f0fff4, #d4f7e0);
    border: 2px solid #00c853;
    border-radius: 16px;
    padding: 24px 32px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,200,83,0.15);
'>
    <div style='font-size: 48px; margin-bottom: 8px;'>✅</div>
    <div style='font-size: 28px; font-weight: 700; color: #007a33; letter-spacing: 2px;'>REAL</div>
    <div style='font-size: 14px; color: #888; margin-top: 6px;'>Image authentique détectée</div>
    <div style='
        margin-top: 16px;
        background: #00c853;
        color: white;
        border-radius: 50px;
        padding: 8px 24px;
        font-size: 18px;
        font-weight: 600;
        display: inline-block;
    '>Confiance : {confiance*100:.1f}%</div>
</div>
"""
        details = f"""
<div style='font-family: sans-serif; padding: 8px;'>
    <table style='width:100%; border-collapse: collapse; font-size: 14px;'>
        <tr style='border-bottom: 1px solid #eee;'>
            <td style='padding: 10px; color: #666;'>Probabilité REAL</td>
            <td style='padding: 10px; font-weight: 600; color: #007a33;'>{prob*100:.2f}%</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
            <td style='padding: 10px; color: #666;'>Probabilité FAKE</td>
            <td style='padding: 10px; font-weight: 600; color: #333;'>{(1-prob)*100:.2f}%</td>
        </tr>
        <tr style='border-bottom: 1px solid #eee;'>
            <td style='padding: 10px; color: #666;'>Type détecté</td>
            <td style='padding: 10px; font-weight: 600; color: #333;'>Authentique</td>
        </tr>
        <tr>
            <td style='padding: 10px; color: #666;'>Modèle utilisé</td>
            <td style='padding: 10px; font-weight: 600; color: #333;'>FaceDetectorCNN</td>
        </tr>
    </table>
</div>
"""
    return badge, details, float(confiance)


# ── CSS personnalisé ──────────────────────────────────────────
css = """
* { font-family: 'Segoe UI', sans-serif; }

body, .gradio-container {
    background: #f8f9fc !important;
}

.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
}

.header-box {
    background: white;
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 2px 20px rgba(0,0,0,0.06);
    margin-bottom: 24px;
    border: 1px solid #eef0f5;
}

.card {
    background: white !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06) !important;
    border: 1px solid #eef0f5 !important;
    padding: 24px !important;
}

.stats-box {
    background: white;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    border: 1px solid #eef0f5;
}

button.primary {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 14px !important;
    color: white !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(79,70,229,0.3) !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(79,70,229,0.4) !important;
}
"""

# ── Interface ─────────────────────────────────────────────────
with gr.Blocks(css=css, title="AI Face Detector") as demo:

    # Header
    gr.HTML("""
    <div class='header-box'>
        <div style='font-size: 42px; margin-bottom: 12px;'>🔍</div>
        <h1 style='font-size: 28px; font-weight: 700; color: #1a1a2e; margin: 0 0 8px;'>
            Détecteur de Visages IA
        </h1>
        <p style='color: #666; font-size: 15px; margin: 0;'>
            Détectez si une image est <strong>réelle</strong> ou <strong>générée par StyleGAN</strong>
        </p>
    </div>
    """)

    # Stats
    with gr.Row():
        gr.HTML("""
        <div class='stats-box' style='flex:1; margin: 0 8px;'>
            <div style='font-size: 26px; font-weight: 700; color: #4f46e5;'>99.31%</div>
            <div style='font-size: 12px; color: #888; margin-top: 4px;'>Accuracy</div>
        </div>
        """)
        gr.HTML("""
        <div class='stats-box' style='flex:1; margin: 0 8px;'>
            <div style='font-size: 26px; font-weight: 700; color: #4f46e5;'>140K</div>
            <div style='font-size: 12px; color: #888; margin-top: 4px;'>Images d'entraînement</div>
        </div>
        """)
        gr.HTML("""
        <div class='stats-box' style='flex:1; margin: 0 8px;'>
            <div style='font-size: 26px; font-weight: 700; color: #4f46e5;'>0.9999</div>
            <div style='font-size: 12px; color: #888; margin-top: 4px;'>AUC-ROC</div>
        </div>
        """)
        gr.HTML("""
        <div class='stats-box' style='flex:1; margin: 0 8px;'>
            <div style='font-size: 26px; font-weight: 700; color: #4f46e5;'>2.6M</div>
            <div style='font-size: 12px; color: #888; margin-top: 4px;'>Paramètres</div>
        </div>
        """)

    gr.HTML("<div style='height: 20px'></div>")

    # Zone principale
    with gr.Row():
        # Colonne gauche — upload
        with gr.Column(scale=1, elem_classes="card"):
            gr.HTML("<h3 style='margin: 0 0 16px; color: #1a1a2e; font-size: 16px;'>📁 Image à analyser</h3>")
            image_input = gr.Image(
                type="numpy",
                height=320,
                show_label=False,
            )
            btn = gr.Button("🔍 Analyser l'image", variant="primary", size="lg")
            gr.HTML("""
            <p style='font-size: 12px; color: #aaa; text-align: center; margin-top: 12px;'>
                Formats acceptés : JPG, PNG, WEBP
            </p>
            """)

        # Colonne droite — résultats
        with gr.Column(scale=1, elem_classes="card"):
            gr.HTML("<h3 style='margin: 0 0 16px; color: #1a1a2e; font-size: 16px;'>📊 Résultat de l'analyse</h3>")
            badge_output    = gr.HTML("""
            <div style='
                background: #f8f9fc;
                border: 2px dashed #ddd;
                border-radius: 16px;
                padding: 40px;
                text-align: center;
                color: #aaa;
                font-size: 14px;
            '>
                Uploadez une image pour voir le résultat
            </div>
            """)
            gr.HTML("<div style='height: 16px'></div>")
            confiance_slider = gr.Slider(
                minimum=0, maximum=1, value=0,
                label="Niveau de confiance",
                interactive=False
            )
            details_output = gr.HTML()

    # Footer
    gr.HTML("""
    <div style='
        background: white;
        border-radius: 16px;
        padding: 20px 28px;
        margin-top: 20px;
        border: 1px solid #eef0f5;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    '>
        <p style='margin: 0; font-size: 13px; color: #888; text-align: center;'>
            ⚠️ Ce modèle est optimisé pour détecter les visages générés par <strong>StyleGAN</strong> uniquement.
            Les images générées par d'autres outils (MidJourney, DALL·E) peuvent ne pas être détectées.
        </p>
    </div>
    """)

    # Actions
    btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[badge_output, details_output, confiance_slider]
    )
    image_input.change(
        fn=predict,
        inputs=image_input,
        outputs=[badge_output, details_output, confiance_slider]
    )

if __name__ == "__main__":
    demo.launch(share=False)