import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from src.models.model import build_model
from src.models.model_resnet import build_resnet18

# ── Chargement des deux modèles ───────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# CNN
cnn_model = build_model()
cnn_ckpt  = torch.load("output/exp_01.pt", map_location=device)
cnn_model.load_state_dict(cnn_ckpt["model_state"])
cnn_model.to(device)
cnn_model.eval()

# ResNet18
resnet_model = build_resnet18()
resnet_ckpt  = torch.load("output/exp_02_resnet18.pt", map_location=device)
resnet_model.load_state_dict(resnet_ckpt["model_state"])
resnet_model.to(device)
resnet_model.eval()

print("Les deux modèles sont chargés !")

# ── Transform ─────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Fonction de prédiction ────────────────────────────────────
def predict_both(image):
    if image is None:
        return "", "", ""

    img    = Image.fromarray(image).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    # CNN
    with torch.no_grad():
        cnn_logit = cnn_model(tensor).squeeze()
        cnn_prob  = torch.sigmoid(cnn_logit).item()

    # ResNet18
    with torch.no_grad():
        resnet_logit = resnet_model(tensor).squeeze()
        resnet_prob  = torch.sigmoid(resnet_logit).item()

    # Résultats CNN
    cnn_label     = "REAL" if cnn_prob >= 0.5 else "FAKE"
    cnn_confiance = cnn_prob if cnn_prob >= 0.5 else 1 - cnn_prob
    cnn_color     = "#00c853" if cnn_label == "REAL" else "#ff1744"
    cnn_icon      = "✅" if cnn_label == "REAL" else "🚨"

    # Résultats ResNet18
    resnet_label     = "REAL" if resnet_prob >= 0.5 else "FAKE"
    resnet_confiance = resnet_prob if resnet_prob >= 0.5 else 1 - resnet_prob
    resnet_color     = "#00c853" if resnet_label == "REAL" else "#ff1744"
    resnet_icon      = "✅" if resnet_label == "REAL" else "🚨"

    # Accord entre les deux modèles
    accord = cnn_label == resnet_label
    accord_html = f"""
    <div style='
        background: {"#e8f5e9" if accord else "#fff3e0"};
        border: 2px solid {"#00c853" if accord else "#ff9800"};
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin-top: 8px;
    '>
        <div style='font-size: 24px;'>{"🤝" if accord else "⚠️"}</div>
        <div style='font-weight: 700; font-size: 16px; color: {"#2e7d32" if accord else "#e65100"}; margin-top: 4px;'>
            {"Les deux modèles sont d'accord !" if accord else "Les modèles ne sont pas d'accord !"}
        </div>
        <div style='font-size: 13px; color: #666; margin-top: 4px;'>
            {"Verdict fiable ✓" if accord else "Cas ambigu — faire confiance à ResNet18"}
        </div>
    </div>
    """

    # Card CNN
    cnn_html = f"""
    <div style='
        background: linear-gradient(135deg, #f8f9fc, #ffffff);
        border: 2px solid {cnn_color};
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    '>
        <div style='font-size: 13px; font-weight: 600; color: #888; letter-spacing: 1px; margin-bottom: 8px;'>
            MODÈLE 1
        </div>
        <div style='font-size: 20px; font-weight: 700; color: #1a1a2e; margin-bottom: 4px;'>
            FaceDetectorCNN
        </div>
        <div style='font-size: 12px; color: #aaa; margin-bottom: 16px;'>
            2.6M paramètres • From scratch
        </div>
        <div style='font-size: 40px; margin-bottom: 8px;'>{cnn_icon}</div>
        <div style='font-size: 26px; font-weight: 800; color: {cnn_color};'>{cnn_label}</div>
        <div style='
            margin-top: 12px;
            background: {cnn_color};
            color: white;
            border-radius: 50px;
            padding: 8px 20px;
            font-size: 16px;
            font-weight: 600;
            display: inline-block;
        '>Confiance : {cnn_confiance*100:.1f}%</div>
        <div style='margin-top: 12px; font-size: 13px; color: #888;'>
            P(REAL) = {cnn_prob*100:.2f}% | P(FAKE) = {(1-cnn_prob)*100:.2f}%
        </div>
        <div style='margin-top: 8px; font-size: 12px; color: #aaa;'>
            Accuracy test : 99.31%
        </div>
    </div>
    """

    # Card ResNet18
    resnet_html = f"""
    <div style='
        background: linear-gradient(135deg, #f8f9fc, #ffffff);
        border: 2px solid {resnet_color};
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    '>
        <div style='font-size: 13px; font-weight: 600; color: #888; letter-spacing: 1px; margin-bottom: 8px;'>
            MODÈLE 2
        </div>
        <div style='font-size: 20px; font-weight: 700; color: #1a1a2e; margin-bottom: 4px;'>
            FaceDetectorResNet18
        </div>
        <div style='font-size: 12px; color: #aaa; margin-bottom: 16px;'>
            11M paramètres • Pré-entraîné ImageNet
        </div>
        <div style='font-size: 40px; margin-bottom: 8px;'>{resnet_icon}</div>
        <div style='font-size: 26px; font-weight: 800; color: {resnet_color};'>{resnet_label}</div>
        <div style='
            margin-top: 12px;
            background: {resnet_color};
            color: white;
            border-radius: 50px;
            padding: 8px 20px;
            font-size: 16px;
            font-weight: 600;
            display: inline-block;
        '>Confiance : {resnet_confiance*100:.1f}%</div>
        <div style='margin-top: 12px; font-size: 13px; color: #888;'>
            P(REAL) = {resnet_prob*100:.2f}% | P(FAKE) = {(1-resnet_prob)*100:.2f}%
        </div>
        <div style='margin-top: 8px; font-size: 12px; color: #aaa;'>
            Accuracy test : 99.54%
        </div>
    </div>
    """

    return cnn_html, resnet_html, accord_html


# ── CSS ───────────────────────────────────────────────────────
css = """
* { font-family: 'Segoe UI', sans-serif; }
body, .gradio-container { background: #f8f9fc !important; }
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
button.primary {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(79,70,229,0.3) !important;
}
button.primary:hover { transform: translateY(-2px) !important; }
"""

# ── Interface ─────────────────────────────────────────────────
with gr.Blocks(css=css, title="CNN vs ResNet18 — Comparaison") as demo:

    # Header
    gr.HTML("""
    <div style='
        background: white;
        border-radius: 20px;
        padding: 28px 32px;
        text-align: center;
        box-shadow: 0 2px 20px rgba(0,0,0,0.06);
        margin-bottom: 20px;
        border: 1px solid #eef0f5;
    '>
        <div style='font-size: 36px; margin-bottom: 10px;'>⚔️</div>
        <h1 style='font-size: 26px; font-weight: 700; color: #1a1a2e; margin: 0 0 6px;'>
            CNN vs ResNet18 — Comparaison des modèles
        </h1>
        <p style='color: #666; font-size: 14px; margin: 0;'>
            Uploadez une image et comparez les prédictions des deux modèles en temps réel
        </p>
    </div>
    """)

    # Stats comparatives
    with gr.Row():
        gr.HTML("""
        <div style='flex:1;background:white;border-radius:12px;padding:16px;text-align:center;
                    box-shadow:0 2px 12px rgba(0,0,0,0.05);border:1px solid #eef0f5;margin:0 6px'>
            <div style='font-size:11px;color:#aaa;font-weight:600;letter-spacing:1px'>MODÈLE 1</div>
            <div style='font-size:15px;font-weight:700;color:#1a1a2e;margin:4px 0'>FaceDetectorCNN</div>
            <div style='font-size:22px;font-weight:800;color:#4f46e5'>99.31%</div>
            <div style='font-size:11px;color:#aaa'>Accuracy • 2.6M params</div>
        </div>
        """)
        gr.HTML("""
        <div style='flex:1;background:white;border-radius:12px;padding:16px;text-align:center;
                    box-shadow:0 2px 12px rgba(0,0,0,0.05);border:1px solid #eef0f5;margin:0 6px'>
            <div style='font-size:11px;color:#aaa;font-weight:600;letter-spacing:1px'>MODÈLE 2</div>
            <div style='font-size:15px;font-weight:700;color:#1a1a2e;margin:4px 0'>FaceDetectorResNet18</div>
            <div style='font-size:22px;font-weight:800;color:#7c3aed'>99.54%</div>
            <div style='font-size:11px;color:#aaa'>Accuracy • 11M params</div>
        </div>
        """)
        gr.HTML("""
        <div style='flex:1;background:white;border-radius:12px;padding:16px;text-align:center;
                    box-shadow:0 2px 12px rgba(0,0,0,0.05);border:1px solid #eef0f5;margin:0 6px'>
            <div style='font-size:11px;color:#aaa;font-weight:600;letter-spacing:1px'>DATASET</div>
            <div style='font-size:15px;font-weight:700;color:#1a1a2e;margin:4px 0'>140K Images</div>
            <div style='font-size:22px;font-weight:800;color:#059669'>StyleGAN</div>
            <div style='font-size:11px;color:#aaa'>REAL vs FAKE</div>
        </div>
        """)

    gr.HTML("<div style='height:16px'></div>")

    # Zone principale
    with gr.Row():
        # Upload
        with gr.Column(scale=1):
            gr.HTML("""
            <div style='background:white;border-radius:16px;padding:20px;
                        box-shadow:0 2px 16px rgba(0,0,0,0.06);border:1px solid #eef0f5;'>
                <h3 style='margin:0 0 16px;color:#1a1a2e;font-size:15px;'>📁 Image à analyser</h3>
            """)
            image_input = gr.Image(type="numpy", height=300, show_label=False)
            btn = gr.Button("⚔️ Comparer les deux modèles", variant="primary", size="lg")
            gr.HTML("""
                <p style='font-size:12px;color:#aaa;text-align:center;margin-top:10px;'>
                    JPG, PNG, WEBP acceptés
                </p>
            </div>
            """)

        # Résultats
        with gr.Column(scale=2):
            with gr.Row():
                cnn_output    = gr.HTML()
                resnet_output = gr.HTML()
            accord_output = gr.HTML()

    # Footer
    gr.HTML("""
    <div style='background:white;border-radius:16px;padding:16px 24px;margin-top:16px;
                border:1px solid #eef0f5;box-shadow:0 2px 12px rgba(0,0,0,0.04);'>
        <p style='margin:0;font-size:12px;color:#888;text-align:center;'>
            ⚠️ Les deux modèles sont optimisés pour détecter les visages générés par
            <strong>StyleGAN v1</strong> uniquement.
        </p>
    </div>
    """)

    btn.click(
        fn=predict_both,
        inputs=image_input,
        outputs=[cnn_output, resnet_output, accord_output]
    )
    image_input.change(
        fn=predict_both,
        inputs=image_input,
        outputs=[cnn_output, resnet_output, accord_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)