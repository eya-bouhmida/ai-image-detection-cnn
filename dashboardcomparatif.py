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

cnn_model = build_model()
cnn_ckpt  = torch.load("output/exp_01.pt", map_location=device)
cnn_model.load_state_dict(cnn_ckpt["model_state"])
cnn_model.to(device)
cnn_model.eval()

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

# ── Palette ───────────────────────────────────────────────────
BG       = "#dbeafe"
CARD2    = "#ffffff"
BORDER   = "#bfdbfe"
ACCENT   = "#0f172a"
TEXT     = "#0f172a"
MUTED    = "#3b82f6"
REAL_COL = "#16a34a"
FAKE_COL = "#dc2626"

# ── Fonction de prédiction ────────────────────────────────────
def predict_both(image):
    if image is None:
        return "", "", ""

    img    = Image.fromarray(image).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        cnn_prob = torch.sigmoid(cnn_model(tensor).squeeze()).item()
    with torch.no_grad():
        resnet_prob = torch.sigmoid(resnet_model(tensor).squeeze()).item()

    cnn_label     = "REAL" if cnn_prob >= 0.5 else "FAKE"
    cnn_confiance = cnn_prob if cnn_prob >= 0.5 else 1 - cnn_prob
    cnn_color     = REAL_COL if cnn_label == "REAL" else FAKE_COL
    cnn_icon      = "✦" if cnn_label == "REAL" else "✕"

    resnet_label     = "REAL" if resnet_prob >= 0.5 else "FAKE"
    resnet_confiance = resnet_prob if resnet_prob >= 0.5 else 1 - resnet_prob
    resnet_color     = REAL_COL if resnet_label == "REAL" else FAKE_COL
    resnet_icon      = "✦" if resnet_label == "REAL" else "✕"

    accord     = cnn_label == resnet_label
    accord_col = "#16a34a" if accord else "#d97706"

    accord_html = f"""
<div style='display:flex;align-items:center;gap:14px;
    background:{CARD2};border:1px solid {BORDER};
    border-left:4px solid {accord_col};
    border-radius:10px;padding:14px 18px;margin-top:12px;
    font-family:Inter,sans-serif;'>
    <div style='width:34px;height:34px;border-radius:50%;flex-shrink:0;
        background:{ACCENT};color:#dbeafe;font-size:16px;font-weight:700;
        display:flex;align-items:center;justify-content:center;'>
        {"✓" if accord else "!"}
    </div>
    <div>
        <div style='font-size:13px;font-weight:700;color:{TEXT};margin-bottom:3px;'>
            {"Les deux modèles sont d'accord" if accord else "Les modèles divergent"}
        </div>
        <div style='font-size:11px;color:#64748b;'>
            {"Verdict fiable — les deux pointent vers " + cnn_label + "." if accord else "Cas ambigu — privilégier ResNet18."}
        </div>
    </div>
</div>"""

    def make_card(label, confiance, prob, color, icon, model_num, model_name, model_sub, acc):
        return f"""
<div style='background:{CARD2};border:1px solid {BORDER};
    border-top:4px solid {ACCENT};border-radius:12px;padding:22px;
    font-family:Inter,sans-serif;'>
    <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:18px;'>
        <div>
            <div style='font-size:10px;font-weight:700;color:{MUTED};
                letter-spacing:.1em;text-transform:uppercase;margin-bottom:5px;'>Modèle {model_num}</div>
            <div style='font-size:15px;font-weight:700;color:{TEXT};margin-bottom:3px;'>{model_name}</div>
            <div style='font-size:11px;color:#64748b;'>{model_sub}</div>
        </div>
        <div style='background:{color}18;border:1px solid {color}80;color:{color};
            font-size:11px;font-weight:700;padding:5px 13px;
            border-radius:20px;letter-spacing:.05em;'>
            {icon} {label}
        </div>
    </div>
    <div style='margin-bottom:18px;'>
        <div style='font-size:50px;font-weight:800;color:{ACCENT};line-height:1;letter-spacing:-2px;'>
            {confiance*100:.1f}<span style='font-size:20px;font-weight:600;color:{MUTED};'>%</span>
        </div>
        <div style='font-size:11px;color:#64748b;margin-top:4px;'>indice de confiance</div>
    </div>
    <div style='margin-bottom:18px;'>
        <div style='display:flex;justify-content:space-between;font-size:10px;color:#64748b;margin-bottom:6px;'>
            <span>REAL &nbsp;{prob*100:.1f}%</span>
            <span>FAKE &nbsp;{(1-prob)*100:.1f}%</span>
        </div>
        <div style='background:{BG};border-radius:4px;height:6px;overflow:hidden;'>
            <div style='width:{prob*100:.1f}%;height:100%;background:{color};border-radius:4px;'></div>
        </div>
    </div>
    <div style='display:flex;justify-content:space-between;align-items:center;
        padding-top:14px;border-top:1px solid {BORDER};font-size:11px;color:#64748b;'>
        <span>Accuracy sur le test set</span>
        <span style='font-size:13px;font-weight:700;color:{ACCENT};'>{acc}</span>
    </div>
</div>"""

    cnn_html    = make_card(cnn_label, cnn_confiance, cnn_prob, cnn_color, cnn_icon,
                            "1", "FaceDetectorCNN", "2.6M params · from scratch", "99.31%")
    resnet_html = make_card(resnet_label, resnet_confiance, resnet_prob, resnet_color, resnet_icon,
                            "2", "FaceDetectorResNet18", "11M params · pré-entraîné ImageNet", "99.54%")

    return cnn_html, resnet_html, accord_html


# ── CSS ───────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }
body, .gradio-container { background: #dbeafe !important; }
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; padding: 32px 24px !important; }

.tabs > .tab-nav {
    background: #ffffff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 10px !important;
    padding: 4px !important;
    margin-bottom: 20px !important;
}
.tabs > .tab-nav > button {
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #3b82f6 !important;
    padding: 8px 22px !important;
    border: none !important;
    background: transparent !important;
}
.tabs > .tab-nav > button.selected {
    background: #0f172a !important;
    color: #dbeafe !important;
}

button.lg.primary {
    background: #0f172a !important; border: none !important;
    border-radius: 10px !important; font-size: 14px !important;
    font-weight: 700 !important; color: #dbeafe !important;
    letter-spacing: .03em !important; box-shadow: none !important;
}
button.lg.primary:hover { background: #1e293b !important; transform: none !important; }
"""

HEADER_HTML = f"""
<div style='margin-bottom:28px;font-family:Inter,sans-serif;'>
    <div style='display:flex;align-items:center;gap:14px;'>
        <div style='width:44px;height:44px;border-radius:10px;
            background:{ACCENT};color:#dbeafe;font-size:22px;font-weight:800;
            display:flex;align-items:center;justify-content:center;'>⚡</div>
        <div>
            <h1 style='font-size:22px;font-weight:800;color:{TEXT};margin:0 0 3px;'>
                Deepfake Face Detector
            </h1>
            <p style='font-size:12px;color:{MUTED};margin:0;'>
                CNN from-scratch vs ResNet18 pré-entraîné &nbsp;·&nbsp; StyleGAN v1 &nbsp;·&nbsp; 140k images
            </p>
        </div>
    </div>
</div>"""

STATS_HTML = f"""
<div style='display:flex;gap:10px;margin-bottom:20px;font-family:Inter,sans-serif;'>
    <div style='flex:1;background:{CARD2};border:1px solid {BORDER};border-radius:10px;padding:16px 18px;'>
        <div style='font-size:10px;font-weight:700;color:{MUTED};letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;'>Modèle 1</div>
        <div style='font-size:14px;font-weight:700;color:{TEXT};margin-bottom:2px;'>FaceDetectorCNN</div>
        <div style='font-size:26px;font-weight:800;color:{ACCENT};line-height:1;margin-bottom:4px;'>99.31%</div>
        <div style='font-size:11px;color:#64748b;'>Accuracy &nbsp;·&nbsp; 2.6M params</div>
    </div>
    <div style='flex:1;background:{CARD2};border:1px solid {BORDER};border-radius:10px;padding:16px 18px;'>
        <div style='font-size:10px;font-weight:700;color:{MUTED};letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;'>Modèle 2</div>
        <div style='font-size:14px;font-weight:700;color:{TEXT};margin-bottom:2px;'>FaceDetectorResNet18</div>
        <div style='font-size:26px;font-weight:800;color:{ACCENT};line-height:1;margin-bottom:4px;'>99.54%</div>
        <div style='font-size:11px;color:#64748b;'>Accuracy &nbsp;·&nbsp; 11M params</div>
    </div>
    <div style='flex:1;background:{CARD2};border:1px solid {BORDER};border-radius:10px;padding:16px 18px;'>
        <div style='font-size:10px;font-weight:700;color:{MUTED};letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;'>Dataset</div>
        <div style='font-size:14px;font-weight:700;color:{TEXT};margin-bottom:2px;'>140 000 images</div>
        <div style='font-size:26px;font-weight:800;color:{ACCENT};line-height:1;margin-bottom:4px;'>StyleGAN</div>
        <div style='font-size:11px;color:#64748b;'>REAL vs FAKE</div>
    </div>
</div>"""

# ── Interface ─────────────────────────────────────────────────
with gr.Blocks(title="Deepfake Face Detector") as demo:

    gr.HTML(HEADER_HTML)
    gr.HTML(STATS_HTML)

    with gr.Tabs():

        # ── Onglet 1 — Détection ──────────────────────────
        with gr.Tab("🔍  Détection"):

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML(f"""
<div style='background:{CARD2};border:1px solid {BORDER};border-radius:12px;
    padding:16px 16px 10px;margin-bottom:12px;font-family:Inter,sans-serif;'>
    <div style='font-size:12px;font-weight:700;color:{ACCENT};margin-bottom:12px;'>
        📁 Image à analyser
    </div>""")
                    image_input = gr.Image(type="numpy", height=290, show_label=False)
                    gr.HTML("</div>")
                    btn = gr.Button("Lancer l'analyse →", variant="primary", size="lg")
                    gr.HTML(f"""
<p style='font-size:11px;color:#64748b;text-align:center;margin-top:10px;font-family:Inter,sans-serif;'>
    JPG &nbsp;·&nbsp; PNG &nbsp;·&nbsp; WEBP acceptés
</p>""")

                with gr.Column(scale=2):
                    with gr.Row():
                        cnn_output    = gr.HTML()
                        resnet_output = gr.HTML()
                    accord_output = gr.HTML()

            gr.HTML(f"""
<div style='margin-top:22px;padding:13px 20px;background:{CARD2};
    border:1px solid {BORDER};border-radius:10px;
    font-size:12px;color:#64748b;font-family:Inter,sans-serif;
    display:flex;align-items:center;gap:10px;'>
    <span>⚠️</span>
    <span>Optimisé pour les visages générés par
        <strong style='color:{ACCENT};'>StyleGAN v1</strong> uniquement.
    </span>
</div>""")

        # ── Onglet 2 — Performances CNN ───────────────────
        with gr.Tab("📊  Performances CNN"):

            gr.HTML(f"""
<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px;
    margin-bottom:24px;font-family:Inter,sans-serif;'>
    <div style='background:{CARD2};border:1px solid {BORDER};border-top:3px solid {ACCENT};
        border-radius:10px;padding:14px 16px;text-align:center;'>
        <div style='font-size:10px;font-weight:700;color:{MUTED};letter-spacing:.08em;
            text-transform:uppercase;margin-bottom:6px;'>Accuracy</div>
        <div style='font-size:28px;font-weight:800;color:{ACCENT};'>99.31%</div>
        <div style='font-size:11px;color:#64748b;margin-top:2px;'>Test set</div>
    </div>
    <div style='background:{CARD2};border:1px solid {BORDER};border-top:3px solid {MUTED};
        border-radius:10px;padding:14px 16px;text-align:center;'>
        <div style='font-size:10px;font-weight:700;color:{MUTED};letter-spacing:.08em;
            text-transform:uppercase;margin-bottom:6px;'>AUC-ROC</div>
        <div style='font-size:28px;font-weight:800;color:{ACCENT};'>0.9995</div>
        <div style='font-size:11px;color:#64748b;margin-top:2px;'>Quasi-parfait</div>
    </div>
    <div style='background:{CARD2};border:1px solid {BORDER};border-top:3px solid {REAL_COL};
        border-radius:10px;padding:14px 16px;text-align:center;'>
        <div style='font-size:10px;font-weight:700;color:{MUTED};letter-spacing:.08em;
            text-transform:uppercase;margin-bottom:6px;'>F1-score</div>
        <div style='font-size:28px;font-weight:800;color:{ACCENT};'>99.32%</div>
        <div style='font-size:11px;color:#64748b;margin-top:2px;'>Classe FAKE</div>
    </div>
    <div style='background:{CARD2};border:1px solid {BORDER};border-top:3px solid #d97706;
        border-radius:10px;padding:14px 16px;text-align:center;'>
        <div style='font-size:10px;font-weight:700;color:{MUTED};letter-spacing:.08em;
            text-transform:uppercase;margin-bottom:6px;'>Époques</div>
        <div style='font-size:28px;font-weight:800;color:{ACCENT};'>30</div>
        <div style='font-size:11px;color:#64748b;margin-top:2px;'>From scratch</div>
    </div>
</div>""")

            gr.HTML(f"""
<div style='background:{CARD2};border:1px solid {BORDER};border-radius:12px;
    padding:20px;margin-bottom:16px;font-family:Inter,sans-serif;'>
    <div style='font-size:13px;font-weight:700;color:{ACCENT};margin-bottom:14px;'>
        📈 Courbes d'entraînement — Loss &amp; Accuracy
    </div>""")
            gr.Image(value="output/visualisations_finales.png", show_label=False, height=480)
            gr.HTML("</div>")

            gr.HTML(f"""
<div style='background:{CARD2};border:1px solid {BORDER};border-radius:12px;
    padding:20px;margin-bottom:16px;font-family:Inter,sans-serif;'>
    <div style='font-size:13px;font-weight:700;color:{ACCENT};margin-bottom:14px;'>
        🔲 Matrice de confusion — 20 000 images test
    </div>""")
            gr.Image(value="output/confusion_matrix.png", show_label=False, height=420)
            gr.HTML("</div>")

            gr.HTML(f"""
<div style='background:{CARD2};border:1px solid {BORDER};border-radius:12px;
    padding:20px;margin-bottom:16px;font-family:Inter,sans-serif;'>
    <div style='font-size:13px;font-weight:700;color:{ACCENT};margin-bottom:14px;'>
        📋 Évaluation complète — ROC, métriques, distributions
    </div>""")
            gr.Image(value="output/evaluation_complete.png", show_label=False, height=520)
            gr.HTML("</div>")

            gr.HTML(f"""
<div style='background:{CARD2};border:1px solid {BORDER};border-radius:12px;
    padding:20px;margin-bottom:8px;font-family:Inter,sans-serif;'>
    <div style='font-size:13px;font-weight:700;color:{ACCENT};margin-bottom:16px;'>
        📊 Résumé des métriques
    </div>
    <table style='width:100%;border-collapse:collapse;font-size:13px;'>
        <thead>
            <tr style='background:{ACCENT};'>
                <th style='padding:10px 14px;text-align:left;color:#dbeafe;font-weight:600;'>Métrique</th>
                <th style='padding:10px 14px;text-align:center;color:#dbeafe;font-weight:600;'>Train</th>
                <th style='padding:10px 14px;text-align:center;color:#dbeafe;font-weight:600;'>Test</th>
            </tr>
        </thead>
        <tbody>
            <tr style='background:#f8fafc;'>
                <td style='padding:9px 14px;color:{TEXT};border-bottom:1px solid {BORDER};'>Accuracy finale</td>
                <td style='padding:9px 14px;text-align:center;font-weight:700;color:{ACCENT};border-bottom:1px solid {BORDER};'>99.86%</td>
                <td style='padding:9px 14px;text-align:center;font-weight:700;color:{REAL_COL};border-bottom:1px solid {BORDER};'>99.31%</td>
            </tr>
            <tr>
                <td style='padding:9px 14px;color:{TEXT};border-bottom:1px solid {BORDER};'>Meilleure accuracy</td>
                <td style='padding:9px 14px;text-align:center;font-weight:700;color:{ACCENT};border-bottom:1px solid {BORDER};'>99.86%</td>
                <td style='padding:9px 14px;text-align:center;font-weight:700;color:{REAL_COL};border-bottom:1px solid {BORDER};'>99.33%</td>
            </tr>
            <tr style='background:#f8fafc;'>
                <td style='padding:9px 14px;color:{TEXT};border-bottom:1px solid {BORDER};'>Loss finale</td>
                <td style='padding:9px 14px;text-align:center;font-weight:700;color:{ACCENT};border-bottom:1px solid {BORDER};'>0.0044</td>
                <td style='padding:9px 14px;text-align:center;font-weight:700;color:{ACCENT};border-bottom:1px solid {BORDER};'>0.0264</td>
            </tr>
            <tr>
                <td style='padding:9px 14px;color:{TEXT};border-bottom:1px solid {BORDER};'>AUC-ROC</td>
                <td style='padding:9px 14px;text-align:center;font-weight:700;color:{ACCENT};border-bottom:1px solid {BORDER};'>—</td>
                <td style='padding:9px 14px;text-align:center;font-weight:700;color:{MUTED};border-bottom:1px solid {BORDER};'>0.9995</td>
            </tr>
            <tr style='background:#f8fafc;'>
                <td style='padding:9px 14px;color:{TEXT};border-bottom:1px solid {BORDER};'>Precision FAKE</td>
                <td style='padding:9px 14px;text-align:center;font-weight:700;color:{ACCENT};border-bottom:1px solid {BORDER};'>—</td>
                <td style='padding:9px 14px;text-align:center;font-weight:700;color:{ACCENT};border-bottom:1px solid {BORDER};'>99.27%</td>
            </tr>
            <tr>
                <td style='padding:9px 14px;color:{TEXT};border-bottom:1px solid {BORDER};'>Recall FAKE</td>
                <td style='padding:9px 14px;text-align:center;font-weight:700;color:{ACCENT};border-bottom:1px solid {BORDER};'>—</td>
                <td style='padding:9px 14px;text-align:center;font-weight:700;color:{ACCENT};border-bottom:1px solid {BORDER};'>99.36%</td>
            </tr>
            <tr style='background:#f8fafc;'>
                <td style='padding:9px 14px;color:{TEXT};'>F1-score FAKE</td>
                <td style='padding:9px 14px;text-align:center;font-weight:700;color:{ACCENT};'>—</td>
                <td style='padding:9px 14px;text-align:center;font-weight:700;color:{REAL_COL};'>99.32%</td>
            </tr>
        </tbody>
    </table>
</div>""")

    btn.click(fn=predict_both, inputs=image_input,
              outputs=[cnn_output, resnet_output, accord_output])
    image_input.change(fn=predict_both, inputs=image_input,
                       outputs=[cnn_output, resnet_output, accord_output])

if __name__ == "__main__":
    demo.launch(share=False, css=css)