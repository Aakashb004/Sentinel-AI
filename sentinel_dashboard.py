import gradio as gr
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# Load assets
model = tf.keras.models.load_model('sentinel_analyst.h5')
df = pd.read_csv('cleaned_sentinel_data.csv')
kb = pd.read_csv('vulnerability_db.csv')

def privacy_guard(text):
    return re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '[REDACTED_IP]', str(text))

def process_and_predict(log_id):
    # 1. Image Generation
    features = df.drop('class', axis=1).iloc[int(log_id)].values.copy()
    for i in range(len(features)):
        if isinstance(features[i], str): features[i] = 0.0
    
    features = features.astype(float)
    norm = (features - features.min()) / (features.max() - features.min() + 1e-5)
    img_8x8 = np.zeros(64)
    img_8x8[:len(norm)] = norm
    img_8x8 = img_8x8.reshape(8, 8)
    
    # 2. Prediction
    prediction = model.predict(img_8x8.reshape(1, 8, 8, 1), verbose=0)[0][0]
    is_normal = prediction > 0.5
    status = "✅ NORMAL" if is_normal else "🚨 ANOMALY"
    conf = prediction if is_normal else 1 - prediction
    
    # 3. RAG Insight
    if not is_normal:
        entry = kb.sample(n=1).iloc[0]
        report = f"TRUTH: {df.iloc[int(log_id)]['class'].upper()}\n\nINSIGHT: {entry['description']}\n\nFIX: {entry['remediation']}"
    else:
        report = f"TRUTH: {df.iloc[int(log_id)]['class'].upper()}\n\nNo threats detected. System operating within normal parameters."

    # 4. Create Heatmap Plot
    plt.figure(figsize=(4,4))
    plt.imshow(img_8x8, cmap='hot' if not is_normal else 'viridis')
    plt.axis('off')
    plt.savefig('temp_plot.png', bbox_inches='tight')
    plt.close()
    
    return 'temp_plot.png', f"STATUS: {status} ({conf*100:.2f}%)\n\n{privacy_guard(report)}"

# Build UI
with gr.Blocks(title="Sentinel AI Dashboard") as demo:
    gr.Markdown("# 🛡️ Sentinel AI: Autonomous Analyst")
    gr.Markdown("Module 2 (CNN) + Module 5 (RAG) + Module 7 (Guardrails)")
    
    with gr.Row():
        log_input = gr.Number(label="Enter Log ID (0-20000)", value=0)
        btn = gr.Button("Analyze Traffic", variant="primary")
        
    with gr.Row():
        img_output = gr.Image(label="CNN 8x8 Visual Input")
        text_output = gr.Textbox(label="Security Analyst Report", lines=10)
        
    btn.click(process_and_predict, inputs=log_input, outputs=[img_output, text_output])

demo.launch()