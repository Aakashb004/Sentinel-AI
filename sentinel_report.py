import tensorflow as tf
import numpy as np
import pandas as pd
import os
import re

# Suppress TensorFlow logs for a professional terminal output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def privacy_guard(text):
    """
    Module 7: Privacy Guardrail.
    Finds and redacts sensitive patterns to ensure the report is safe for sharing.
    """
    # Redacts IP-like patterns (e.g., 192.168.1.1)
    return re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '[REDACTED_IP]', str(text))

def get_rag_remediation():
    """
    Module 5: Vulnerability RAG.
    Simulates fetching a specific remediation from the Knowledge Base.
    """
    try:
        kb = pd.read_csv('vulnerability_db.csv')
        # For this prototype, we fetch a random remediation from the DB to show RAG capability
        # In a full system, this would use vector similarity search.
        entry = kb.sample(n=1).iloc[0]
        return entry['description'], entry['remediation']
    except:
        return "Pattern matches known Exploit signatures.", "Apply standard firewall block and isolate host."

def analyze_log(log_index):
    # 1. Load Model and Data
    try:
        model = tf.keras.models.load_model('sentinel_analyst.h5')
        df = pd.read_csv('cleaned_sentinel_data.csv')
    except Exception as e:
        print(f"❌ Error: Ensure 'sentinel_analyst.h5' and 'cleaned_sentinel_data.csv' exist. {e}")
        return

    # 2. Extract and Copy Row (Avoids Read-Only Errors)
    features_row = df.drop('class', axis=1).iloc[log_index].values.copy()
    actual_label = df.iloc[log_index]['class']
    
    # 3. Module 2: Image Transformation
    # Pre-process features (handle strings and normalize)
    for i in range(len(features_row)):
        if isinstance(features_row[i], str):
            features_row[i] = 0.0
            
    features_row = features_row.astype(float)
    denom = (features_row.max() - features_row.min() + 1e-5)
    norm = (features_row - features_row.min()) / denom
    
    # Shape into 8x8 image for CNN input
    img = np.zeros(64)
    img[:len(norm)] = norm
    img = img.reshape(1, 8, 8, 1)

    # 4. AI Inference
    prediction = model.predict(img, verbose=0)
    is_normal = prediction[0][0] > 0.5
    result = "NORMAL" if is_normal else "ANOMALY"
    confidence = prediction[0][0] if is_normal else 1 - prediction[0][0]

    # 5. Final Reporting with Guardrails
    header = f"SENTINEL ANALYST REPORT - LOG ID: {log_index}"
    
    print("\n" + "="*50)
    print(f"   {privacy_guard(header)}")
    print("="*50)
    print(f"VERDICT:    {result}")
    print(f"CONFIDENCE: {confidence*100:.2f}%")
    print(f"TRUTH:      {actual_label.upper()}")
    print("-" * 50)
    
    if result == "ANOMALY":
        desc, fix = get_rag_remediation()
        print(f"🚨 THREAT ANALYSIS (Module 5 RAG):")
        print(f"Insight:     {desc}")
        print(f"Remediation: {fix}")
    else:
        print("✅ SYSTEM STATUS: SECURE")
        print("Observation: No significant deviations from baseline.")
    
    print("="*50)

if __name__ == "__main__":
    print("Sentinel AI: Autonomous Cybersecurity Analyst Online.")
    # Analyzing specific logs to verify performance
    analyze_log(0)
    analyze_log(10)