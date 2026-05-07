import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the cleaned data
df = pd.read_csv('cleaned_sentinel_data.csv')

# 2. Convert categories to numbers (Preprocessing)
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category').cat.codes

def get_image(index):
    # Take features, normalize them to 0-255 (pixel range)
    features = df.iloc[index, :-1].values
    norm = ((features - features.min()) / (features.max() - features.min() + 1e-5) * 255).astype(np.uint8)
    
    # Pad to 64 pixels and reshape to 8x8
    img_data = np.zeros(64)
    img_data[:len(norm)] = norm
    return img_data.reshape(8, 8)

# 3. Compare Normal vs Anomaly
normal_idx = df[df['class'] == 1].index[0] # First 'normal' row
attack_idx = df[df['class'] == 0].index[0] # First 'anomaly' row

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(get_image(normal_idx), cmap='gray')
plt.title("Sentinel View: NORMAL")

plt.subplot(1, 2, 2)
plt.imshow(get_image(attack_idx), cmap='hot') # Using 'hot' to highlight the attack
plt.title("Sentinel View: ANOMALY")

plt.show()