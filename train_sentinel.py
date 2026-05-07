import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# 1. Load the cleaned data
df = pd.read_csv('cleaned_sentinel_data.csv')

# 2. Preprocess: Convert categorical text to numbers
for col in df.select_dtypes(include=['object']).columns:
    if col != 'class': # Keep the label separate for now
        df[col] = df[col].astype('category').cat.codes

# 3. Map the labels: 'normal' -> 1, 'anomaly' -> 0
df['class'] = df['class'].map({'normal': 1, 'anomaly': 0})

# 4. Prepare Features (X) and Labels (y)
X = df.drop('class', axis=1).values
y = df['class'].values

# Normalize features to 0-1 range
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-5)

# 5. Reshape into 8x8 Grayscale Images
X_images = np.zeros((len(X), 8, 8, 1)) 
for i in range(len(X)):
    padded = np.zeros(64)
    padded[:len(X[i])] = X[i]
    X_images[i] = padded.reshape(8, 8, 1)

# 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_images, y, test_size=0.2, random_state=42)

# 7. Build the Sentinel CNN Architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (2, 2), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid') # 1 for Normal, 0 for Anomaly
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 8. Train the Brain
print("\n--- Sentinel AI: Beginning CNN Training ---")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 9. Save the trained analyst
model.save('sentinel_analyst.h5')
print("\n✅ Success! Trained model saved as 'sentinel_analyst.h5'")