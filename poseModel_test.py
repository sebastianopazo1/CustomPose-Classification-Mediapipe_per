import keras
import pandas as pd
from keras import layers, Sequential
import argparse
from matplotlib import pyplot as plt
import os
import numpy as np
import time
import joblib  # Para guardar el scaler
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--train", type=str, required=True,
                help="Path to training CSV data")
ap.add_argument("-t", "--test", type=str, required=True,
                help="Path to test CSV data")
ap.add_argument("-o", "--save", type=str, required=True,
                help="Path to save .keras model, e.g., dir/model.keras")

args = vars(ap.parse_args())
train_csv = args["train"]
test_csv = args["test"]
path_to_save = args["save"]

# Ensure file extension is .keras
if not path_to_save.endswith('.keras'):
    path_to_save = os.path.join(path_to_save, 'model.keras')

# Ensure directory exists
os.makedirs(os.path.dirname(path_to_save), exist_ok=True)

# Load training data
print('[INFO] Loading training data...')
train_df = pd.read_csv(train_csv)
class_list = sorted(train_df['Pose_Class'].unique())
class_number = len(class_list)

# Prepare training data
x_train = train_df.copy()
y_train = x_train.pop('Pose_Class')
if 'Image_Path' in x_train.columns:
    x_train.pop('Image_Path')  # Remove Image_Path column if it exists

# Encode class labels
y_train_encoded, class_mapping = y_train.factorize()
x_train = x_train.astype('float64')
y_train = keras.utils.to_categorical(y_train_encoded)

# Load and prepare test data
print('[INFO] Loading test data...')
test_df = pd.read_csv(test_csv)
x_test = test_df.copy()
y_test = x_test.pop('Pose_Class')
if 'Image_Path' in x_test.columns:
    x_test.pop('Image_Path')  # Remove Image_Path column if it exists

# Ensure y_test uses the same encoding as y_train
y_test_encoded = pd.Categorical(y_test, categories=class_mapping).codes
y_test_encoded = np.where(y_test_encoded == -1, 0, y_test_encoded)  # Handle missing classes
x_test = x_test.astype('float64')
y_test = keras.utils.to_categorical(y_test_encoded, num_classes=class_number)

print('[INFO] Loaded Training and Test Datasets')
print(f'Training samples: {len(x_train)}')
print(f'Test samples: {len(x_test)}')

# Normalize the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# **Guardar el scaler en la misma carpeta del modelo**
scaler_path = os.path.join(os.path.dirname(path_to_save), 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"[INFO] Scaler guardado en: {scaler_path}")

# Create model
model = Sequential([
    layers.Dense(512, activation='relu', input_shape=[x_train.shape[1]]),
    layers.Dense(256, activation='relu'),
    layers.Dense(class_number, activation="softmax")
])

print('Model Summary:')
model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint = keras.callbacks.ModelCheckpoint(path_to_save,
                                             monitor='val_accuracy',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='max')

earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=20,
                                              restore_best_weights=True)

print('[INFO] Model Training Started ...')

# Training
history = model.fit(x_train, y_train,
                    epochs=40,
                    batch_size=32,  # Increased batch size
                    validation_data=(x_test, y_test),
                    callbacks=[checkpoint, earlystopping])

print('[INFO] Model Training Completed')
print(f'[INFO] Model Successfully Saved in {path_to_save}')

# Evaluate on test set
print("\n[INFO] Evaluating on Test Set...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Detailed classification report
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes, 
                            target_names=class_mapping))

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()

# Save plot with a unique name
timestamp = int(time.time())
plt.savefig(f'metrics_{timestamp}.png', bbox_inches='tight')
print(f'[INFO] Successfully Saved metrics_{timestamp}.png')
