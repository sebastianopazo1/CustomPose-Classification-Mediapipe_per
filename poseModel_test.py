import keras
import pandas as pd
from keras import layers, Sequential
import argparse
from matplotlib import pyplot as plt
import os
import numpy as np
from sklearn.metrics import classification_report

# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--train", type=str, required=True,
                help="path to training csv Data")
ap.add_argument("-t", "--test", type=str, required=True,
                help="path to test csv Data")
ap.add_argument("-o", "--save", type=str, required=True,
                help="path to save .h5 model, eg: dir/model.h5")

args = vars(ap.parse_args())
train_csv = args["train"]
test_csv = args["test"]
path_to_save = args["save"]

# Add this after parsing arguments
if not args["save"].endswith('.keras'):
    path_to_save = os.path.join(args["save"], 'model.keras')
else:
    path_to_save = args["save"]

# Make sure the directory exists
os.makedirs(os.path.dirname(path_to_save), exist_ok=True)

# Then use path_to_save for the checkpoint
checkpoint_path = path_to_save
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                           monitor='val_accuracy',
                                           verbose=1,
                                           save_best_only=True,
                                           mode='max')
# Load training data
print('[INFO] Loading training data...')
train_df = pd.read_csv(train_csv)
class_list = train_df['Pose_Class'].unique()
class_list = sorted(class_list)
class_number = len(class_list)

# Prepare training data
x_train = train_df.copy()
y_train = x_train.pop('Pose_Class')
if 'Image_Path' in x_train.columns:
    x_train.pop('Image_Path')  # Remove Image_Path column if it exists
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
y_test_encoded = pd.Categorical(y_test, categories=class_mapping).codes
x_test = x_test.astype('float64')
y_test = keras.utils.to_categorical(y_test_encoded)

print('[INFO] Loaded Training and Test Datasets')
print(f'Training samples: {len(x_train)}')
print(f'Test samples: {len(x_test)}')

# Create model
model = Sequential([
    layers.Dense(512, activation='relu', input_shape=[x_train.shape[1]]),
    layers.Dense(256, activation='relu'),
    layers.Dense(class_number, activation="softmax")
])

print('Model Summary: ')
model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint_path = path_to_save
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                           monitor='val_accuracy',
                                           verbose=1,
                                           save_best_only=True,
                                           mode='max')
earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                             patience=20)

print('[INFO] Model Training Started ...')
# Training
history = model.fit(x_train, y_train,
                   epochs=200,
                   batch_size=16,
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

# Save plot
if os.path.exists('metrics.png'):
    os.remove('metrics.png')
plt.savefig('metrics.png', bbox_inches='tight')
print('[INFO] Successfully Saved metrics.png')