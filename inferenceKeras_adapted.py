import os
import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import argparse
import joblib

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="Path to saved .keras model, e.g., dir/model.keras")
ap.add_argument("-c", "--conf", type=float, required=True,
                help="Min prediction confidence to detect pose class (0<conf<1)")
ap.add_argument("-i", "--source", type=str, required=True,
                help="Path to sample image or video")
ap.add_argument("--save", action='store_true',
                help="Save processed video or image")

args = vars(ap.parse_args())
source = args["source"]
path_saved_model = args["model"]
threshold = args["conf"]
save = args['save']

# Asegurar extensión .keras
if not path_saved_model.endswith('.keras'):
    path_saved_model = os.path.splitext(path_saved_model)[0] + '.keras'

# Cargar el modelo
print("[INFO] Loading model...")
model = tf.keras.models.load_model(path_saved_model, compile=True)

# Cargar el RobustScaler
scaler_path = os.path.join(os.path.dirname(path_saved_model), 'scaler.pkl')
if not os.path.exists(scaler_path):
    print(f"[ERROR] No se encontró el scaler en: {scaler_path}")
    exit(1)
scaler = joblib.load(scaler_path)

# Configuración de MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


n_landmarks = 33
col_names = []
for i in range(n_landmarks):
    name = mp_pose.PoseLandmark(i).name
    col_names.extend([f"{name}_X", f"{name}_Y", f"{name}_Z", f"{name}_V"])


class_names = [
    'Fall forward', 'Fall sitting', 'Walk', 'Sit down', 'Kneel',
    'Fall backwards', 'Fall right', 'Pick up object', 'Fall left'
]

if source.isnumeric():
    cap = cv2.VideoCapture(int(source))
else:
    cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print('[ERROR] Source not found or cannot be opened!')
    exit(1)


if save:
    if os.path.isfile(source) and source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        os.makedirs('ImageOutput', exist_ok=True)
    else:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        result = cv2.VideoWriter('output.avi',
                               cv2.VideoWriter_fourcc(*'MJPG'),
                               10, size)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Procesar imagen
    results = pose.process(image_rgb)
    
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # Dibujar landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS)

        row = []
        for i in range(n_landmarks):
            landmark = results.pose_landmarks.landmark[i]
            row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

        X = pd.DataFrame([row], columns=col_names)

        X_scaled = scaler.transform(X)


        prediction = model.predict(X_scaled, verbose=0)
        predicted_class = class_names[np.argmax(prediction)]
        prediction_conf = np.max(prediction)


        if prediction_conf > threshold:
            text = f"{predicted_class} ({prediction_conf:.2f})"
            cv2.putText(image, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar imagen
    cv2.imshow('Pose Classification', image)

    if save:
        if os.path.isfile(source) and source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            name = os.path.basename(source)
            cv2.imwrite(f'ImageOutput/{name}', image)
            break
        else:
            result.write(image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
if save and not source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
    result.release()
cv2.destroyAllWindows()