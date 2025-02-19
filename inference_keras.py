import os
import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
import argparse
import joblib  # Para cargar el scaler

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

if not path_saved_model.endswith('.keras'):
    path_saved_model = os.path.splitext(path_saved_model)[0] + '.keras'

model = tf.keras.models.load_model(path_saved_model, compile=True)

scaler_path = os.path.join(os.path.dirname(path_saved_model), 'scaler.pkl')
if not os.path.exists(scaler_path):
    print(f"[ERROR] No se encontró el scaler en: {scaler_path}")
    exit(1)
scaler = joblib.load(scaler_path)


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

n_landmarks = 33
col_names = []
for i in range(n_landmarks):
    name = mp_pose.PoseLandmark(i).name
    col_names.extend([f"{name}_X", f"{name}_Y", f"{name}_Z", f"{name}_V"])

# Definir nombres de clases
class_names = [
    'Fall forward', 'Fall sitting', 'Walk', 'Sit down', 'Kneel',
    'Fall backwards', 'Fall right', 'Pick up object', 'Fall left'
]

# Cargar la fuente de video/imágenes
if source.isnumeric():
    cap = cv2.VideoCapture(int(source))
else:
    cap = cv2.VideoCapture(source)

# Validar si la fuente es correcta
if not cap.isOpened():
    print('[ERROR] Source not found or cannot be opened!')
    exit(1)

# Configurar salida de video si se va a guardar
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

    # Convertir la imagen de BGR a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar la imagen y detectar la pose
    results = pose.process(image_rgb)

    # Convertir la imagen de vuelta a BGR para visualizar
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Dibujar los landmarks de la pose si hay detección
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS)

        # Extraer características de la pose detectada
        row = []
        for i in range(n_landmarks):
            landmark = results.pose_landmarks.landmark[i]
            row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

        # Convertir a DataFrame con los nombres correctos
        X = pd.DataFrame([row], columns=col_names)

        # **Aplicar la normalización con el scaler del entrenamiento**
        X_scaled = scaler.transform(X)

        # Hacer la predicción
        prediction = model.predict(X_scaled)

        # Obtener la clase con mayor probabilidad
        predicted_class = class_names[np.argmax(prediction)]
        prediction_conf = np.max(prediction)

        # Mostrar la clase si la confianza supera el umbral
        if prediction_conf > threshold:
            cv2.putText(image, f"{predicted_class} ({prediction_conf:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar imagen con clasificación
    cv2.imshow('Pose Classification', image)

    # Guardar la imagen/video procesado si se activó la opción
    if save:
        if os.path.isfile(source) and source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            name = os.path.basename(source)
            cv2.imwrite(f'ImageOutput/{name}', image)
            break  # Solo guardar una vez para imágenes
        else:
            result.write(image)

    # Salir si se presiona 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
if save and not source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
    result.release()
cv2.destroyAllWindows()
