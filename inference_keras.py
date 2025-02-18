import os
import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to saved .keras model, eg: dir/model.keras")
ap.add_argument("-c", "--conf", type=float, required=True,
                help="min prediction conf to detect pose class (0<conf<1)")
ap.add_argument("-i", "--source", type=str, required=True,
                help="path to sample image")
ap.add_argument("--save", action='store_true',
                help="Save video")

args = vars(ap.parse_args())
source = args["source"]
path_saved_model = args["model"]
threshold = args["conf"]
save = args['save']

##############
torso_size_multiplier = 2.5
n_landmarks = 33
n_dimensions = 3
landmark_names = [
    'nose',
    'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear',
    'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_pinky_1', 'right_pinky_1',
    'left_index_1', 'right_index_1',
    'left_thumb_2', 'right_thumb_2',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index',
]
class_names = [
    'Chair', 'Cobra', 'Dog',
    'Tree', 'Warrior'
]
##############

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

col_names = []
for i in range(n_landmarks):
    name = mp_pose.PoseLandmark(i).name
    name_x = name + '_X'
    name_y = name + '_Y'
    name_z = name + '_Z'
    name_v = name + '_V'
    col_names.append(name_x)
    col_names.append(name_y)
    col_names.append(name_z)
    col_names.append(name_v)

# Asegurarse que el path termine en .keras
if not path_saved_model.endswith('.keras'):
    path_saved_model = os.path.splitext(path_saved_model)[0] + '.keras'

# Cargar el modelo usando tf.keras
model = tf.keras.models.load_model(path_saved_model, compile=True)

# Check if source is image/video/webcam
if source.isnumeric():
    cap = cv2.VideoCapture(int(source))
else:
    cap = cv2.VideoCapture(source)

# Get video properties if source is video
if not source.isnumeric() and not os.path.isfile(source):
    print('[ERROR] Source not found !!')
    exit(1)

if save:
    if os.path.isfile(source):
        if source.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            if not os.path.exists('ImageOutput'):
                os.makedirs('ImageOutput')
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

    # Convert the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image and detect the pose
    results = pose.process(image)

    # Convert the image color back so it can be displayed
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS)

        row = []
        for i in range(n_landmarks):
            row.append(results.pose_landmarks.landmark[i].x)
            row.append(results.pose_landmarks.landmark[i].y)
            row.append(results.pose_landmarks.landmark[i].z)
            row.append(results.pose_landmarks.landmark[i].visibility)

        X = pd.DataFrame([row], columns=col_names)
        prediction = model.predict(X)

        predicted_class = class_names[prediction.argmax()]
        prediction_conf = prediction.max()

        if prediction_conf > threshold:
            cv2.putText(image, predicted_class,
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Pose Classification', image)

    if save:
        if os.path.isfile(source):
            if source.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                name = source.split('/')[-1]
                cv2.imwrite(f'ImageOutput/{name}', image)
                break
            else:
                result.write(image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
if save:
    if not source.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        if not os.path.isfile(source):
            result.release()
cv2.destroyAllWindows()