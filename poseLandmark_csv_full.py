import os
import cv2
import mediapipe as mp
import glob
import pandas as pd
import argparse
import numpy as np
import math
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")
ap.add_argument("-o", "--save", type=str, required=True,
                help="path to save csv files, eg: dir/")
ap.add_argument("-t", "--test_size", type=float, default=0.2,
                help="proportion of test set (default: 0.2)")

args = vars(ap.parse_args())

path_data_dir = args["dataset"]
path_to_save = args["save"]
test_size = args["test_size"]


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Crear nombres de columnas
col_names = []
for i in range(33):  # 33 landmarks
    name = mp_pose.PoseLandmark(i).name
    col_names.extend([f'{name}_X', f'{name}_Y', f'{name}_Z', f'{name}_V'])

full_lm_list = []
target_list = []
image_paths = []  

class_list = sorted(os.listdir(path_data_dir))

for class_name in class_list:
    path_to_class = os.path.join(path_data_dir, class_name)
    img_list = glob.glob(path_to_class + '/*.jpg') + \
               glob.glob(path_to_class + '/*.jpeg') + \
               glob.glob(path_to_class + '/*.png')
    img_list = sorted(img_list)

    for img in img_list:
        image = cv2.imread(img)
        if image is None:
            print(f'[ERROR] Error in reading {img} -- Skipping.....')
            continue
        
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)
        
        if result.pose_landmarks:
           
            lm_list = []
            for landmarks in result.pose_landmarks.landmark:
                lm_list.append(landmarks)
                
            center_x = (lm_list[24].x + lm_list[23].x) * 0.5
            center_y = (lm_list[24].y + lm_list[23].y) * 0.5
            
            shoulders_x = (lm_list[12].x + lm_list[11].x) * 0.5
            shoulders_y = (lm_list[12].y + lm_list[11].y) * 0.5
            
            max_distance = 0
            for lm in lm_list:
                distance = math.sqrt((lm.x - center_x)**2 + (lm.y - center_y)**2)
                if distance > max_distance:
                    max_distance = distance
                    
            torso_size = math.sqrt((shoulders_x - center_x)**2 + (shoulders_y - center_y)**2)
            max_distance = max(torso_size * 2.5, max_distance)
            
            pre_lm = list(np.array([[(landmark.x-center_x)/max_distance, 
                                    (landmark.y-center_y)/max_distance,
                                    landmark.z/max_distance, 
                                    landmark.visibility] for landmark in lm_list]).flatten())
            
            full_lm_list.append(pre_lm)
            target_list.append(class_name)
            image_paths.append(img)
            
            print(f'{os.path.split(img)[1]} Landmarks added Successfully')
    print(f'[INFO] {class_name} Successfully Completed')

data = pd.DataFrame(full_lm_list, columns=col_names)
data['Pose_Class'] = target_list
data['Image_Path'] = image_paths


train_data, test_data = train_test_split(data, test_size=test_size, 
                                        stratify=data['Pose_Class'],
                                        random_state=42)

train_path = os.path.join(path_to_save, 'train.csv')
test_path = os.path.join(path_to_save, 'test.csv')

train_data.to_csv(train_path, index=False)
test_data.to_csv(test_path, index=False)

print(f'[INFO] Training set saved to: {train_path}')
print(f'[INFO] Testing set saved to: {test_path}')
print(f'[INFO] Training samples: {len(train_data)}')
print(f'[INFO] Testing samples: {len(test_data)}')

# Mostrar distribución de clases
print("\nClass distribution:")
print("\nTraining set:")
print(train_data['Pose_Class'].value_counts())
print("\nTesting set:")
print(test_data['Pose_Class'].value_counts())