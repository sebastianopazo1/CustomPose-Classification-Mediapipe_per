import os
import shutil
from sklearn.model_selection import train_test_split
import argparse

def split_dataset(input_dir, output_dir, test_size=0.2, random_state=42):
    
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    class_list = sorted(os.listdir(input_dir))

    for class_name in class_list:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        class_path = os.path.join(input_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

        train_images, test_images = train_test_split(
            images, 
            test_size=test_size, 
            random_state=random_state
        )

        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_dir, class_name, img)
            shutil.copy2(src, dst)

        for img in test_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(test_dir, class_name, img)
            shutil.copy2(src, dst)

        print(f'Clase {class_name}: {len(train_images)} imágenes de entrenamiento, {len(test_images)} imágenes de prueba')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="Directorio del dataset de entrada")
    ap.add_argument("-o", "--output", required=True,
                    help="Directorio donde se guardara el dataset dividido")
    ap.add_argument("-t", "--test_size", type=float, default=0.2,
                    help="Proporcion del conjunto de prueba")
    args = vars(ap.parse_args())

    split_dataset(args["input"], args["output"], args["test_size"])