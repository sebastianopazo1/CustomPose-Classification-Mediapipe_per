import keras
import pandas as pd
from keras import layers, Sequential
import argparse
from matplotlib import pyplot as plt
import os
import numpy as np
import time
import joblib
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau

def create_model(input_shape, num_classes, dropout_rate=0.3):
    """
    Función para crear un modelo más robusto con arquitectura mejorada
    """
    model = Sequential([
        # Capa de entrada con normalización
        layers.BatchNormalization(input_shape=input_shape),
        
        # Primera capa densa con más unidades y regularización
        layers.Dense(1024, kernel_regularizer=l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),
        
        # Segunda capa con skip connection
        layers.Dense(512, kernel_regularizer=l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),
        
        # Tercera capa con atención a las características espaciales
        layers.Dense(256, kernel_regularizer=l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),
        
        # Capa de salida
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

def main():
    # Argumentos de línea de comandos (mantener los existentes)
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--train", type=str, required=True,
                    help="Path to training CSV data")
    ap.add_argument("-t", "--test", type=str, required=True,
                    help="Path to test CSV data")
    ap.add_argument("-o", "--save", type=str, required=True,
                    help="Path to save .keras model")
    
    args = vars(ap.parse_args())
    
    # Cargar y preparar datos
    print('[INFO] Loading training data...')
    train_df = pd.read_csv(args["train"])
    test_df = pd.read_csv(args["test"])
    
    # Preparación de datos mejorada
    x_train = train_df.copy()
    y_train = x_train.pop('Pose_Class')
    x_test = test_df.copy()
    y_test = x_test.pop('Pose_Class')
    
    # Eliminar columnas no necesarias
    for df in [x_train, x_test]:
        if 'Image_Path' in df.columns:
            df.pop('Image_Path')
    
    # Usar RobustScaler en lugar de StandardScaler para mejor manejo de outliers
    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # Codificación de etiquetas
    y_train_encoded, class_mapping = y_train.factorize()
    y_test_encoded = pd.Categorical(y_test, categories=class_mapping).codes
    
    num_classes = len(class_mapping)
    y_train = keras.utils.to_categorical(y_train_encoded)
    y_test = keras.utils.to_categorical(y_test_encoded, num_classes=num_classes)
    
    # Crear y compilar modelo
    model = create_model(input_shape=[x_train.shape[1]], num_classes=num_classes)
    
    # Compilación con optimizador mejorado
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks mejorados
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            args["save"],
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001
        )
    ]
    
    # Entrenamiento con batch size adaptativo
    batch_size = min(32, len(x_train) // 10)
    history = model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluación y métricas
    print("\n[INFO] Evaluating on Test Set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Reporte de clasificación detallado
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes, 
                              target_names=class_mapping))
    
    # Guardar gráficas de métricas
    plot_metrics(history)

def plot_metrics(history):
    timestamp = int(time.time())
    plt.figure(figsize=(15, 5))
    
    # Plot de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot de precisión
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'metrics_{timestamp}.png')
    print(f'[INFO] Metrics plot saved as metrics_{timestamp}.png')

if __name__ == "__main__":
    main()