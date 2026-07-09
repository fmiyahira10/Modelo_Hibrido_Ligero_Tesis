import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from pathlib import Path

# ===================================================================
# 1. CONFIGURACIÓN DE RUTAS RELATIVAS Y CARGA DE MATRICES
# ===================================================================
# Raíz del proyecto: sube 2 niveles desde src/deep_learning/
BASE_DIR = Path(__file__).resolve().parents[2]

print("Cargando matrices NumPy de la Fase 3...")
X_train = np.load(os.path.join(BASE_DIR,'data' ,'final', 'X_train_attack.npy'))
X_val = np.load(os.path.join(BASE_DIR,'data' ,'final', 'X_val_attack.npy'))
y_train = np.load(os.path.join(BASE_DIR,'data' ,'final', 'y_train_attack.npy'))
y_val = np.load(os.path.join(BASE_DIR, 'data' ,'final', 'y_val_attack.npy'))

num_clases = len(np.unique(y_train))

# --- REFORMATEO TRIDIMENSIONAL EXIGIDO POR CONV1D ---
# Pasamos de (muestras, 13) a (muestras, 13, 1)
X_train_reshaped = np.expand_dims(X_train, axis=-1)
X_val_reshaped = np.expand_dims(X_val, axis=-1)

print(f"Estructura final tensores Entrenamiento : {X_train_reshaped.shape}")
print(f"Estructura final tensores Validación    : {X_val_reshaped.shape}")

# ===================================================================
# 2. CONSTRUCCIÓN DE LA ARQUITECTURA CONVOLUCIONAL LIGERA
# ===================================================================
print("\nDiseñando la arquitectura del Submodelo A (CNN_Attack)...")

input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2]) # (13, 1)
inputs = Input(shape=input_shape, name='Input_Tráfico')

# Bloque Convolucional 1
x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', name='Conv1D_1')(inputs)
x = BatchNormalization(name='BatchNorm_1')(x)
x = MaxPooling1D(pool_size=2, name='MaxPool_1')(x)
x = Dropout(0.2, seed=42, name='Dropout_1')(x)

# Bloque Convolucional 2
x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name='Conv1D_2')(x)
x = BatchNormalization(name='BatchNorm_2')(x)
x = Flatten(name='Flatten_Vectores')(x)

# Capa Intermedia de Representación Latente (BOTTLENECK / EMBEDDING)
# Reducimos las dimensiones a un vector compacto de 16 rasgos abstractos potenciados
embedding_layer = Dense(16, activation='relu', name='Embedding_Attack')(x)

# Capa de Clasificación para el entrenamiento supervisado de la CNN
outputs = Dense(num_clases, activation='softmax', name='Salida_Clasificador_CNN')(embedding_layer)

# Instanciación formal del Modelo Completo de entrenamiento
model_attack = Model(inputs=inputs, outputs=outputs, name='CNN_Attack_Full')

model_attack.summary()

# ===================================================================
# 3. COMPILACIÓN Y CONFIGURACIÓN DE OPTIMIZADORES
# ===================================================================
model_attack.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy', # Usamos sparse porque y_train contiene códigos enteros (0 a 5)
    metrics=['accuracy']
)

# Callbacks estratégicos para evitar Overfitting y salvaguardar el mejor estado
callbacks_list = [
    EarlyStopping(
        monitor='val_loss', 
        patience=5, # Si la pérdida de validación no mejora en 5 épocas, detiene el entrenamiento
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(BASE_DIR, 'src', 'Embedding', 'best_cnn_attack_model.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# ===================================================================
# 4. ENTRENAMIENTO SUPERVISADO DEL SUBMODELO
# ===================================================================
print("\nIniciando entrenamiento del submodelo de vectores de ataque...")
history = model_attack.fit(
    X_train_reshaped, y_train,
    validation_data=(X_val_reshaped, y_val),
    epochs=20,
    batch_size=256, # Batch size balanceado para entrenamiento ágil en CPU/GPU
    callbacks=callbacks_list,
    verbose=1
)

print("\n¡Entrenamiento de la CNN terminado y mejor modelo guardado en disco duro!")


# ===================================================================
# 5. GENERACIÓN Y ALMACENAMIENTO DE CURVAS DE APRENDIZAJE (LOSS & ACC)
# ===================================================================
import matplotlib.pyplot as plt

print("\nGenerando gráficos de rendimiento del submodelo...")

# Definir la ruta de salida para la documentación de la tesis
output_graph_path = os.path.join(BASE_DIR, 'src', 'Embedding', 'curvas_rendimiento_attack.png')

# Configurar el lienzo de visualización con dos subgráficas (Loss y Accuracy)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- Gráfica 1: Evolución de la Función de Pérdida (Loss) ---
ax1.plot(history.history['loss'], label='Pérdida de Entrenamiento (Train Loss)', color='#1f77b4', linewidth=2.5)
ax1.plot(history.history['val_loss'], label='Pérdida de Validación (Val Loss)', color='#ff7f0e', linewidth=2.5)
ax1.set_title('Evolución del Loss (Crossentropy)', fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel('Épocas de Entrenamiento', fontsize=11)
ax1.set_ylabel('Valor de Pérdida', fontsize=11)
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.5)

# --- Gráfica 2: Evolución de la Exactitud (Accuracy) ---
ax2.plot(history.history['accuracy'], label='Exactitud de Entrenamiento (Train Acc)', color='#2ca02c', linewidth=2.5)
ax2.plot(history.history['val_accuracy'], label='Exactitud de Validación (Val Acc)', color='#d62728', linewidth=2.5)
ax2.set_title('Evolución de la Exactitud (Accuracy)', fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('Épocas de Entrenamiento', fontsize=11)
ax2.set_ylabel('Tasa de Acierto (0.0 - 1.0)', fontsize=11)
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, linestyle='--', alpha=0.5)

# Ajustar diseño general para evitar solapamiento de textos
plt.suptitle('Métricas del Pipeline de Entrenamiento - Submodelo A (CNN_Attack)', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()

# Guardar la gráfica en disco en alta resolución para tu informe de tesis
plt.savefig(output_graph_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"¡Gráfica científica guardada exitosamente en: {output_graph_path}!")