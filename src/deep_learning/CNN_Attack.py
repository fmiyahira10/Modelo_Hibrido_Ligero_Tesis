import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# ===================================================================
# 1. CONFIGURACIÓN DE RUTAS RELATIVAS Y CARGA DE MATRICES
# ===================================================================
BASE_DIR = r'C:\Users\i21327\Desktop\Tesis\Modelo_Hibrido_Ligero_Tesis'

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
        filepath=os.path.join(BASE_DIR, 'best_cnn_attack_model.keras'),
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