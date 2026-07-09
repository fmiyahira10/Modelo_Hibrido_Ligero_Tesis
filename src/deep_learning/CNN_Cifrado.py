import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from pathlib import Path

# Configuración de estética académica para publicaciones (IEEE/Elsevier)
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
})

# ===================================================================
# 1. CONFIGURACIÓN DE RUTAS RELATIVAS Y CARGA DE MATRICES (DARKNET)
# ===================================================================
# Raíz del proyecto: sube 2 niveles desde src/deep_learning/
BASE_DIR = Path(__file__).resolve().parents[2]

print("Cargando matrices matriciales NumPy del carril de cifrado...")
X_train = np.load(os.path.join(BASE_DIR,'data','final', 'X_train_encryption.npy'))
X_val = np.load(os.path.join(BASE_DIR,'data','final', 'X_val_encryption.npy'))
X_test = np.load(os.path.join(BASE_DIR,'data','final', 'X_test_encryption.npy'))

y_train = np.load(os.path.join(BASE_DIR,'data','final', 'y_train_encryption.npy'))
y_val = np.load(os.path.join(BASE_DIR,'data','final', 'y_val_encryption.npy'))
y_test = np.load(os.path.join(BASE_DIR,'data','final', 'y_test_encryption.npy'))

le_encryption = joblib.load(os.path.join(BASE_DIR,'models', 'label_encoder_encryption.pkl'))

# --- REFORMA TRIDIMENSIONAL PARA PROCESAMIENTO CONVOLUCIONAL ---
X_train_reshaped = np.expand_dims(X_train, axis=-1)
X_val_reshaped = np.expand_dims(X_val, axis=-1)
X_test_reshaped = np.expand_dims(X_test, axis=-1)

print(f"Formato tensor de Entrenamiento : {X_train_reshaped.shape}")
print(f"Formato tensor de Validación    : {X_val_reshaped.shape}")
print(f"Formato tensor de Prueba        : {X_test_reshaped.shape}")

# ===================================================================
# 2. ARQUITECTURA DE LA RED NEURONAL BLINDADA CONTRA OVERFITTING
# ===================================================================
print("\nDiseñando arquitectura optimizada para Submodelo B (CNN_Encryption)...")

input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2]) # (62, 1)
inputs = Input(shape=input_shape, name='Input_Descriptores_Crudos')

# Bloque Convolucional Ligero 1: Aumentamos Dropout para mitigar el ruido inicial de las 62 variables
x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', name='Conv1D_Cifrado_1')(inputs)
x = BatchNormalization(name='BatchNorm_Cifrado_1')(x)
x = MaxPooling1D(pool_size=2, name='MaxPool_Cifrado_1')(x)
x = Dropout(0.3, seed=42, name='Dropout_Cifrado_1')(x)

# Bloque Convolucional Ligero 2
x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name='Conv1D_Cifrado_2')(x)
x = BatchNormalization(name='BatchNorm_Cifrado_2')(x)
x = MaxPooling1D(pool_size=2, name='MaxPool_Cifrado_2')(x) # Agregamos un segundo pooling para reducir dimensiones
x = Flatten(name='Flatten_Cifrado')(x)

# --- ESCUDO ANTISOBREAJUSTE CRÍTICO ---
# Apagamos aleatoriamente el 40% de las conexiones aplanadas para forzar la generalización
x = Dropout(0.4, seed=42, name='Dropout_Pre_Bottleneck')(x)

# CAPA BOTTLENECK: Espacio latente comprimido con regularización L2
x = Dense(
    16, 
    activation='relu', 
    kernel_regularizer=tf.keras.regularizers.l2(0.001), # Penaliza pesos gigantes estructurales
    name='Embedding_Encryption'
)(x)

# Dropout final de protección antes de la salida sigmoidal
x = Dropout(0.2, seed=42, name='Dropout_Post_Bottleneck')(x)

# Capa de salida binaria (0 = No Cifrado, 1 = Cifrado)
outputs = Dense(1, activation='sigmoid', name='Salida_Clasificador_Binario')(x)

model_encryption = Model(inputs=inputs, outputs=outputs, name='CNN_Encryption_Full')
model_encryption.summary()

# ===================================================================
# 3. COMPILACIÓN Y PARÁMETROS DE ENTRENAMIENTO (CORREGIDO + HISTORIAL)
# ===================================================================
model_encryption.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy', # Pérdida idónea para targets binarios (0/1)
    metrics=['accuracy']
)

callbacks_list = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        filepath=os.path.join(BASE_DIR, 'src', 'Embedding', 'best_cnn_encryption_model.keras'),
        monitor='val_loss', save_best_only=True, verbose=1
    )
]

print("\nIniciando entrenamiento del submodelo de cifrado...")
# CORRECCIÓN: Asignamos el entrenamiento a la variable 'history' para capturar las métricas
history = model_encryption.fit(
    X_train_reshaped, y_train,
    validation_data=(X_val_reshaped, y_val),
    epochs=20,
    batch_size=128, # Ajustamos el tamaño del lote por la densidad de características (62)
    callbacks=callbacks_list,
    verbose=1
)

# ===================================================================
# 3.5. GENERACIÓN Y ALMACENAMIENTO DE CURVAS DE APRENDIZAJE (CARRIL B)
# ===================================================================
print("\nGenerando gráficos de rendimiento del submodelo de cifrado...")
output_perf_path = os.path.join(BASE_DIR, 'src', 'Embedding', 'curvas_rendimiento_encryption.png')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

# --- Gráfica 1: Evolución de la Función de Pérdida Binaria (Loss) ---
ax1.plot(history.history['loss'], label='Pérdida de Entrenamiento (Train Loss)', color='#1f77b4', linewidth=2.5)
ax1.plot(history.history['val_loss'], label='Pérdida de Validación (Val Loss)', color='#ff7f0e', linewidth=2.5)
ax1.set_title('Evolución del Loss (Binary Crossentropy)', fontweight='bold', pad=10)
ax1.set_xlabel('Épocas de Entrenamiento')
ax1.set_ylabel('Valor de Pérdida')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.5)

# --- Gráfica 2: Evolución de la Exactitud Binaria (Accuracy) ---
ax2.plot(history.history['accuracy'], label='Exactitud de Entrenamiento (Train Acc)', color='#2ca02c', linewidth=2.5)
ax2.plot(history.history['val_accuracy'], label='Exactitud de Validación (Val Acc)', color='#d62728', linewidth=2.5)
ax2.set_title('Evolución de la Exactitud (Accuracy Binaria)', fontweight='bold', pad=10)
ax2.set_xlabel('Épocas de Entrenamiento')
ax2.set_ylabel('Tasa de Acierto (0.0 - 1.0)')
ax2.legend(loc='lower right')
ax2.grid(True, linestyle=':', alpha=0.5)

plt.suptitle('Métricas del Pipeline de Entrenamiento - Submodelo B (CNN_Encryption)', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

# Guardar la gráfica en alta calidad técnica para la documentación de tu tesis
plt.savefig(output_perf_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"¡Gráfica de curvas de rendimiento guardada exitosamente en: '{output_perf_path}'!")

# ===================================================================
# 4. FASE 5: EXTRACCIÓN Y VALIDACIÓN VISUAL t-SNE DEL ESPACIO LATENTE
# ===================================================================
print("\n[Fase 5] Truncando el clasificador para extraer los embeddings de prueba...")
modelo_extractor = Model(inputs=model_encryption.input, outputs=model_encryption.get_layer('Embedding_Encryption').output)
embeddings_16d = modelo_extractor.predict(X_test_reshaped)

print("Calculando reducción t-SNE para el reporte visual de la tesis...")
# Procesamos los embeddings de prueba para el gráfico
num_muestras_tsne = min(4000, len(embeddings_16d))
indices = np.random.choice(len(embeddings_16d), size=num_muestras_tsne, replace=False)

embeddings_muestra = embeddings_16d[indices]
y_muestra_codificada = y_test[indices]
labels_muestra_reales = le_encryption.inverse_transform(y_muestra_codificada)

tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_muestra)

# Generación formal del gráfico
fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
sns.scatterplot(
    x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], 
    hue=labels_muestra_reales, palette=['#1f77b4', '#ff7f0e'], # Colores contrastantes profesionales
    alpha=0.7, s=15, ax=ax
)

ax.set_title("Visualización t-SNE del Espacio Latente Binario (CNN_Encryption)", pad=15)
ax.set_xlabel("Dimensión t-SNE 1")
ax.set_ylabel("Dimensión t-SNE 2")
ax.legend(title="Estado de Encriptación", loc='best')
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
ruta_grafico = os.path.join(BASE_DIR, 'src', 'Embedding', 'espacio_latente_tsne_encryption.png')
plt.savefig(ruta_grafico, dpi=300)
plt.close()

print(f"\n[PROCESO COMPLETADO] Modelo guardado y t-SNE exportado en: '{ruta_grafico}'")