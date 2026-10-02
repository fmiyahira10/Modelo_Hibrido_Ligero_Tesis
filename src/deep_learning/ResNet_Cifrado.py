import os
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Add, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

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
BASE_DIR = Path(__file__).resolve().parents[2]

print("Cargando matrices NumPy del carril de cifrado...")
X_train = np.load(os.path.join(BASE_DIR, 'data', 'final', 'X_train_encryption.npy'))
X_val = np.load(os.path.join(BASE_DIR, 'data', 'final', 'X_val_encryption.npy'))
X_test = np.load(os.path.join(BASE_DIR, 'data', 'final', 'X_test_encryption.npy'))

y_train = np.load(os.path.join(BASE_DIR, 'data', 'final', 'y_train_encryption.npy'))
y_val = np.load(os.path.join(BASE_DIR, 'data', 'final', 'y_val_encryption.npy'))
y_test = np.load(os.path.join(BASE_DIR, 'data', 'final', 'y_test_encryption.npy'))

le_encryption = joblib.load(os.path.join(BASE_DIR, 'models', 'scalers_encoders', 'label_encoder_encryption.pkl'))

input_dim = X_train.shape[1] # 62 características activas
print(f"Dimensiones de entrada   : {input_dim} variables activas")

# ===================================================================
# 2. ARQUITECTURA RESNET-MLP OPTIMIZADA (PRE-ACTIVATION + PIRAMIDAL)
# ===================================================================
print("\nDiseñando arquitectura ResNet-MLP sintonizada (Carril Cifrado)...")

def preact_residual_block_mlp(x, units, dropout_rate=0.2, prefix="res_block"):
    """Bloque Residual Tabular con Pre-Activación para flujo limpio de identidad"""
    # Camino no lineal F(x)
    h = BatchNormalization(name=f"{prefix}_pre_bn1")(x)
    h = Activation('relu', name=f"{prefix}_pre_act1")(h)
    h = Dense(units, kernel_regularizer=tf.keras.regularizers.l2(5e-4), name=f"{prefix}_dense1")(h)
    
    h = BatchNormalization(name=f"{prefix}_bn2")(h)
    h = Activation('relu', name=f"{prefix}_act2")(h)
    h = Dropout(dropout_rate, seed=42, name=f"{prefix}_drop")(h)
    h = Dense(units, kernel_regularizer=tf.keras.regularizers.l2(5e-4), name=f"{prefix}_dense2")(h)
    
    # Proyección lineal en la rama shortcut solo si cambian las dimensiones
    if x.shape[-1] != units:
        shortcut = Dense(units, kernel_regularizer=tf.keras.regularizers.l2(5e-4), name=f"{prefix}_shortcut")(x)
    else:
        shortcut = x
        
    # Conexión directa pura: identidad limpia
    out = Add(name=f"{prefix}_add")([shortcut, h])
    return out

inputs = Input(shape=(input_dim,), name='Input_Descriptores_Cifrado')

# Proyección inicial acotada
x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(5e-4), name='Proyeccion_Inicial')(inputs)

# Bloque 1: Expansión y refinamiento (64 unidades)
x = preact_residual_block_mlp(x, units=64, dropout_rate=0.20, prefix="ResBlock_1")

# Bloque 2: Compresión piramidal dirigida (32 unidades)
x = preact_residual_block_mlp(x, units=32, dropout_rate=0.20, prefix="ResBlock_2")

# Normalización previa al cuello de botella
x = BatchNormalization(name='BatchNorm_Pre_Bottleneck')(x)
x = Activation('relu', name='Act_Pre_Bottleneck')(x)
x = Dropout(0.3, seed=42, name='Dropout_Pre_Bottleneck')(x)

# CAPA BOTTLENECK: Representación latente compacta de 16 dimensiones
latent_bottleneck = Dense(
    16, 
    activation='relu', 
    kernel_regularizer=tf.keras.regularizers.l2(5e-4), 
    name='Embedding_Encryption_ResNet_MLP'
)(x)

x = Dropout(0.15, seed=42, name='Dropout_Post_Bottleneck')(latent_bottleneck)

# Salida sigmoidal
outputs = Dense(1, activation='sigmoid', name='Salida_Clasificador_ResNet_MLP')(x)

model_resnet_mlp = Model(inputs=inputs, outputs=outputs, name='ResNet_MLP_Encryption_Tuned')
model_resnet_mlp.summary()

# ===================================================================
# 3. COMPILACIÓN CON LABEL SMOOTHING Y CALLBACKS ADAPTATIVOS
# ===================================================================
model_resnet_mlp.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), # LR reducido a la mitad
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05), # Mitiga sobreajuste en probabilidades
    metrics=['accuracy']
)

callbacks_list = [
    EarlyStopping(
        monitor='val_loss', 
        patience=6, 
        restore_best_weights=True, 
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=2, 
        min_lr=1e-5, 
        verbose=1 # Reduce el paso para eliminar oscilaciones en serrucho
    ),
    ModelCheckpoint(
        filepath=os.path.join(BASE_DIR, 'models', 'deep_learning', 'best_resnet_mlp_encryption_model.keras'),
        monitor='val_loss', 
        save_best_only=True, 
        verbose=1
    )
]

print("\nIniciando entrenamiento sintonizado del ResNet-MLP...")
history = model_resnet_mlp.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=256, # Batch size duplicado para estabilizar BatchNormalization
    callbacks=callbacks_list,
    verbose=1
)

# ===================================================================
# 3.5. GENERACIÓN Y ALMACENAMIENTO DE CURVAS DE APRENDIZAJE
# ===================================================================
print("\nGenerando gráficos de convergencia (Curvas de Loss y Accuracy)...")
output_perf_path = os.path.join(BASE_DIR, 'reports', 'figures', 'curvas_rendimiento_resnet_mlp_encryption.png')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

ax1.plot(history.history['loss'], label='Pérdida Entrenamiento (Train Loss)', color='#1f77b4', linewidth=2.5)
ax1.plot(history.history['val_loss'], label='Pérdida Validación (Val Loss)', color='#ff7f0e', linewidth=2.5)
ax1.set_title('Evolución del Loss (Binary Crossentropy)', fontweight='bold', pad=10)
ax1.set_xlabel('Épocas de Entrenamiento')
ax1.set_ylabel('Valor de Pérdida')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.5)

ax2.plot(history.history['accuracy'], label='Exactitud Entrenamiento (Train Acc)', color='#2ca02c', linewidth=2.5)
ax2.plot(history.history['val_accuracy'], label='Exactitud Validación (Val Acc)', color='#d62728', linewidth=2.5)
ax2.set_title('Evolución de la Exactitud (Accuracy Binaria)', fontweight='bold', pad=10)
ax2.set_xlabel('Épocas de Entrenamiento')
ax2.set_ylabel('Tasa de Acierto')
ax2.legend(loc='lower right')
ax2.grid(True, linestyle=':', alpha=0.5)

plt.suptitle('Métricas de Entrenamiento - Arquitectura ResNet-MLP Sintonizada (Cifrado)', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(output_perf_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Curvas guardadas exitosamente en: '{output_perf_path}'")

# ===================================================================
# 4. EXTRACCIÓN Y VALIDACIÓN VISUAL t-SNE DEL ESPACIO LATENTE (16D)
# ===================================================================
print("\nExtrayendo representaciones latentes desde el Bottleneck...")
extractor_model = Model(
    inputs=model_resnet_mlp.input, 
    outputs=model_resnet_mlp.get_layer('Embedding_Encryption_ResNet_MLP').output
)
embeddings_16d = extractor_model.predict(X_test)

print("Calculando proyección t-SNE sobre el conjunto de prueba independiente...")
num_muestras_tsne = min(4000, len(embeddings_16d))
indices = np.random.choice(len(embeddings_16d), size=num_muestras_tsne, replace=False)

embeddings_muestra = embeddings_16d[indices]
y_muestra_codificada = y_test[indices]
labels_muestra_reales = le_encryption.inverse_transform(y_muestra_codificada)

tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_muestra)

fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
sns.scatterplot(
    x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], 
    hue=labels_muestra_reales, palette=['#1f77b4', '#ff7f0e'],
    alpha=0.7, s=15, ax=ax
)

ax.set_title("Visualización t-SNE del Espacio Latente (ResNet-MLP Sintonizada - Cifrado)", pad=15)
ax.set_xlabel("Dimensión t-SNE 1")
ax.set_ylabel("Dimensión t-SNE 2")
ax.legend(title="Estado de Encriptación", loc='best')
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
ruta_grafico = os.path.join(BASE_DIR, 'reports', 'figures', 'espacio_latente_tsne_resnet_mlp_encryption.png')
plt.savefig(ruta_grafico, dpi=300)
plt.close()

print(f"\n[PROCESO COMPLETADO] Modelo guardado y t-SNE exportado en: '{ruta_grafico}'")