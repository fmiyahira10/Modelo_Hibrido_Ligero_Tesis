import os
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

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

print("Cargando matrices NumPy del carril de cifrado...")
X_train = np.load(os.path.join(BASE_DIR, 'data', 'final', 'X_train_encryption.npy'))
X_val = np.load(os.path.join(BASE_DIR, 'data', 'final', 'X_val_encryption.npy'))
X_test = np.load(os.path.join(BASE_DIR, 'data', 'final', 'X_test_encryption.npy'))

y_test = np.load(os.path.join(BASE_DIR, 'data', 'final', 'y_test_encryption.npy'))

le_encryption = joblib.load(os.path.join(BASE_DIR, 'models', 'scalers_encoders', 'label_encoder_encryption.pkl'))

input_dim = X_train.shape[1] # 62 características activas
print(f"Dimensiones de entrada : {input_dim} variables activas")
print(f"Tensor Entrenamiento   : {X_train.shape}")
print(f"Tensor Validación      : {X_val.shape}")
print(f"Tensor Prueba          : {X_test.shape}")

# ===================================================================
# 2. ARQUITECTURA DEL AUTOENCODER DENSO BLINDADO CONTRA OVERFITTING
# ===================================================================
print("\nDiseñando arquitectura del Autoencoder Denso (Submodelo Alternativo B)...")

inputs = Input(shape=(input_dim,), name='Input_Descriptores_Cifrado')

# --- BLOQUE ENCODER (Compresión Progresiva: 62 -> 32 -> 16) ---
x = Dense(32, activation='relu', name='Encoder_Dense_1')(inputs)
x = BatchNormalization(name='BatchNorm_Enc_1')(x)
x = Dropout(0.3, seed=42, name='Dropout_Enc_1')(x)

# CUELLO DE BOTELLA (BOTTLENECK): 16 dimensiones latentes con regularización L2
latent_bottleneck = Dense(
    16, 
    activation='relu', 
    kernel_regularizer=tf.keras.regularizers.l2(0.001),
    name='Embedding_Encryption_AE'
)(x)

# --- BLOQUE DECODER (Reconstrucción Progresiva: 16 -> 32 -> 62) ---
x = Dropout(0.2, seed=42, name='Dropout_Dec_1')(latent_bottleneck)
x = Dense(32, activation='relu', name='Decoder_Dense_1')(x)
x = BatchNormalization(name='BatchNorm_Dec_1')(x)

# Capa de reconstrucción lineal: aproxima las 62 variables continuas escaladas
outputs = Dense(input_dim, activation='linear', name='Reconstruction_Output')(x)

autoencoder_encryption = Model(inputs=inputs, outputs=outputs, name='Autoencoder_Encryption_Full')
autoencoder_encryption.summary()

# ===================================================================
# 3. COMPILACIÓN Y ENTRENAMIENTO RECONSTRUCTIVO (MSE + MAE)
# ===================================================================
autoencoder_encryption.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

callbacks_list = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        filepath=os.path.join(BASE_DIR, 'models', 'deep_learning', 'best_autoencoder_encryption_model.keras'),
        monitor='val_loss', save_best_only=True, verbose=1
    )
]

print("\nIniciando entrenamiento no supervisado del Autoencoder de Cifrado...")
history = autoencoder_encryption.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=20,
    batch_size=128,
    callbacks=callbacks_list,
    verbose=1
)

# ===================================================================
# 3.5. GENERACIÓN Y ALMACENAMIENTO DE CURVAS DE APRENDIZAJE
# ===================================================================
print("\nGenerando gráficos de convergencia del Autoencoder de Cifrado...")
output_perf_path = os.path.join(BASE_DIR, 'reports', 'figures', 'curvas_rendimiento_autoencoder_encryption.png')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

# Gráfica 1: Pérdida Cuadrática Media (MSE)
ax1.plot(history.history['loss'], label='Pérdida Entrenamiento (Train MSE)', color='#1f77b4', linewidth=2.5)
ax1.plot(history.history['val_loss'], label='Pérdida Validación (Val MSE)', color='#ff7f0e', linewidth=2.5)
ax1.set_title('Evolución de la Función de Pérdida (MSE)', fontweight='bold', pad=10)
ax1.set_xlabel('Épocas de Entrenamiento')
ax1.set_ylabel('Mean Squared Error')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.5)

# Gráfica 2: Error Absoluto Medio (MAE)
ax2.plot(history.history['mae'], label='Error Entrenamiento (Train MAE)', color='#2ca02c', linewidth=2.5)
ax2.plot(history.history['val_mae'], label='Error Validación (Val MAE)', color='#d62728', linewidth=2.5)
ax2.set_title('Evolución del Error Absoluto (MAE)', fontweight='bold', pad=10)
ax2.set_xlabel('Épocas de Entrenamiento')
ax2.set_ylabel('Mean Absolute Error')
ax2.legend(loc='upper right')
ax2.grid(True, linestyle=':', alpha=0.5)

plt.suptitle('Métricas de Reconstrucción - Autoencoder Denso (Detección de Cifrado)', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(output_perf_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"¡Gráfica de curvas de aprendizaje guardada en: '{output_perf_path}'!")

# ===================================================================
# 4. EXTRACCIÓN Y VALIDACIÓN VISUAL t-SNE DEL ESPACIO LATENTE (16D)
# ===================================================================
print("\nExtrayendo representaciones latentes desde el Bottleneck...")
modelo_extractor = Model(
    inputs=autoencoder_encryption.input, 
    outputs=autoencoder_encryption.get_layer('Embedding_Encryption_AE').output
)
embeddings_16d = modelo_extractor.predict(X_test)

print("Calculando proyección t-SNE para evaluación del agrupamiento latente...")
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

ax.set_title("Visualización t-SNE del Espacio Latente (Autoencoder Denso - Cifrado)", pad=15)
ax.set_xlabel("Dimensión t-SNE 1")
ax.set_ylabel("Dimensión t-SNE 2")
ax.legend(title="Estado de Encriptación", loc='best')
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
ruta_grafico = os.path.join(BASE_DIR, 'reports', 'figures', 'espacio_latente_tsne_autoencoder_encryption.png')
plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n[PROCESO COMPLETADO] Autoencoder guardado y t-SNE exportado en: '{ruta_grafico}'")