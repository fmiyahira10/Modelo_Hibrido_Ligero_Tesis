import os
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estética académica
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
})

# ===================================================================
# 1. CONFIGURACIÓN DE RUTAS RELATIVAS Y CARGA DE MATRICES (ATAQUES)
# ===================================================================
BASE_DIR = Path(__file__).resolve().parents[2]

print("Cargando matrices NumPy del carril de vectores de ataque...")
X_train = np.load(os.path.join(BASE_DIR, 'data', 'final', 'X_train_attack.npy'))
X_val = np.load(os.path.join(BASE_DIR, 'data', 'final', 'X_val_attack.npy'))
X_test = np.load(os.path.join(BASE_DIR, 'data', 'final', 'X_test_attack.npy'))
y_test = np.load(os.path.join(BASE_DIR, 'data', 'final', 'y_test_attack.npy'))

le_attack = joblib.load(os.path.join(BASE_DIR, 'models', 'scalers_encoders', 'label_encoder_attack.pkl'))

input_dim = X_train.shape[1] # 13 características
print(f"Dimensiones de entrada : {input_dim} variables")

# ===================================================================
# 2. ARQUITECTURA SIMÉTRICA Y COMPACTA (ESTABILIZADA)
# ===================================================================
print("\nDiseñando arquitectura optimizada para Autoencoder de Ataques...")

inputs = Input(shape=(input_dim,), name='Input_Descriptores_Ataques')

# --- BLOQUE ENCODER ---
# Expansión no lineal inicial para capturar correlaciones cruzadas complejas
enc = Dense(64, activation='relu', name='Encoder_Expansion')(inputs)
enc = BatchNormalization(name='BatchNorm_Enc_1')(enc)

# Reducción intermedia guiada
enc = Dense(32, activation='relu', name='Encoder_Compression')(enc)
enc = BatchNormalization(name='BatchNorm_Enc_2')(enc)

# Cuello de botella de 16 dimensiones latentes con regularización L2 controlada
latent_bottleneck = Dense(
    16, 
    activation='relu', 
    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    name='Embedding_Attack_AE'
)(enc)

# --- BLOQUE DECODER (Simétrico, sin Dropout disruptivo) ---
dec = Dense(32, activation='relu', name='Decoder_Decompression')(latent_bottleneck)
dec = BatchNormalization(name='BatchNorm_Dec_1')(dec)

dec = Dense(64, activation='relu', name='Decoder_Expansion')(dec)
dec = BatchNormalization(name='BatchNorm_Dec_2')(dec)

# Salida reconstructiva lineal continua
outputs = Dense(input_dim, activation='linear', name='Reconstruction_Output')(dec)

autoencoder_attack = Model(inputs=inputs, outputs=outputs, name='Autoencoder_Attack_Robust')
autoencoder_attack.summary()

# ===================================================================
# 3. COMPILACIÓN CON GRADIENT CLIPPING Y LOG-COSH / HUBER (ROBUSTO A OUTLIERS)
# ===================================================================
# clipnorm=1.0 previene la explosión de gradientes que causó el salto a 4e6
# Huber loss amortigua el impacto de los valores extremos en variables de tráfico de red
autoencoder_attack.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),
    loss='huber',
    metrics=['mae']
)

callbacks_list = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        filepath=os.path.join(BASE_DIR, 'models', 'deep_learning', 'best_autoencoder_attack_model.keras'),
        monitor='val_loss', save_best_only=True, verbose=1
    )
]

print("\nIniciando entrenamiento estabilizado del Autoencoder...")
history = autoencoder_attack.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=25,
    batch_size=256,
    callbacks=callbacks_list,
    verbose=1
)

# ===================================================================
# 3.5. GENERACIÓN Y ALMACENAMIENTO DE CURVAS DE APRENDIZAJE
# ===================================================================
print("\nGenerando gráficos de convergencia del Autoencoder...")
output_perf_path = os.path.join(BASE_DIR, 'reports', 'figures', 'curvas_rendimiento_autoencoder_attack.png')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

ax1.plot(history.history['loss'], label='Pérdida Entrenamiento (Train Huber)', color='#1f77b4', linewidth=2.5)
ax1.plot(history.history['val_loss'], label='Pérdida Validación (Val Huber)', color='#ff7f0e', linewidth=2.5)
ax1.set_title('Evolución de la Pérdida (Huber Loss Robusta)', fontweight='bold', pad=10)
ax1.set_xlabel('Épocas de Entrenamiento')
ax1.set_ylabel('Huber Loss')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.5)

ax2.plot(history.history['mae'], label='Error Entrenamiento (Train MAE)', color='#2ca02c', linewidth=2.5)
ax2.plot(history.history['val_mae'], label='Error Validación (Val MAE)', color='#d62728', linewidth=2.5)
ax2.set_title('Evolución del Error Absoluto (MAE)', fontweight='bold', pad=10)
ax2.set_xlabel('Épocas de Entrenamiento')
ax2.set_ylabel('Mean Absolute Error')
ax2.legend(loc='upper right')
ax2.grid(True, linestyle=':', alpha=0.5)

plt.suptitle('Convergencia del Autoencoder Reconstructivo - Vectores de Ataque', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(output_perf_path, dpi=300, bbox_inches='tight')
plt.show()

# ===================================================================
# 4. EXTRACCIÓN Y VALIDACIÓN VISUAL t-SNE
# ===================================================================
print("\nExtrayendo representaciones latentes desde el Bottleneck...")
modelo_extractor = Model(
    inputs=autoencoder_attack.input, 
    outputs=autoencoder_attack.get_layer('Embedding_Attack_AE').output
)
embeddings_16d = modelo_extractor.predict(X_test)

print("Calculando proyección t-SNE...")
num_muestras_tsne = min(5000, len(embeddings_16d))
indices = np.random.choice(len(embeddings_16d), size=num_muestras_tsne, replace=False)

embeddings_muestra = embeddings_16d[indices]
y_muestra_codificada = y_test[indices]
labels_muestra_reales = le_attack.inverse_transform(y_muestra_codificada)

tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_muestra)

fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
paleta_colores = sns.color_palette("bright", len(le_attack.classes_))

sns.scatterplot(
    x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], 
    hue=labels_muestra_reales, palette=paleta_colores,
    alpha=0.7, s=15, ax=ax
)

ax.set_title("Visualización t-SNE del Espacio Latente (Autoencoder Robusto)", pad=15)
ax.set_xlabel("Dimensión t-SNE 1")
ax.set_ylabel("Dimensión t-SNE 2")
ax.legend(title="Macro-Vectores de Ataque", bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
ruta_grafico = os.path.join(BASE_DIR, 'reports', 'figures', 'espacio_latente_tsne_autoencoder_attack.png')
plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n[PROCESO COMPLETADO] Autoencoder guardado y t-SNE exportado en: '{ruta_grafico}'")