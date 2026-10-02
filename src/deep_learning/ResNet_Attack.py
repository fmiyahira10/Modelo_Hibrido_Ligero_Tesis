import os
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Add, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estética académica para publicaciones
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
# Raíz del proyecto: sube 2 niveles desde src/deep_learning/
BASE_DIR = Path(__file__).resolve().parents[2]

print("Cargando matrices NumPy del carril de vectores de ataque...")
X_train = np.load(os.path.join(BASE_DIR, 'data', 'final', 'X_train_attack.npy'))
X_val = np.load(os.path.join(BASE_DIR, 'data', 'final', 'X_val_attack.npy'))
X_test = np.load(os.path.join(BASE_DIR, 'data', 'final', 'X_test_attack.npy'))

y_train = np.load(os.path.join(BASE_DIR, 'data', 'final', 'y_train_attack.npy'))
y_val = np.load(os.path.join(BASE_DIR, 'data', 'final', 'y_val_attack.npy'))
y_test = np.load(os.path.join(BASE_DIR, 'data', 'final', 'y_test_attack.npy'))

le_attack = joblib.load(os.path.join(BASE_DIR, 'models', 'scalers_encoders', 'label_encoder_attack.pkl'))
num_clases = len(np.unique(y_train))

input_dim = X_train.shape[1] # 13 variables
print(f"Dimensiones de entrada   : {input_dim} características")
print(f"Tensor Entrenamiento     : {X_train.shape}")
print(f"Tensor Validación        : {X_val.shape}")
print(f"Tensor Prueba            : {X_test.shape}")

# ===================================================================
# 2. ARQUITECTURA RESNET-MLP PARA DATOS TABULARES
# ===================================================================
print("\nDiseñando arquitectura ResNet-MLP (Submodelo A - Ataques)...")

def residual_block_mlp(x, units, dropout_rate=0.2, prefix="res_block"):
    """Bloque Residual Tabular: F(x) + x"""
    shortcut = x
    h = Dense(units, kernel_regularizer=tf.keras.regularizers.l2(0.001), name=f"{prefix}_dense1")(x)
    h = BatchNormalization(name=f"{prefix}_bn1")(h)
    h = Activation('relu', name=f"{prefix}_act1")(h)
    h = Dropout(dropout_rate, seed=42, name=f"{prefix}_drop")(h)
    h = Dense(units, kernel_regularizer=tf.keras.regularizers.l2(0.001), name=f"{prefix}_dense2")(h)
    h = BatchNormalization(name=f"{prefix}_bn2")(h)
    
    out = Add(name=f"{prefix}_add")([shortcut, h])
    out = Activation('relu', name=f"{prefix}_out")(out)
    return out

inputs = Input(shape=(input_dim,), name='Input_Descriptores_Ataques')

# Proyección inicial para expandir las variables a la dimensión de los bloques residuales
x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='Proyeccion_Inicial')(inputs)
x = BatchNormalization(name='BatchNorm_Inicial')(x)

# Bloques Residuales Densos (Skip Connections)
x = residual_block_mlp(x, units=64, dropout_rate=0.25, prefix="ResBlock_1")
x = residual_block_mlp(x, units=64, dropout_rate=0.25, prefix="ResBlock_2")

# Capa de regularización pre-bottleneck
x = Dropout(0.3, seed=42, name='Dropout_Pre_Bottleneck')(x)

# CAPA BOTTLENECK: Representación latente de 16 dimensiones con regularización L2
latent_bottleneck = Dense(
    16, 
    activation='relu', 
    kernel_regularizer=tf.keras.regularizers.l2(0.001), 
    name='Embedding_Attack_ResNet_MLP'
)(x)

x = Dropout(0.2, seed=42, name='Dropout_Post_Bottleneck')(latent_bottleneck)

# Capa de clasificación supervisada (6 macro-clases de ataque)
outputs = Dense(num_clases, activation='softmax', name='Salida_Clasificador_ResNet_MLP')(x)

model_resnet_mlp = Model(inputs=inputs, outputs=outputs, name='ResNet_MLP_Attack_Full')
model_resnet_mlp.summary()

# ===================================================================
# 3. COMPILACIÓN Y ENTRENAMIENTO SUPERVISADO
# ===================================================================
model_resnet_mlp.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_list = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        filepath=os.path.join(BASE_DIR, 'models', 'deep_learning', 'best_resnet_mlp_attack_model.keras'),
        monitor='val_loss', save_best_only=True, verbose=1
    )
]

print("\nIniciando entrenamiento supervisado del ResNet-MLP de Ataques...")
history = model_resnet_mlp.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=256,
    callbacks=callbacks_list,
    verbose=1
)

# ===================================================================
# 3.5. GENERACIÓN Y ALMACENAMIENTO DE CURVAS DE APRENDIZAJE
# ===================================================================
print("\nGenerando gráficos de convergencia (Curvas de Loss y Accuracy)...")
output_perf_path = os.path.join(BASE_DIR, 'reports', 'figures', 'curvas_rendimiento_resnet_mlp_attack.png')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

# Gráfica 1: Pérdida (Sparse Categorical Crossentropy)
ax1.plot(history.history['loss'], label='Pérdida Entrenamiento (Train Loss)', color='#1f77b4', linewidth=2.5)
ax1.plot(history.history['val_loss'], label='Pérdida Validación (Val Loss)', color='#ff7f0e', linewidth=2.5)
ax1.set_title('Evolución del Loss (Multiclase)', fontweight='bold', pad=10)
ax1.set_xlabel('Épocas de Entrenamiento')
ax1.set_ylabel('Valor de Pérdida')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.5)

# Gráfica 2: Exactitud (Accuracy)
ax2.plot(history.history['accuracy'], label='Exactitud Entrenamiento (Train Acc)', color='#2ca02c', linewidth=2.5)
ax2.plot(history.history['val_accuracy'], label='Exactitud Validación (Val Acc)', color='#d62728', linewidth=2.5)
ax2.set_title('Evolución de la Exactitud (Accuracy)', fontweight='bold', pad=10)
ax2.set_xlabel('Épocas de Entrenamiento')
ax2.set_ylabel('Tasa de Acierto')
ax2.legend(loc='lower right')
ax2.grid(True, linestyle=':', alpha=0.5)

plt.suptitle('Métricas de Entrenamiento - Arquitectura ResNet-MLP (Vectores de Ataque)', fontsize=14, fontweight='bold', y=0.98)
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
    outputs=model_resnet_mlp.get_layer('Embedding_Attack_ResNet_MLP').output
)
embeddings_16d = extractor_model.predict(X_test)

print("Calculando proyección t-SNE sobre el conjunto de prueba independiente...")
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

ax.set_title("Visualización t-SNE del Espacio Latente (ResNet-MLP - Ataques)", pad=15)
ax.set_xlabel("Dimensión t-SNE 1")
ax.set_ylabel("Dimensión t-SNE 2")
ax.legend(title="Macro-Vectores de Ataque", bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
ruta_grafico = os.path.join(BASE_DIR, 'reports', 'figures', 'espacio_latente_tsne_resnet_mlp_attack.png')
plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n[PROCESO COMPLETADO] Modelo guardado y t-SNE exportado en: '{ruta_grafico}'")