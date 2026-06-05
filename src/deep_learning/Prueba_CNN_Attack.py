import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Configuración estética profesional para el artículo
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
})

# ===================================================================
# 1. CONFIGURACIÓN DE RUTAS Y CARGA DE DATOS DE PRUEBA
# ===================================================================
BASE_DIR = r'C:\Users\i21327\Desktop\Tesis\Modelo_Hibrido_Ligero_Tesis'

print("Cargando datos de prueba independientes...")
X_test = np.load(os.path.join(BASE_DIR,'data' ,'final', 'X_test_attack.npy'))
y_test = np.load(os.path.join(BASE_DIR,'data' ,'final', 'y_test_attack.npy'))

# Recuperar el codificador de etiquetas para poner los nombres reales en el gráfico
le_attack = joblib.load(os.path.join(BASE_DIR,'models', 'label_encoder_attack.pkl'))

# Formatear el tensor para la entrada Conv1D (muestras, 13, 1)
X_test_reshaped = np.expand_dims(X_test, axis=-1)

# ===================================================================
# 2. CARGA DEL MEJOR MODELO Y RECORTE DE LA CAPA LATENTE
# ===================================================================
print("\nCargando el mejor modelo convolucional guardado (Época 6)...")
modelo_completo = load_model(os.path.join(BASE_DIR,'src', 'Embedding', 'best_cnn_attack_model.keras'))

# Creamos un submodelo truncado que tenga la misma entrada, 
# pero cuya salida sea exclusivamente la capa dense llamada 'Embedding_Attack'
modelo_extractor = Model(
    inputs=modelo_completo.input,
    outputs=modelo_completo.get_layer('Embedding_Attack').output
)

print("\nGenerando representaciones latentes (Embeddings) de los datos de prueba...")
embeddings_16d = modelo_extractor.predict(X_test_reshaped)
print(f"-> Dimensiones de la matriz de embeddings: {embeddings_16d.shape} (¡Pasamos de 13 descriptores crudos a 16 rasgos abstractos!)")

# ===================================================================
# 3. REDUCCIÓN DE DIMENSIONALIDAD CON t-SNE (16D -> 2D)
# ===================================================================
print("\nEjecutando algoritmo t-SNE sobre el espacio latente...")
# Tomamos una muestra de control de hasta 5,000 registros para que el t-SNE compute rápido y no sature el gráfico
num_muestras_grafico = min(60000, len(embeddings_16d))
indices_aleatorios = np.random.choice(len(embeddings_16d), size=num_muestras_grafico, replace=False)

embeddings_muestra = embeddings_16d[indices_aleatorios]
y_muestra_codificada = y_test[indices_aleatorios]
labels_muestra_reales = le_attack.inverse_transform(y_muestra_codificada)

tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_muestra)

# ===================================================================
# 4. GENERACIÓN DEL GRÁFICO ACADÉMICO (plt.subplots)
# ===================================================================
print("\nDibujando mapa de dispersión del espacio latente...")
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

# Diccionario de colores distinguibles para tus 6 macro-clases de ataque
paleta_colores = sns.color_palette("bright", len(le_attack.classes_))

sns.scatterplot(
    x=embeddings_2d[:, 0], 
    y=embeddings_2d[:, 1], 
    hue=labels_muestra_reales,
    palette=paleta_colores,
    alpha=0.7,
    s=15, # Tamaño de los puntos
    ax=ax
)

ax.set_title("Visualización t-SNE del Espacio Latente Generado por CNN_Attack (Set de Prueba)", pad=15)
ax.set_xlabel("Dimensión t-SNE 1")
ax.set_ylabel("Dimensión t-SNE 2")
ax.legend(title="Macro-Vectores de Ataque", bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
ruta_grafico = os.path.join(BASE_DIR, 'espacio_latente_tsne_attacks.png')
plt.savefig(ruta_grafico, dpi=300)
plt.close()

print(f"\n[PROCESO VISUAL COMPLETADO] Imagen exportada con éxito en: {ruta_grafico}")