import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de tipografía y estilo académico para el artículo
plt.rcParams.update({
    'font.size': 8,          # Tamaño reducido para mapas de calor densos
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.titlesize': 13
})

# ===================================================================
# 1. CONFIGURACIÓN DE RUTA Y CARGA DE DATOS
# ===================================================================
BASE_DIR = r'C:\Users\i21327\Desktop\Tesis\Modelo_Hibrido_Ligero_Tesis'
ruta_entrada = os.path.join(BASE_DIR, 'data', 'processed', 'Darknet_Etiquetado_Grupos.parquet')

print(f"Cargando dataset para Matriz de Spearman: {ruta_entrada}")
df = pd.read_parquet(ruta_entrada, engine='pyarrow')
df.columns = df.columns.str.strip()

# Columns to omit (identifiers and labels)
columnas_omitir = ['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Timestamp', 'Label', 'Label.1', 'Encryption_Label']
columnas_numericas = [col for col in df.select_dtypes(include=[np.number]).columns if col not in columnas_omitir]

# ===================================================================
# 2. FILTRADO METODOLÓGICO DE VARIANZA CERO (Sustento del Artículo)
# ===================================================================
# Identificamos columnas donde todos los valores son 0 (su varianza es 0)
columnas_activas = [col for col in columnas_numericas if (df[col] != 0).any()]

print(f"Total características numéricas: {len(columnas_numericas)}")
print(f"Características activas (excluyendo varianza cero): {len(columnas_activas)}")

# Para evitar un gráfico saturado e ilegible de 60x60 en el artículo,
# seleccionamos un subconjunto altamente representativo de los descriptores activos esenciales
variables_heatmap = [
    'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets', 
    'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s', 
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd Subflow Bytes', 'Bwd Subflow Bytes',
    'FWD Init Win Bytes', 'Bwd Init Win Bytes', 'Fwd Act Data Pkts', 'Fwd Seg Size Min'
]

# Asegurar que solo graficamos las que existen y pasaron el filtro de actividad
variables_finales = [v for v in variables_heatmap if v in columnas_activas]

# ===================================================================
# 3. CÁLCULO DE SPEARMAN Y GENERACIÓN DEL GRÁFICO (plt.subplots)
# ===================================================================
print("\nCalculando matriz de correlación de Spearman para variables activas...")
# Reemplazar infinitos por NaN para proteger la correlación
df_corr_temp = df[variables_finales].replace([np.inf, -np.inf], np.nan).dropna()
matriz_spearman = df_corr_temp.corr(method='spearman')

# Creamos la figura usando subplots de forma reglamentaria
fig, ax = plt.subplots(figsize=(14, 12), dpi=300)

# Mapa de calor divergente profesional
sns.heatmap(
    matriz_spearman,
    annot=True,              # Mostrar los coeficientes numéricos
    fmt=".2f",               # Dos decimales de precisión
    cmap='coolwarm',         # Paleta térmica estándar en papers
    vmin=-1.0, vmax=1.0,     # Límites matemáticos del coeficiente
    linewidths=0.3,          # Línea delgada de separación
    cbar_kws={"label": "Coeficiente de Correlación de Spearman ($\\rho$)"},
    square=True,             # Celdas perfectamente cuadradas
    ax=ax
)

ax.set_title("Matriz de Correlación de Spearman para Descriptores de Tráfico Cifrado", pad=20)

# Ajuste estricto de etiquetas para evitar colisiones u omisiones
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Guardar directo en disco con alta densidad de pixeles
ruta_salida_png = os.path.join(BASE_DIR, 'darknet_correlacion_spearman.png')
plt.savefig(ruta_salida_png, dpi=300)
plt.close()

print(f"\n[PROCESO GRÁFICO TERMINADO] Imagen exportada con éxito en: {ruta_salida_png}")