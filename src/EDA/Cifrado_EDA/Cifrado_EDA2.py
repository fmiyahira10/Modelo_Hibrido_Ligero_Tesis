import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de tipografía y estilo estilo IEEE/Elsevier
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 13
})

# ===================================================================
# 1. CONFIGURACIÓN DE RUTAS RELATIVAS EXACTAS
# ===================================================================
BASE_DIR = r'C:\Users\i21327\Desktop\Tesis\Modelo_Hibrido_Ligero_Tesis'
ruta_entrada = os.path.join(BASE_DIR, 'data', 'processed', 'Darknet_Etiquetado_Grupos.parquet')

print(f"Leyendo matriz de Darknet desde: {ruta_entrada}")
if not os.path.exists(ruta_entrada):
    raise FileNotFoundError(f"Ruta inválida. Asegúrate de que el archivo existe en: {ruta_entrada}")

df = pd.read_parquet(ruta_entrada, engine='pyarrow')
df.columns = df.columns.str.strip()

columnas_omitir = ['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Timestamp', 'Label', 'Label.1', 'Encryption_Label']
columnas_crudas = [col for col in df.select_dtypes(include=[np.number]).columns if col not in columnas_omitir]

# ===================================================================
# GRÁFICO 1: ANÁLISIS DE SPARSITY (Ceros Ordenados para Selección)
# ===================================================================
print("\nGenerando Gráfico 1: Análisis de Características de Varianza Cero...")

# Calcular porcentajes reales y ordenar de forma ascendente/descendente exacta
porcentaje_ceros = (df[columnas_crudas] == 0).mean() * 100
df_ceros = pd.DataFrame({'Característica': porcentaje_ceros.index, 'Porcentaje': porcentaje_ceros.values})
df_ceros_filtrado = df_ceros[df_ceros['Porcentaje'] > 10].sort_values(by='Porcentaje', ascending=True)

# Utilizar subplots de manera reglamentaria
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

sns.barplot(
    x='Porcentaje', 
    y='Característica', 
    data=df_ceros_filtrado, 
    palette='magma',
    hue='Característica',
    legend=False,
    ax=ax
)

# Línea de umbral metodológico de eliminación por varianza cero
ax.axvline(x=100.0, color='red', linestyle='--', linewidth=1.5, label='Varianza Cero ($100\%$ Constante)')
ax.set_title("Auditoría de Inactividad y Sparsity en Descriptores de Red (CIC-Darknet2020)", pad=15)
ax.set_xlabel("Porcentaje de Valores Cero ($0.0$) Encontrados (%)")
ax.set_ylabel("Variables Crudas de Red")
ax.legend(loc='lower left')
ax.grid(True, axis='x', linestyle=':', alpha=0.6)

plt.tight_layout()
# Guardar directamente sin usar plt.show()
ruta_grafico_1 = os.path.join(BASE_DIR, 'darknet_auditoria_sparsity.png')
plt.savefig(ruta_grafico_1, dpi=300)
plt.close()
print(f"-> Archivo guardado con éxito en: {ruta_grafico_1}")

# ===================================================================
# GRÁFICO 2: BOXPLOTS DE OUTLIERS DE CARACTERÍSTICAS DE FLUJO CLAVE
# ===================================================================
print("\nGenerando Gráfico 2: Diagrama de Cajas de Outliers...")

# Seleccionamos un grupo representativo de variables de flujo para que no se sature el Boxplot
variables_clave = [
    'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets', 
    'Total Length of Fwd Packet', 'Total Length of Bwd Packet', 
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean'
]

# Sustituir infinitos temporales y aplicar transformación logarítmica de visualización para legibilidad
df_log = df[variables_clave].copy().replace([np.inf, -np.inf], np.nan).dropna()
for col in variables_clave:
    # Asegurar valores positivos para evitar indeterminación en el logaritmo
    df_log[col] = np.log10(np.abs(df_log[col]) + 1)

fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

df_melted = df_log.melt(var_name='Descriptor', value_name='Magnitud Criptográfica ($\log_{10}(|x| + 1)$)')

sns.boxplot(
    x='Magnitud Criptográfica ($\log_{10}(|x| + 1)$)',
    y='Descriptor',
    data=df_melted,
    palette='Set2',
    orient='h',
    fliersize=2,
    linewidth=1.1,
    ax=ax
)

ax.set_title("Distribución de Outliers e Impacto Asimétrico en Características Clave de Cifrado", pad=15)
ax.set_ylabel("Descriptores de Flujo")
ax.grid(True, axis='x', linestyle=':', alpha=0.5)

plt.tight_layout()
ruta_grafico_2 = os.path.join(BASE_DIR, 'darknet_distribucion_outliers.png')
plt.savefig(ruta_grafico_2, dpi=300)
plt.close()
print(f"-> Archivo guardado con éxito en: {ruta_grafico_2}")

print("\n[PROCESO VISUAL COMPLETADO] Gráficos listos para ser insertados en tu documento.")