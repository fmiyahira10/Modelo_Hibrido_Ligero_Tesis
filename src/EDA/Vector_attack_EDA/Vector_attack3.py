import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración estética para artículos (estilo IEEE/Elsevier)
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# ===================================================================
# 1. CARGA DEL DATASET PURIFICADO
# ===================================================================
archivo_limpio = r'C:\Users\Felix\Desktop\Tesis\data\processed\Attack_Dataset_Clean.parquet'
print(f"Cargando dataset purificado para análisis de outliers: {archivo_limpio}")
df = pd.read_parquet(archivo_limpio, engine='pyarrow')

columnas_features = [
    'flow_duration', 'fwd_packets', 'bwd_packets', 'fwd_bytes', 'bwd_bytes',
    'fwd_packet_len_mean', 'bwd_packet_len_mean', 'fwd_iat_mean', 'bwd_iat_mean',
    'fwd_tcp_window', 'bwd_tcp_window', 'flow_packets_per_sec', 'flow_bytes_per_sec'
]

# ===================================================================
# 2. CÁLCULO MATEMÁTICO DE OUTLIERS (Método de Tukey / IQR)
# ===================================================================
print("\n[1/2] Calculando volumen de valores atípicos mediante el método IQR...")

reporte_outliers = []
total_filas = len(df)

for col in columnas_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Definición matemática de límites de Tukey
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    # Conteo de registros fuera de los límites
    num_outliers = ((df[col] < limite_inferior) | (df[col] > limite_superior)).sum()
    porcentaje = (num_outliers / total_filas) * 100
    
    reporte_outliers.append({
        'Característica': col,
        'Límite Inferior': limite_inferior,
        'Límite Superior': limite_superior,
        'Cant. Outliers': num_outliers,
        'Porcentaje (%)': round(porcentaje, 2)
    })

df_outliers_reporte = pd.DataFrame(reporte_outliers)
print("\n=== REPORTE CUANTITATIVO DE OUTLIERS ===")
print(df_outliers_reporte.to_string(index=False))

# ===================================================================
# 3. GENERACIÓN DEL GRÁFICO ACADÉMICO (Boxplots con Escala Logarítmica)
# ===================================================================
print("\n[2/2] Generando diagrama de cajas (Boxplots) adaptado...")

# Creamos una copia temporal aplicando log10(x + 1) SOLO para la visualización 
# Esto permite compactar los extremos y que el jurado vea la distribución real
df_log = df[columnas_features].copy()
for col in columnas_features:
    df_log[col] = np.log10(df_log[col] + 1)

# Configurar figura para subplots eficientes
plt.figure(figsize=(14, 8), dpi=300)

# Derivar un dataframe largo (melt) para graficar todo en un solo heatmap/boxplot ordenado
df_melted = df_log.melt(var_name='Característica', value_name='Valor (Escala Log10)')

sns.boxplot(
    x='Valor (Escala Log10)', 
    y='Característica', 
    data=df_melted, 
    palette='Set3',
    orient='h',
    fliersize=2, # Tamaño de los puntos outliers
    linewidth=1.2
)

plt.title("Distribución de Características e Identificación de Outliers (Transformación Log10)", pad=15)
plt.xlabel("Magnitud del Flujo (log10(x + 1))")
plt.ylabel("Descriptores de Tráfico")
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('grafico_outliers_boxplots.png', dpi=300)
plt.close()

print("\n[PROCESO TERMINADO] Gráfico 'grafico_outliers_boxplots.png' exportado.")