import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estética académica (Estilo IEEE/Elsevier)
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14
})

# ===================================================================
# 1. CARGA Y LIMPIEZA INICIAL DE SEGURIDAD (Basada en Diagnóstico)
# ===================================================================
archivo_parquet = r'C:\Users\Felix\Desktop\Tesis\data\processed\Attack_Dataset_Homologado.parquet'
print("Cargando dataset para procesamiento estadístico y gráfico...")
df = pd.read_parquet(archivo_parquet, engine='pyarrow')

columnas_analisis = [
    'flow_duration', 'fwd_packets', 'bwd_packets', 'fwd_bytes', 'bwd_bytes',
    'fwd_packet_len_mean', 'bwd_packet_len_mean', 'fwd_iat_mean', 'bwd_iat_mean',
    'fwd_tcp_window', 'bwd_tcp_window', 'flow_packets_per_sec', 'flow_bytes_per_sec'
]

# --- CORRECCIÓN DE VALORES NEGATIVOS ANÓMALOS ---
print("Aplicando filtros de validez matemática...")
for col in ['flow_duration', 'flow_packets_per_sec', 'flow_bytes_per_sec']:
    df = df[df[col] >= 0]

# Tratamiento de infinitos residuales
df[columnas_analisis] = df[columnas_analisis].replace([np.inf, -np.inf], np.nan)
df.dropna(subset=columnas_analisis, inplace=True)

# ===================================================================
# 2. GRÁFICO 1: MATRIZ DE CORRELACIÓN DE SPEARMAN
# ===================================================================
print("Calculando Matriz de Correlación de Spearman (No lineal)...")
plt.figure(figsize=(12, 10), dpi=300)

# Calculamos Spearman debido a la alta asimetría detectada en las desviaciones
corr_matrix = df[columnas_analisis].corr(method='spearman')

# Crear mapa de calor limpio
sns.heatmap(
    corr_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap='coolwarm', 
    linewidths=0.5, 
    cbar_kws={"label": "Coeficiente de Correlación"},
    square=True
)

plt.title("Matriz de Correlación de Spearman para Características de Flujo (IDS)", pad=20)
plt.tight_layout()
plt.savefig('grafico_correlacion_spearman.png', dpi=300)
plt.close()
print("-> Gráfico 'grafico_correlacion_spearman.png' guardado con éxito.")

# ===================================================================
# 3. GRÁFICO 2: AUDITORÍA VISUAL DE VALORES CERO (Sparsity)
# ===================================================================
print("Generando gráfico de distribución de vacíos y valores cero...")
plt.figure(figsize=(11, 6), dpi=300)

# Calculamos los porcentajes de ceros reales basados en tu salida de consola
porcentaje_ceros = (df[columnas_analisis] == 0).mean() * 100
df_ceros = pd.DataFrame({'Característica': porcentaje_ceros.index, 'Porcentaje': porcentaje_ceros.values})
df_ceros = df_ceros.sort_values(by='Porcentaje', ascending=False)

# Crear gráfico de barras
sns.barplot(
    x='Porcentaje', 
    y='Característica', 
    data=df_ceros, 
    palette='viridis',
    hue='Característica',
    legend=False
)

plt.axvline(x=20.0, color='red', linestyle='--', alpha=0.7, label='Umbral Crítico de Dispersión (20%)')
plt.title("Porcentaje de Valores Cero (0) por Característica Estadística", pad=15)
plt.xlabel("Porcentaje (%) de Presencia en el Dataset")
plt.ylabel("Características de Red")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('grafico_auditoria_ceros.png', dpi=300)
plt.close()
print("-> Gráfico 'grafico_auditoria_ceros.png' guardado con éxito.")

print("\n[PROCESO TERMINADO] Los gráficos están listos en alta resolución para tu artículo.")