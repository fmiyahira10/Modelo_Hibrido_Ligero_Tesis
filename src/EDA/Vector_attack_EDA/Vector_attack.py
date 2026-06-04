import pandas as pd
import numpy as np

# ===================================================================
# 1. CARGA DEL DATASET DE ATAQUES HOMOLOGADO
# ===================================================================
archivo_ataques = r'C:\Users\Felix\Desktop\Tesis\data\processed\Attack_Dataset_Homologado.parquet'
print(f"Abriendo {archivo_ataques} para análisis estadístico profundo...")
df = pd.read_parquet(archivo_ataques, engine='pyarrow')

# Lista de las 13 características de flujo a evaluar
columnas_analisis = [
    'flow_duration', 'fwd_packets', 'bwd_packets', 'fwd_bytes', 'bwd_bytes',
    'fwd_packet_len_mean', 'bwd_packet_len_mean', 'fwd_iat_mean', 'bwd_iat_mean',
    'fwd_tcp_window', 'bwd_tcp_window', 'flow_packets_per_sec', 'flow_bytes_per_sec'
]

# Target
target = 'attack_vector'

# ===================================================================
# 2. AUDITORÍA DE CALIDAD DE DATOS (NaN, Infinitos y Ceros)
# ===================================================================
print("\n[1/3] Ejecutando auditoría de calidad de datos numéricos...")

reporte_calidad = []
for col in columnas_analisis:
    total_registros = len(df)
    nulos = df[col].isna().sum()
    infinitos = np.isinf(df[col]).sum()
    ceros = (df[col] == 0).sum()
    
    reporte_calidad.append({
        'Característica': col,
        'Nulos (NaN)': nulos,
        'Infinitos (Inf)': infinitos,
        'Valores Cero (0)': ceros,
        '% Ceros': (ceros / total_registros) * 100
    })

df_calidad = pd.DataFrame(reporte_calidad)
print("\n=== REPORTE DE INTEGRIDAD DE VARIABLES ===")
print(df_calidad.to_string(index=False))

# ===================================================================
# 3. ANÁLISIS ESTADÍSTICO DESCRIPTIVO TOTAL
# ===================================================================
print("\n[2/3] Calculando métricas descriptivas centrales y de dispersión...")

# Reemplazamos momentáneamente los Inf por NaN para que no alteren la media y el max
df_temp = df[columnas_analisis].replace([np.inf, -np.inf], np.nan)
descriptivos = df_temp.describe().T[['mean', 'std', 'min', '50%', 'max']]
descriptivos.rename(columns={'50%': 'median'}, inplace=True)

print("\n=== RESUMEN ESTADÍSTICO DE TRAFICO ===")
print(descriptivos.to_string())

# ===================================================================
# 4. COMPORTAMIENTO DE LAS CARACTERÍSTICAS POR MACRO-CLASE
# ===================================================================
print("\n[3/3] Analizando perfiles de tráfico promedio por Macro-Clase...")

# Calculamos la media de cada característica agrupada por tipo de ataque
perfil_ataques = df.replace([np.inf, -np.inf], np.nan).groupby(target)[columnas_analisis].mean().T

print("\n=== MATRIZ DE COMPORTAMIENTO MEDIO POR ATAQUE ===")
print(perfil_ataques.to_string())
print("\n" + "="*70)
print("Análisis completado. Por favor, compárteme estos tres bloques de resultados.")
print("="*70)