import pandas as pd
import numpy as np

# ===================================================================
# 1. CARGA DEL DATASET MAESTRO DE ATAQUES
# ===================================================================
archivo_entrada = r'C:\Users\Felix\Desktop\Tesis\data\processed\Attack_Dataset_Homologado.parquet'
print(f"Iniciando limpieza fina sobre: {archivo_entrada}")
df = pd.read_parquet(archivo_entrada, engine='pyarrow')

columnas_features = [
    'flow_duration', 'fwd_packets', 'bwd_packets', 'fwd_bytes', 'bwd_bytes',
    'fwd_packet_len_mean', 'bwd_packet_len_mean', 'fwd_iat_mean', 'bwd_iat_mean',
    'fwd_tcp_window', 'bwd_tcp_window', 'flow_packets_per_sec', 'flow_bytes_per_sec'
]
target = 'attack_vector'

dimensiones_originales = df.shape
print(f"Dimensiones iniciales: {dimensiones_originales[0]:,} registros con {dimensiones_originales[1]} columnas.")

# ===================================================================
# 2. EJECUCIÓN DEL PIPELINE DE LIMPIEZA
# ===================================================================

# --- PASO A: Eliminación de Valores Negativos ---
print("\n[PASO A] Eliminando registros con inconsistencias temporales o de tasa (Valores < 0)...")
filtro_positivos = (df['flow_duration'] >= 0) & (df['flow_packets_per_sec'] >= 0) & (df['flow_bytes_per_sec'] >= 0)
df = df[filtro_positivos]
print(f"-> Registros tras remover negativos: {len(df):,}")

# --- PASO B: Remoción de Infinitos e Indeterminaciones ---
print("\n[PASO B] Buscando y removiendo valores Infinitos (Inf)...")
df[columnas_features] = df[columnas_features].replace([np.inf, -np.inf], np.nan)

# Dropna estricto sobre las columnas de características y el target
df.dropna(subset=columnas_features + [target], inplace=True)
print(f"-> Registros tras remover nulos/infinitos: {len(df):,}")

# --- PASO C: Remoción de Duplicados (Prevención de Data Leakage) ---
print("\n[PASO C] Eliminando flujos duplicados en la matriz de características...")
# Eliminamos duplicados basándonos en los valores de los 13 descriptores estadísticos
df.drop_duplicates(subset=columnas_features, keep='first', inplace=True)
print(f"-> Registros tras remover duplicados: {len(df):,}")

# ===================================================================
# 3. VERIFICACIÓN Y GUARDADO DE LA MATRIZ DEPURADA
# ===================================================================
print("\n" + "="*50)
print("             RESUMEN DEL PROCESO DE LIMPIEZA")
print("="*50)
print(f"Registros Iniciales : {dimensiones_originales[0]:,}")
print(f"Registros Finales   : {df.shape[0]:,}")
print(f"Total Eliminados    : {dimensiones_originales[0] - df.shape[0]:,}")
print(f"Reducción del       : {((dimensiones_originales[0] - df.shape[0]) / dimensiones_originales[0]) * 100:.2f}% (Ruido/Duplicados)")
print("="*50)

# Guardamos el archivo listo para la división y el balanceo
archivo_salida = r'C:\Users\Felix\Desktop\Tesis\data\processed\Attack_Dataset_Clean.parquet'
df.to_parquet(archivo_salida, engine='pyarrow')
print(f"\nMatriz depurada exportada con éxito en: '{archivo_salida}'")