import pandas as pd
import numpy as np
from pathlib import Path

# ===================================================================
# 1. CONFIGURACIÓN DE RUTA DE INGRESO DE DATOS
# ===================================================================
# Raíz del proyecto: sube 3 niveles desde data/scripts/vector_ataque/
BASE_DIR = Path(__file__).resolve().parents[3]

path_CICIDS2017 = BASE_DIR / 'data' / 'processed' / 'CICIDS2017.parquet'
path_UNSW       = BASE_DIR / 'data' / 'processed' / 'UNSW-NB15-V3.parquet'

print("Cargando datasets para el modelo de clasificación de ataques...")
df_IDS2017 = pd.read_parquet(path_CICIDS2017, engine='pyarrow')
df_UNSW = pd.read_parquet(path_UNSW, engine='pyarrow')

# Lista universal de las 13 características estadísticas de flujo compartidas
columnas_finales = [
    'flow_duration', 'fwd_packets', 'bwd_packets', 'fwd_bytes', 'bwd_bytes',
    'fwd_packet_len_mean', 'bwd_packet_len_mean', 'fwd_iat_mean', 'bwd_iat_mean',
    'fwd_tcp_window', 'bwd_tcp_window', 'flow_packets_per_sec', 'flow_bytes_per_sec'
]

# ===================================================================
# 2. PROCESAMIENTO Y HOMOLOGACIÓN DE CICIDS2017
# ===================================================================
print("\nProcesando y renombrando características en CICIDS2017...")

# Limpieza preventiva de espacios ocultos en los nombres de las columnas
df_IDS2017.rename(columns=lambda x: x.strip(), inplace=True)

map_cicids = {
    'Flow Duration': 'flow_duration', 
    'Total Fwd Packets': 'fwd_packets',
    'Total Backward Packets': 'bwd_packets', 
    'Total Length of Fwd Packets': 'fwd_bytes',
    'Total Length of Bwd Packets': 'bwd_bytes', 
    'Fwd Packet Length Mean': 'fwd_packet_len_mean',
    'Bwd Packet Length Mean': 'bwd_packet_len_mean', 
    'Fwd IAT Mean': 'fwd_iat_mean',
    'Bwd IAT Mean': 'bwd_iat_mean', 
    'Init_Win_bytes_forward': 'fwd_tcp_window',
    'Init_Win_bytes_backward': 'bwd_tcp_window', 
    'Flow Packets/s': 'flow_packets_per_sec',
    'Flow Bytes/s': 'flow_bytes_per_sec'
}

df_IDS2017.rename(columns=map_cicids, inplace=True)

# Mantener la columna cruda de etiquetas para el mapeo taxonómico posterior
df_IDS2017['attack_vector_raw'] = df_IDS2017['Label']

# Filtrado y conversión estricta a float64 para garantizar la compatibilidad matemática
df_IDS2017 = df_IDS2017[columnas_finales + ['attack_vector_raw']].astype({
    'flow_duration': 'float64', 'fwd_packets': 'float64', 'bwd_packets': 'float64',
    'fwd_bytes': 'float64', 'bwd_bytes': 'float64', 'fwd_tcp_window': 'float64',
    'bwd_tcp_window': 'float64'
})

# ===================================================================
# 3. PROCESAMIENTO, CÁLCULO Y HOMOLOGACIÓN DE UNSW-NB15
# ===================================================================
print("Procesando y deduciendo variables temporales en UNSW-NB15...")

df_UNSW.rename(columns=lambda x: x.strip(), inplace=True)

map_unsw = {
    'dur': 'flow_duration', 
    'spkts': 'fwd_packets', 
    'dpkts': 'bwd_packets',
    'sbytes': 'fwd_bytes', 
    'dbytes': 'bwd_bytes', 
    'smeansz': 'fwd_packet_len_mean',
    'dmeansz': 'bwd_packet_len_mean', 
    'sintpkt': 'fwd_iat_mean', 
    'dintpkt': 'bwd_iat_mean',
    'swin': 'fwd_tcp_window', 
    'dwin': 'bwd_tcp_window'
}

df_UNSW.rename(columns=map_unsw, inplace=True)

# Cálculo matemático de tasas por segundo para equiparar con CICIDS2017
df_UNSW['flow_packets_per_sec'] = np.where(
    df_UNSW['flow_duration'] > 0, 
    (df_UNSW['fwd_packets'] + df_UNSW['bwd_packets']) / df_UNSW['flow_duration'], 
    0.0
)

df_UNSW['flow_bytes_per_sec'] = np.where(
    df_UNSW['flow_duration'] > 0, 
    (df_UNSW['fwd_bytes'] + df_UNSW['bwd_bytes']) / df_UNSW['flow_duration'], 
    0.0
)

# Mantener la columna cruda de etiquetas para el mapeo taxonómico posterior
df_UNSW['attack_vector_raw'] = df_UNSW['label']

df_UNSW = df_UNSW[columnas_finales + ['attack_vector_raw']]

# ===================================================================
# 4. CONCATENACIÓN E INTEGRACIÓN VERTICAL DE DATASETS
# ===================================================================
print("\nUnificando matrices de características...")
df_ataques_unificado = pd.concat([df_IDS2017, df_UNSW], ignore_index=True)

# Limpieza inicial de indeterminaciones resultantes del cálculo de tasas
df_ataques_unificado.replace([np.inf, -np.inf], np.nan, inplace=True)
df_ataques_unificado.dropna(subset=columnas_finales, inplace=True)

print(f"\n¡Unificación de características completada con éxito!")
print(f"Dimensiones de la matriz integrada para Ataques: {df_ataques_unificado.shape}")

# Guardado en formato Parquet
df_ataques_unificado.to_parquet(BASE_DIR / 'data' / 'processed' / 'Ataques_Unificado_Raw.parquet', engine='pyarrow')
print("Archivo 'Ataques_Unificado_Raw.parquet' exportado correctamente.")