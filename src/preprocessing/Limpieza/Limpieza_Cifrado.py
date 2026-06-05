import pandas as pd
import numpy as np
import os
from pathlib import Path

# ===================================================================
# 1. CONFIGURACIÓN DE RUTA RELATIVA EXACTA
# ===================================================================
# Raíz del proyecto: sube 3 niveles desde src/preprocessing/Limpieza/
BASE_DIR = Path(__file__).resolve().parents[3]
ruta_entrada = os.path.join(BASE_DIR, 'data', 'processed', 'Darknet_Etiquetado_Grupos.parquet')

print(f"Iniciando pipeline de Limpieza Fina sobre: {ruta_entrada}")
df = pd.read_parquet(ruta_entrada, engine='pyarrow')

# Limpieza preventiva de nombres de columnas
df.columns = df.columns.str.strip()

# Identificamos el universo inicial de características de red
columnas_omitir = ['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Timestamp', 'Label', 'Label.1', 'Encryption_Label']
columnas_numericas = [col for col in df.select_dtypes(include=[np.number]).columns if col not in columnas_omitir]

print(f"Muestras iniciales crudas: {df.shape[0]:,} con {len(columnas_numericas)} características numéricas.")

# ===================================================================
# 2. FILTRADO REGLAMENTARIO DE VARIANZA CERO (Sustento de Ligereza)
# ===================================================================
print("\n[PASO A] Evaluando y eliminando variables con varianza cero (100% Constantes)...")

# Nos quedamos únicamente con las columnas donde al menos un valor sea diferente de cero
columnas_activas = [col for col in columnas_numericas if (df[col] != 0).any()]
columnas_eliminadas = [col for col in columnas_numericas if col not in columnas_activas]

print(f"-> Se identificaron y eliminaron {len(columnas_eliminadas)} características muertas.")
print(f"   Columnas eliminadas por el filtro: {columnas_eliminadas}")

# ===================================================================
# 3. TRATAMIENTO DE ANOMALÍAS DE CAPTURA (Valores Negativos y Overflow)
# ===================================================================
print("\n[PASO B] Purgando ruido técnico, infinitos y valores negativos imposibles...")

# Reemplazar infinitos por nulos de forma estricta
df[columnas_activas] = df[columnas_activas].replace([np.inf, -np.inf], np.nan)

# Filtrar valores negativos en campos críticos detectados en tu boxplot anterior
# Duraciones o tasas de transferencia no pueden ser menores a cero físicamente
for col in ['Flow Duration', 'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packets/s', 'Bwd Packets/s']:
    if col in columnas_activas:
        df = df[(df[col] >= 0) | (df[col].isna())]

# Filtro especial contra el desborde (Overflow de CICFlowMeter del orden de 10^15 que vimos en tu EDA)
# El límite lógico superior para temporizadores en ráfagas estándar es 1.2e8 microsegundos (120 segundos)
for col in ['Idle Mean', 'Idle Max', 'Idle Min', 'Flow IAT Max', 'Fwd IAT Max']:
    if col in columnas_activas:
        df = df[(df[col] <= 1.2e8) | (df[col].isna())]

# Dropna definitivo sobre las variables activas resultantes y el target
df.dropna(subset=columnas_activas + ['Encryption_Label'], inplace=True)
print(f"-> Muestras remanentes tras control de anomalías: {df.shape[0]:,}")

# ===================================================================
# 4. REMOCIÓN DE DUPLICADOS (Prevención estricta de Data Leakage)
# ===================================================================
print("\n[PASO C] Eliminando flujos idénticos repetidos en la matriz activa...")
df.drop_duplicates(subset=columnas_activas, keep='first', inplace=True)
print(f"-> Muestras finales limpias: {df.shape[0]:,}")

# ===================================================================
# 5. EXPORTACIÓN DE LA MATRIZ DE ENCRIPTACIÓN DEPURADA
# ===================================================================
ruta_salida = os.path.join(BASE_DIR, 'data', 'processed', 'Encryption_Dataset_Clean.parquet')

# Conservamos únicamente las variables útiles activas y nuestra etiqueta unificada
df_final = df[columnas_activas + ['Encryption_Label']]
df_final.to_parquet(ruta_salida, engine='pyarrow')

print("\n" + "="*60)
print("       RESUMEN DE PURIFICACIÓN - MODELO DE CIFRADO")
print("=============================================================")
print(f"Características iniciales : {len(columnas_numericas)}")
print(f"Características finales   : {len(columnas_activas)}")
print(f"Registros finales limpios : {df_final.shape[0]:,}")
print("=============================================================")
print(f"Matriz óptima guardada con éxito en: '{ruta_salida}'")