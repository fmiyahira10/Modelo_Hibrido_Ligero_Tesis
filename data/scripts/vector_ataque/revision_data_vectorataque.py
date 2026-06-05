import pandas as pd
import numpy as np
from pathlib import Path

# ===================================================================
# CONFIGURACIÓN DE RUTA DE ARCHIVOS
# ===================================================================
# Raíz del proyecto: sube 3 niveles desde data/scripts/vector_ataque/
BASE_DIR = Path(__file__).resolve().parents[3]

path_CICIDS2017 = BASE_DIR / 'data' / 'processed' / 'CICIDS2017.parquet'
path_UNSW       = BASE_DIR / 'data' / 'processed' / 'UNSW-NB15-V3.parquet'

print("Cargando metadatos estructurados de las fuentes reales...")
# Cargamos una muestra mínima (head) para no consumir RAM innecesaria en la fase exploratoria
df_ids = pd.read_parquet(path_CICIDS2017, engine='pyarrow').head(10)
df_unsw = pd.read_parquet(path_UNSW, engine='pyarrow').head(10)

# Limpieza estricta de espacios ocultos en los nombres de las columnas
df_ids.columns = df_ids.columns.str.strip()
df_unsw.columns = df_unsw.columns.str.strip()

# ===================================================================
# DICCIONARIO MAESTRO DE MAPEO CONCEPTO-COLUMNA
# ===================================================================
mapeo_completo = {
    "Concepto Universal": [
        "Duración del Flujo", "Paquetes Fwd", "Paquetes Bwd", 
        "Bytes Fwd", "Bytes Bwd", "Media Tamaño Paquete Fwd", 
        "Media Tamaño Paquete Bwd", "Media Tiempo Interarribo Fwd", "Media Tiempo Interarribo Bwd",
        "Ventana TCP Origen", "Ventana TCP Destino", "Paquetes por Segundo", "Bytes por Segundo"
    ],
    "Columna CICIDS2017": [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Mean",
        "Bwd Packet Length Mean", "Fwd IAT Mean", "Bwd IAT Mean",
        "Init_Win_bytes_forward", "Init_Win_bytes_backward", "Flow Packets/s", "Flow Bytes/s"
    ],
    "Columna UNSW-NB15": [
        "dur", "spkts", "dpkts",
        "sbytes", "dbytes", "smeansz",
        "dmeansz", "sintpkt", "dintpkt",
        "swin", "dwin", "CALCULABLE", "CALCULABLE"
    ]
}

# ===================================================================
# VERIFICADOR OPERACIONAL DE CAMPOS Y TIPOS
# ===================================================================
analisis_features = []

for i in range(len(mapeo_completo["Concepto Universal"])):
    concepto = mapeo_completo["Concepto Universal"][i]
    col_ids = mapeo_completo["Columna CICIDS2017"][i]
    col_unsw = mapeo_completo["Columna UNSW-NB15"][i]
    
    # Comprobar presencia y tipo en CICIDS
    if col_ids in df_ids.columns:
        status_ids = f"OK ({df_ids[col_ids].dtype})"
    else:
        status_ids = "NO ENCONTRADA"
        
    # Comprobar presencia y tipo en UNSW
    if col_unsw == "CALCULABLE":
        status_unsw = "Derivable por Fórmula"
    elif col_unsw in df_unsw.columns:
        status_unsw = f"OK ({df_unsw[col_unsw].dtype})"
    else:
        status_unsw = "NO ENCONTRADA"
        
    analisis_features.append({
        "Concepto Metodológico": concepto,
        "Estado en CICIDS2017": status_ids,
        "Estado en UNSW-NB15": status_unsw
    })

df_reporte = pd.DataFrame(analisis_features)

print("\n==================================================================")
print("             REPORTE TÉCNICO DE COMPATIBILIDAD DE DATASETS")
print("==================================================================")
print(df_reporte.to_string(index=False))
print("\n==================================================================")