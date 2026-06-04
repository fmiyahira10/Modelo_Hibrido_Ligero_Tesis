import pandas as pd
import numpy as np

# ===================================================================
# 1. CARGA DEL DATASET UNIFICADO
# ===================================================================
archivo_entrada = r'C:\Users\Felix\Desktop\Tesis\data\processed\Ataques_Unificado_Raw.parquet'
print(f"Cargando dataset unificado desde {archivo_entrada}...")
df = pd.read_parquet(archivo_entrada, engine='pyarrow')

# ===================================================================
# 2. DEFINICIÓN DEL MAPEO TAXONÓMICO (Hacia tus 6 Macro-Clases)
# ===================================================================
# Diccionario estricto para agrupar las categorías según tu metodología
mapeo_ataques = {
    # Clase: Normal
    'BENIGN': 'Normal', 
    'benign': 'Normal',
    
    # Clase: DoS
    'DoS Hulk': 'DoS', 
    'DDoS': 'DoS', 
    'DoS': 'DoS', 
    'DoS GoldenEye': 'DoS', 
    'DoS slowloris': 'DoS', 
    'DoS Slowhttptest': 'DoS',
    
    # Clase: Reconnaissance
    'PortScan': 'Reconnaissance', 
    'Reconnaissance': 'Reconnaissance', 
    'Analysis': 'Reconnaissance',
    
    # Clase: Generic
    'Generic': 'Generic', 
    'Comb': 'Generic',
    
    # Clase: Malware_Exploits
    'Exploits': 'Malware_Exploits', 
    'Fuzzers': 'Malware_Exploits', 
    'Bot': 'Malware_Exploits', 
    'Infiltration': 'Malware_Exploits', 
    'Heartbleed': 'Malware_Exploits', 
    'Shellcode': 'Malware_Exploits', 
    'Worms': 'Malware_Exploits',
    
    # Clase: Access_Attacks
    'FTP-Patator': 'Access_Attacks', 
    'SSH-Patator': 'Access_Attacks', 
    'Web Attack - Brute Force': 'Access_Attacks', 
    'Web Attack - XSS': 'Access_Attacks', 
    'Web Attack - Sql Injection': 'Access_Attacks'
}

# ===================================================================
# 3. APLICACIÓN DEL MAPEO Y CONTEO ESTADÍSTICO
# ===================================================================
print("\nAplicando homologación de etiquetas...")
df['attack_vector'] = df['attack_vector_raw'].map(mapeo_ataques)

# Verificación de posibles etiquetas que se hayan quedado fuera del mapeo
if df['attack_vector'].isna().any():
    valores_nulos = df[df['attack_vector'].isna()]['attack_vector_raw'].unique()
    print(f"[ALERTA] Quedaron etiquetas sin mapear: {valores_nulos}")
    # Por seguridad, llenamos cualquier residuo como Normal
    df['attack_vector'].fillna('Normal', inplace=True)

# Eliminar la columna intermedia para optimizar memoria RAM
df.drop(columns=['attack_vector_raw'], inplace=True)

# ===================================================================
# 4. REPORTE DE CONTEO Y ANÁLISIS DE DESBALANCE
# ===================================================================
conteo_final = df['attack_vector'].value_counts()
proporciones = df['attack_vector'].value_counts(normalize=True) * 100

print("\n" + "="*50)
print("     DISTRIBUCIÓN DE LAS 6 MACRO-CLASES DE ATAQUE")
print("="*50)
for clase in conteo_final.index:
    print(f"- {clase:<18}: {conteo_final[clase]:>9,} registros ({proporciones[clase]:.4f}%)")
print("="*50)

# ===================================================================
# 5. GUARDAR DATASET HOMOLOGADO
# ===================================================================
archivo_salida = r'C:\Users\Felix\Desktop\Tesis\data\processed\Attack_Dataset_Homologado.parquet'
df.to_parquet(archivo_salida, engine='pyarrow')
print(f"\nDataset guardado exitosamente en: '{archivo_salida}'")