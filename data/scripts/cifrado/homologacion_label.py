import pandas as pd

# ===================================================================
# 1. CARGA DEL DATASET ORIGINAL DE DARKNET
# ===================================================================
path_darknet = r'C:\Users\Felix\Desktop\Tesis\data\processed\Darknet.parquet'  # Reemplaza con tu archivo local (.parquet o .csv)
print("Cargando dataset CIC-Darknet2020 original...")
df_dark = pd.read_parquet(path_darknet, engine='pyarrow')

# Limpieza preventiva por si existen espacios en blanco en los nombres de columnas
df_dark.columns = df_dark.columns.str.strip()

# ===================================================================
# 2. AGRUPACIÓN EXCLUSIVA DE ETIQUETAS (Label)
# ===================================================================
print("Agrupando las etiquetas en dos categorías: Cifrado y No Cifrado...")

# Mapeo directo de tus cuatro clases observadas
mapeo_binario = {
    'Non-Tor': 'No Cifrado',
    'NonVPN': 'No Cifrado',
    'VPN': 'Cifrado',
    'Tor': 'Cifrado'
}

# Creamos la nueva columna objetivo basada únicamente en el agrupamiento
df_dark['Encryption_Label'] = df_dark['Label'].map(mapeo_binario)

# ===================================================================
# 3. REPORTE DE CONTEO FINAL
# ===================================================================
conteo = df_dark['Encryption_Label'].value_counts()
proporciones = df_dark['Encryption_Label'].value_counts(normalize=True) * 100

print("\n" + "="*50)
print("       AGRUPAMIENTO DEL DATASET DE CIFRADO")
print("="*50)
print(f"No Cifrado (Non-Tor / NonVPN) : {conteo['No Cifrado']:>7,} registros ({proporciones['No Cifrado']:.2f}%)")
print(f"Cifrado (VPN / Tor)           : {conteo['Cifrado']:>7,} registros ({proporciones['Cifrado']:.2f}%)")
print("="*50)

# ===================================================================
# 4. GUARDAR DATASET CON NUEVAS ETIQUETAS
# ===================================================================
archivo_salida = r'C:\Users\Felix\Desktop\Tesis\data\processed\Darknet_Etiquetado_Grupos.parquet'
df_dark.to_parquet(archivo_salida, engine='pyarrow')
print(f"\nDataset guardado con todas sus columnas y nuevos grupos en: '{archivo_salida}'")