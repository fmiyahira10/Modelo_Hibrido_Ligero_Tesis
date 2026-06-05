import pandas as pd
import numpy as np
import os

# ===================================================================
# 1. CONFIGURACIÓN DE RUTAS RELATIVAS AUTOMÁTICAS
# ===================================================================
# Detecta la ubicación actual del script (asumiendo que corre desde src/ o la raíz)
BASE_DIR = r'C:\Users\i21327\Desktop\Tesis\Modelo_Hibrido_Ligero_Tesis'

# Construcción de rutas relativas basadas en la estructura de tu proyecto
ruta_entrada = os.path.join(BASE_DIR, 'data', 'processed', 'Darknet_Etiquetado_Grupos.parquet')

print(f"Buscando dataset en ruta relativa: {ruta_entrada}")
if not os.path.exists(ruta_entrada):
    raise FileNotFoundError(f"No se encontró el archivo en {ruta_entrada}. Verifica la ejecución desde la raíz del proyecto.")

df = pd.read_parquet(ruta_entrada, engine='pyarrow')

# Limpieza preventiva de nombres de columnas
df.columns = df.columns.str.strip()

# ===================================================================
# 2. SELECCIÓN DE CARACTERÍSTICAS NUMÉRICAS CRUDAS DE FLUJO
# ===================================================================
# Identificamos columnas a omitir (identificadores y targets)
columnas_omitir = ['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Timestamp', 'Label', 'Label.1', 'Encryption_Label']

# Seleccionamos todas las columnas numéricas que queden excluyendo las de omisión
columnas_crudas = [col for col in df.select_dtypes(include=[np.number]).columns if col not in columnas_omitir]

print(f"\nSe han seleccionado {len(columnas_crudas)} características crudas de red para la auditoría.")

# ===================================================================
# 3. AUDITORÍA DE CALIDAD DE DATOS (Nulos, Ceros e Infinitos)
# ===================================================================
print("\n[1/2] Analizando consistencia matemática y vacíos...")

reporte_calidad = []
total_registros = len(df)

for col in columnas_crudas:
    nulos = df[col].isna().sum()
    infinitos = np.isinf(df[col]).sum()
    ceros = (df[col] == 0).sum()
    
    reporte_calidad.append({
        'Característica': col,
        'Nulos (NaN)': nulos,
        'Infinitos (Inf)': infinitos,
        'Valores Cero (0)': ceros,
        '% Ceros': round((ceros / total_registros) * 100, 2)
    })

df_calidad = pd.DataFrame(reporte_calidad)
print("\n=== REPORTE DE INTEGRIDAD DE VARIABLES (DARKNET) ===")
print(df_calidad.to_string(index=False))

# ===================================================================
# 4. AUDITORÍA DE OUTLIERS (Método IQR)
# ===================================================================
print("\n[2/2] Calculando volumen de valores atípicos mediante rango intercuartílico...")

reporte_outliers = []

for col in columnas_crudas:
    # Tratamiento preventivo temporal de infinitos para no romper los percentiles
    col_limpia = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(col_limpia) == 0:
        continue
        
    Q1 = col_limpia.quantile(0.25)
    Q3 = col_limpia.quantile(0.75)
    IQR = Q3 - Q1
    
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    num_outliers = ((col_limpia < limite_inferior) | (col_limpia > limite_superior)).sum()
    porcentaje_outliers = (num_outliers / total_registros) * 100
    
    reporte_outliers.append({
        'Característica': col,
        'Límite Inferior': round(limite_inferior, 2),
        'Límite Superior': round(limite_superior, 2),
        'Cant. Outliers': num_outliers,
        'Porcentaje (%)': round(porcentaje_outliers, 2)
    })

df_outliers_reporte = pd.DataFrame(reporte_outliers)
print("\n=== REPORTE CUANTITATIVO DE OUTLIERS (DARKNET) ===")
print(df_outliers_reporte.to_string(index=False))