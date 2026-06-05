import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
import joblib

# ===================================================================
# 1. CONFIGURACIÓN DE RUTA RELATIVAS AUTOMÁTICAS
# ===================================================================
BASE_DIR = r'C:\Users\i21327\Desktop\Tesis\Modelo_Hibrido_Ligero_Tesis'
ruta_entrada = os.path.join(BASE_DIR, 'data', 'processed', 'Encryption_Dataset_Clean.parquet')

print(f"Abriendo matriz depurada de cifrado: {ruta_entrada}")
df = pd.read_parquet(ruta_entrada, engine='pyarrow')

# Extracción dinámica de las 62 características activas finales
columnas_features = [col for col in df.columns if col != 'Encryption_Label']
target = 'Encryption_Label'

# ===================================================================
# 2. CODIFICACIÓN BINARIA DEL TARGET (0 = No Cifrado, 1 = Cifrado)
# ===================================================================
print("\n[Fase 2] Mapeando etiquetas criptográficas a formato binario (0/1)...")
le_encryption = LabelEncoder()
df['target_encoded'] = le_encryption.fit_transform(df[target])

# Serialización del codificador para la etapa final de inferencia paralela
joblib.dump(le_encryption, os.path.join(BASE_DIR, 'label_encoder_encryption.pkl'))

print("Mapeo binario de control establecido:")
for clase, codigo in zip(le_encryption.classes_, le_encryption.transform(le_encryption.classes_)):
    print(f" - {clase:<15} -> Código Asignado: {codigo}")

X = df[columnas_features]
y = df['target_encoded']

# ===================================================================
# 3. DIVISIÓN DE DATOS ESTRATIFICADA (Fase 3: 70% / 15% / 15%)
# ===================================================================
print("\n[Fase 3] Segmentando subconjuntos de datos sin Data Leakage...")

# Separación inicial: 70% Entrenamiento y 30% Temporal (Validación + Prueba)
# El parámetro 'stratify=y' es obligatorio para asegurar que la proporción de 
# tráfico VPN/Tor se mantenga idéntica en los tres entornos.
X_train_raw, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# División equitativa del remanente temporal: 15% Validación y 15% Prueba
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# ===================================================================
# 4. ESCALADO ROBUSTO (Mitigación del 23.8% de Outliers Temporales)
# ===================================================================
print("\n[Fase 2] Normalizando descriptores crudos mediante RobustScaler...")
scaler_robust = RobustScaler(with_centering=True, with_scaling=True)

# REGLA DE ORO CIENTÍFICA: El fit se hace EXCLUSIVAMENTE con el set de entrenamiento
X_train_scaled = scaler_robust.fit_transform(X_train_raw)

# Transformación pasiva de Validación y Prueba utilizando los centroides de entrenamiento
X_val_scaled = scaler_robust.transform(X_val_raw)
X_test_scaled = scaler_robust.transform(X_test_raw)

# Serialización del escalador robusto de encriptación
joblib.dump(scaler_robust, os.path.join(BASE_DIR, 'robust_scaler_encryption.pkl'))

# ===================================================================
# 5. SERIALIZACIÓN MATRICIAL NUMPY (.npy) PARA TENSORFLOW
# ===================================================================
print("\nExportando matrices de tensores a disco duro...")
np.save(os.path.join(BASE_DIR, 'X_train_encryption.npy'), X_train_scaled)
np.save(os.path.join(BASE_DIR, 'X_val_encryption.npy'), X_val_scaled)
# El conjunto de prueba se almacena intacto para la Fase 7 (Evaluación Experimental)
np.save(os.path.join(BASE_DIR, 'X_test_encryption.npy'), X_test_scaled)

np.save(os.path.join(BASE_DIR, 'y_train_encryption.npy'), y_train.to_numpy())
np.save(os.path.join(BASE_DIR, 'y_val_encryption.npy'), y_val.to_numpy())
np.save(os.path.join(BASE_DIR, 'y_test_encryption.npy'), y_test.to_numpy())

print("\n" + "="*60)
print("     INFRAESTRUCTURA DE DATOS DE CIFRADO CONCLUIDA")
print("=============================================================")
print(f"Tensor de Entrenamiento (X_train_encryption) : {X_train_scaled.shape}")
print(f"Tensor de Validación    (X_val_encryption)   : {X_val_scaled.shape}")
print(f"Tensor de Prueba Final  (X_test_encryption)  : {X_test_scaled.shape}")
print("=============================================================")
print("Fases 1, 2 y 3 cerradas con éxito para ambos modelos de la tesis.")