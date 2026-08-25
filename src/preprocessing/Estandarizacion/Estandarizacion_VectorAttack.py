import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
import joblib
from pathlib import Path
import os

# ===================================================================
# 1. CARGA DE LA MATRIZ DEPURADA
# ===================================================================
# Raíz del proyecto: sube 3 niveles desde src/preprocessing/Estandarizacion/
BASE_DIR = Path(__file__).resolve().parents[3]
archivo_entrada = BASE_DIR / 'data' / 'processed' / 'Attack_Dataset_Clean.parquet'
print(f"Abriendo matriz depurada: {archivo_entrada}")
df = pd.read_parquet(archivo_entrada, engine='pyarrow')

columnas_features = [
    'flow_duration', 'fwd_packets', 'bwd_packets', 'fwd_bytes', 'bwd_bytes',
    'fwd_packet_len_mean', 'bwd_packet_len_mean', 'fwd_iat_mean', 'bwd_iat_mean',
    'fwd_tcp_window', 'bwd_tcp_window', 'flow_packets_per_sec', 'flow_bytes_per_sec'
]
target = 'attack_vector'

# ===================================================================
# 2. CODIFICACIÓN DE ETIQUETAS
# ===================================================================
le_attack = LabelEncoder()
df['attack_vector_encoded'] = le_attack.fit_transform(df[target])
joblib.dump(le_attack, os.path.join(BASE_DIR, 'models', 'scalers_encoders', 'label_encoder_attack.pkl'))

X = df[columnas_features]
y = df['attack_vector_encoded']

# ===================================================================
# 3. DIVISIÓN DE DATOS ESTRATIFICADA (70% / 15% / 15%)
# ===================================================================
X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# ===================================================================
# 4. UNDERSAMPLING ESTRATÉGICO DE LA CLASE NORMAL (Solo en Train)
# ===================================================================
df_train_temp = pd.DataFrame(X_train_raw, columns=columnas_features)
df_train_temp['target'] = y_train_raw.values

codigo_normal = le_attack.transform(['Normal'])[0]
df_train_normal = df_train_temp[df_train_temp['target'] == codigo_normal]
df_train_ataques = df_train_temp[df_train_temp['target'] != codigo_normal]

# Submuestreo balanceado
df_train_normal_sampled = df_train_normal.sample(n=450000, random_state=42)
df_train_final = pd.concat([df_train_normal_sampled, df_train_ataques], ignore_index=True)
df_train_final = df_train_final.sample(frac=1, random_state=42).reset_index(drop=True)

X_train = df_train_final[columnas_features]
y_train = df_train_final['target']

# ===================================================================
# 5. ESCALADO ROBUSTO (Mitigación científica de Outliers)
# ===================================================================
print("\n[Fase 2 - CORREGIDO] Escalando variables continuas mediante RobustScaler...")
# Implementamos RobustScaler para mitigar el impacto del 24.8% de outliers detectados
scaler = RobustScaler(with_centering=True, with_scaling=True)

# Ajustamos exclusivamente con el conjunto de entrenamiento para evitar Data Leakage
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Guardar escalador robusto
joblib.dump(scaler, os.path.join(BASE_DIR, 'models', 'scalers_encoders', 'robust_scaler_attack.pkl'))

# ===================================================================
# 6. SERIALIZACIÓN DE MATRICES FINALES
# ===================================================================
np.save(os.path.join(BASE_DIR, 'data', 'final', 'X_train_attack.npy'), X_train_scaled)
np.save(os.path.join(BASE_DIR, 'data', 'final', 'X_val_attack.npy'), X_val_scaled)
np.save(os.path.join(BASE_DIR, 'data', 'final', 'X_test_attack.npy'), X_test_scaled)
np.save(os.path.join(BASE_DIR, 'data', 'final', 'y_train_attack.npy'), y_train.to_numpy())
np.save(os.path.join(BASE_DIR, 'data', 'final', 'y_val_attack.npy'), y_val.to_numpy())
np.save(os.path.join(BASE_DIR, 'data', 'final', 'y_test_attack.npy'), y_test.to_numpy())

print("\n" + "="*50)
print("     MATRICES EXPORTADAS MEDIANTE ROBUSTSCALER")
print("="*50)
print(f"X_train_attack: {X_train_scaled.shape}")
print(f"X_val_attack  : {X_val_scaled.shape}")
print(f"X_test_attack : {X_test_scaled.shape}")
print("="*50)