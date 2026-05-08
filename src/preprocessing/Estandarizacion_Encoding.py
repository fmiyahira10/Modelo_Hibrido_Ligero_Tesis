import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

path = os.path.join(os.path.dirname(__file__), '../../data/processed/Dataset_Final2.parquet')
df_final = pd.read_parquet(path=path, engine='pyarrow')

print(f"Dimensiones iniciales: {df_final.shape}")

#Separar las caracteristicas y las etiquetas
X = df_final.drop(columns=['is_encrypted','attack_vector'])
y_cifrado = df_final['is_encrypted']
y_ataque = df_final['attack_vector']

#Codificación de y_ataque

encoder = LabelEncoder()
y_ataque_encoded = encoder.fit_transform(y_ataque)

#diccionario de mapeo
mapeo_clases = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print(f"Diccionario de clases: {mapeo_clases}")

#Division de Train/Test (80/20)
X_train, X_test, y_train_cifrado, y_test_cifrado, y_train_ataque, y_test_ataque = train_test_split(
  X, y_cifrado, y_ataque_encoded, 
  test_size=0.20, 
  random_state=42, 
  stratify=y_cifrado # Priorizamos mantener la escasa proporción del tráfico cifrado en ambos sets
)

print(f"Tamaño X_train: {X_train.shape}")
print(f"Tamaño X_test: {X_test.shape}")

#Estandarización
scaler = StandardScaler()
#Escalar train set
X_train_scaled = scaler.fit_transform(X_train)
#Escalar test set
X_test_scaled = scaler.transform(X_test)

# 5. Guardar los objetos y los sets finales para pasarlos a la Red Neuronal
# Es vital guardar el scaler y el encoder para cuando pruebes el modelo en el futuro con datos nuevos
final_path = os.path.join(os.path.dirname(__file__), '../../data/final')
joblib.dump(scaler, os.path.join(final_path, 'scaler_red.pkl'))
joblib.dump(encoder, os.path.join(final_path, 'encoder_ataques.pkl'))

# Si deseas guardar los arrays ya listos en formato .npy para cargarlos súper rápido después
np.save(os.path.join(final_path, 'X_train_scaled.npy'), X_train_scaled)
np.save(os.path.join(final_path, 'X_test_scaled.npy'), X_test_scaled)
np.save(os.path.join(final_path, 'y_train_cifrado.npy'), y_train_cifrado)
np.save(os.path.join(final_path, 'y_test_cifrado.npy'), y_test_cifrado)
np.save(os.path.join(final_path, 'y_train_ataque.npy'), y_train_ataque)
np.save(os.path.join(final_path, 'y_test_ataque.npy'), y_test_ataque)
