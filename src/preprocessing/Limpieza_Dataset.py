import pandas as pd
import os
#Undersampling
path = os.path.join(os.path.dirname(__file__), '../../data/processed/Dataset_Final1.parquet')
df_final = pd.read_parquet(path=path, engine='pyarrow')

# 1. Separar los datos en TRES grupos (no en dos)
# Grupo A: Los ataques (los guardamos todos intactos)
df_ataques = df_final[df_final['attack_vector'] != 'Normal']

# Grupo B: El tráfico Normal que SÍ está cifrado (Darknet VPN/Tor - lo guardamos todo intacto)
df_normal_cifrado = df_final[(df_final['attack_vector'] == 'Normal') & (df_final['is_encrypted'] == 1)]

# Grupo C: El tráfico Normal en texto plano (El gigante de 4.6 millones que queremos reducir)
df_normal_plano = df_final[(df_final['attack_vector'] == 'Normal') & (df_final['is_encrypted'] == 0)]

print(f"Tráfico Normal Plano (a reducir): {len(df_normal_plano)}")
print(f"Tráfico Normal Cifrado (intacto): {len(df_normal_cifrado)}")
print(f"Tráfico de Ataques (intacto): {len(df_ataques)}")

# 2. Aplicar Undersampling SOLO al texto plano
# Puedes ajustar este número. Como ya tienes ~24k cifrados y los ataques, 400k es un buen tamaño
n_muestras = 400000 
df_normal_plano_reducido = df_normal_plano.sample(n=n_muestras, random_state=42)

# 3. Reensamblar el rompecabezas
print("\nUniendo el dataset optimizado...")
df_balanceado = pd.concat([df_normal_plano_reducido, df_normal_cifrado, df_ataques], ignore_index=True)

# 4. Mezclar las filas para que el modelo no aprenda el orden
df_balanceado = df_balanceado.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n¡Listo! Tamaño del dataset final: {df_balanceado.shape}")
print("\nDistribución Final de is_encrypted (Fase 1):")
print(df_balanceado['is_encrypted'].value_counts())

print("\nDistribución Final de attack_vector (Fase 2):")
print(df_balanceado['attack_vector'].value_counts())

output_path = os.path.join(os.path.dirname(__file__), '../../data/processed/Dataset_Final2.parquet')
df_balanceado.to_parquet(output_path, engine='pyarrow')