import pandas as pd
import numpy as np
import os

#CICIDS2017
base_path = os.path.join(os.path.dirname(__file__), '../../data/processed')
#path_CICIDS2017 = os.path.join(base_path, 'CICIDS2017_unido.parquet')

#df_IDS2017 = pd.read_parquet(path_CICIDS2017, engine='pyarrow')
    
#print(df_IDS2017.head())
#print(df_IDS2017.info())
#print(df_IDS2017['Label'].value_counts())
#print(df_IDS2017.describe())

#DARKNET2020
#path_darknet = r'Darknet.parquet'

#df_DARK = pd.read_parquet(path_darknet, engine='pyarrow')
    
#print(df_DARK.head())
#print(df_DARK.info())
#print(df_DARK['Label'].value_counts())
#print(df_DARK['Label.1'].value_counts())
#print(df_DARK.describe())
#UNSW-NB15
#path_UNSW = r'UNSW-NB15-V3.parquet'

#df_UNSW = pd.read_parquet(path_UNSW, engine='pyarrow')
    
#print(df_UNSW.head())
#print(df_UNSW.info())
#print(df_UNSW['label'].value_counts())
#print(df_UNSW.describe())

#Dataset Unificado
path = os.path.join(base_path, 'Dataset_Final2.parquet')
df_final = pd.read_parquet(path=path, engine='pyarrow')

# Diccionario para mapear los vectores granulares a Macro-Clases
"""map_macro_ataques = {
    'Normal': 'Normal',
    'Generic': 'Generic_Combined',
    'Combined': 'Generic_Combined',
    'DoS': 'DoS',
    'Reconnaissance': 'Reconnaissance',
    'Analysis': 'Reconnaissance',
    'Exploits': 'Malware_Exploits',
    'Fuzzers': 'Malware_Exploits',
    'Backdoor': 'Malware_Exploits',
    'Botnet': 'Malware_Exploits',
    'Shellcode': 'Malware_Exploits',
    'Worms': 'Malware_Exploits',
    'Infiltration': 'Malware_Exploits',
    'Brute_Force': 'Access_Attacks',
    'Web_Attack': 'Access_Attacks'
}

# Aplicar la agrupación a la columna attack_vector
df_final['attack_vector'] = df_final['attack_vector'].replace(map_macro_ataques)

print("\nNueva Distribución de Macro-Ataques:")
print(df_final['attack_vector'].value_counts())

print(df_final.info())
print(df_final.head(20))
print(df_final.describe())
print("\nDistribución de is_encrypted (Fase 1):")
print(df_final['is_encrypted'].value_counts())
print("\nDistribución de attack_vector (Fase 2):")
print(df_final['attack_vector'].value_counts())
#df_final.to_parquet('Dataset_Final1.parquet', engine='pyarrow')
"""
print(df_final.head(10))



