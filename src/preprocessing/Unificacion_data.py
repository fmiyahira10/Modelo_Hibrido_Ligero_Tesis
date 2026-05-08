import pandas as pd
import numpy as np
import os

base_path = os.path.join(os.path.dirname(__file__), '../../data/processed')

path_CICIDS2017 = os.path.join(base_path, 'CICIDS2017_unido.parquet')
df_IDS2017 = pd.read_parquet(path_CICIDS2017, engine='pyarrow')

path_darknet = os.path.join(os.path.dirname(__file__), '../../data/raw/Darknet.csv')
df_DARK = pd.read_csv(path_darknet)

path_UNSW = os.path.join(os.path.dirname(__file__), '../../data/raw/UNSW-NB15-V3.csv')
df_UNSW = pd.read_csv(path_UNSW)

# 1. Diccionarios de Homologación de Características (Ajustado a 13 - Sin Protocolo)

map_cicids = {
    'Flow Duration': 'flow_duration',
    'Total Fwd Packets': 'fwd_packets',
    'Total Backward Packets': 'bwd_packets',
    'Total Length of Fwd Packets': 'fwd_bytes',
    'Total Length of Bwd Packets': 'bwd_bytes',
    'Fwd Packet Length Mean': 'fwd_packet_len_mean',
    'Bwd Packet Length Mean': 'bwd_packet_len_mean',
    'Fwd IAT Mean': 'fwd_iat_mean',
    'Bwd IAT Mean': 'bwd_iat_mean',
    'Init_Win_bytes_forward': 'fwd_tcp_window',
    'Init_Win_bytes_backward': 'bwd_tcp_window',
    'Flow Packets/s': 'flow_packets_per_sec',
    'Flow Bytes/s': 'flow_bytes_per_sec'
}

map_darknet = {
    'Flow Duration': 'flow_duration',
    'Total Fwd Packet': 'fwd_packets',
    'Total Bwd packets': 'bwd_packets',
    'Total Length of Fwd Packet': 'fwd_bytes',
    'Total Length of Bwd Packet': 'bwd_bytes',
    'Fwd Packet Length Mean': 'fwd_packet_len_mean',
    'Bwd Packet Length Mean': 'bwd_packet_len_mean',
    'Fwd IAT Mean': 'fwd_iat_mean',
    'Bwd IAT Mean': 'bwd_iat_mean',
    'FWD Init Win Bytes': 'fwd_tcp_window', 
    'Bwd Init Win Bytes': 'bwd_tcp_window', 
    'Flow Packets/s': 'flow_packets_per_sec',
    'Flow Bytes/s': 'flow_bytes_per_sec'
}

map_unsw = {
    'dur': 'flow_duration',
    'spkts': 'fwd_packets',
    'dpkts': 'bwd_packets',
    'sbytes': 'fwd_bytes',
    'dbytes': 'bwd_bytes',
    'smeansz': 'fwd_packet_len_mean',
    'dmeansz': 'bwd_packet_len_mean',
    'sintpkt': 'fwd_iat_mean',
    'dintpkt': 'bwd_iat_mean',
    'swin': 'fwd_tcp_window',
    'dwin': 'bwd_tcp_window'
}

# La lista maestra de columnas (13 características + 2 labels)
columnas_finales = [
    'flow_duration', 'fwd_packets', 'bwd_packets', 'fwd_bytes', 'bwd_bytes',
    'fwd_packet_len_mean', 'bwd_packet_len_mean', 'fwd_iat_mean', 'bwd_iat_mean',
    'fwd_tcp_window', 'bwd_tcp_window', 'flow_packets_per_sec', 'flow_bytes_per_sec'
]

# 2. Procesamiento de CICIDS2017
df_IDS2017.rename(columns=lambda x: x.strip(), inplace=True) # Quitar espacios
df_IDS2017.rename(columns=map_cicids, inplace=True)

df_IDS2017['is_encrypted'] = 0
df_IDS2017['attack_vector'] = df_IDS2017['Label'].replace({
    'BENIGN': 'Normal',
    'DoS Hulk': 'DoS', 'DDoS': 'DoS', 'DoS GoldenEye': 'DoS', 'DoS slowloris': 'DoS', 'DoS Slowhttptest': 'DoS',
    'PortScan': 'Reconnaissance',
    'FTP-Patator': 'Brute_Force', 'SSH-Patator': 'Brute_Force',
    'Web Attack - Brute Force': 'Web_Attack', 'Web Attack - XSS': 'Web_Attack', 'Web Attack - Sql Injection': 'Web_Attack',
    'Bot': 'Botnet',
    'Infiltration': 'Infiltration', 'Heartbleed': 'Exploits'
})
df_IDS2017 = df_IDS2017[columnas_finales + ['is_encrypted', 'attack_vector']]


# 3. Procesamiento de UNSW-NB15
df_UNSW.rename(columns=lambda x: x.strip(), inplace=True)
df_UNSW.rename(columns=map_unsw, inplace=True)

# Cálculo de las dos columnas faltantes en UNSW (evitando divisiones por cero)
df_UNSW['flow_packets_per_sec'] = np.where(df_UNSW['flow_duration'] > 0, 
                                           (df_UNSW['fwd_packets'] + df_UNSW['bwd_packets']) / df_UNSW['flow_duration'], 0)
df_UNSW['flow_bytes_per_sec'] = np.where(df_UNSW['flow_duration'] > 0, 
                                         (df_UNSW['fwd_bytes'] + df_UNSW['bwd_bytes']) / df_UNSW['flow_duration'], 0)

df_UNSW['is_encrypted'] = 0
df_UNSW['attack_vector'] = df_UNSW['label'].replace({
    'benign': 'Normal',
    'DoS': 'DoS',
    'Reconnaissance': 'Reconnaissance',
    'Exploits': 'Exploits',
    'Fuzzers': 'Fuzzers',
    'Generic': 'Generic',
    'Comb': 'Combined',
    'Analysis': 'Analysis',
    'Backdoor': 'Backdoor',
    'Shellcode': 'Shellcode',
    'Worms': 'Worms'
})
df_UNSW = df_UNSW[columnas_finales + ['is_encrypted', 'attack_vector']]


# 4. Procesamiento de Darknet2020
df_DARK.rename(columns=lambda x: x.strip(), inplace=True)
df_DARK.rename(columns=map_darknet, inplace=True)

# Lógica para cifrado: 1 si es VPN o Tor, 0 si no lo es
df_DARK['is_encrypted'] = df_DARK['Label'].apply(lambda x: 1 if x in ['VPN', 'Tor'] else 0)
# Como todo es tráfico de navegación/aplicaciones, el vector de ataque es Normal
df_DARK['attack_vector'] = 'Normal' 

df_DARK = df_DARK[columnas_finales + ['is_encrypted', 'attack_vector']]


# 5. Unificación Final
print("Concatenando los tres datasets...")
df_final = pd.concat([df_IDS2017, df_UNSW, df_DARK], ignore_index=True)

# Limpieza fundamental para el modelo de clasificación (remover infinitos y nulos)
df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
df_final.dropna(inplace=True)

print(f"\nDimensiones del Dataset Unificado: {df_final.shape}")
print("\nDistribución de is_encrypted (Fase 1):")
print(df_final['is_encrypted'].value_counts())
print("\nDistribución de attack_vector (Fase 2):")
print(df_final['attack_vector'].value_counts())

# Guardar
df_final.to_parquet('Dataset_Final.parquet', engine='pyarrow')