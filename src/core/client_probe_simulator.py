import socket
import json
import time
import os
import pandas as pd
import numpy as np
from pathlib import Path

def enviar_flujo_a_nids(id_flujo, features_attack, features_encryption, host='127.0.0.1', port=9999):
    """Establece conexión efímera socket y transmite el JSON."""
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        
        payload = {
            "id_flujo": id_flujo,
            "features_attack": features_attack,
            "features_encryption": features_encryption
        }
        
        client_socket.send(json.dumps(payload).encode('utf-8'))
        response = client_socket.recv(4096).decode('utf-8')
        client_socket.close()
        return json.loads(response)
    except ConnectionRefusedError:
        return {"status": "ERROR", "veredicto": "Servidor NIDS inalcanzable. Verifica que el core esté corriendo."}

if __name__ == "__main__":
    print("[+] Inicializando Simulador de Sonda con Tráfico Real de Tesis...")
    
    # 1. Rutas de los datasets limpios para extraer muestras reales
    BASE_DIR = Path(__file__).resolve().parents[2]
    ruta_ataques = BASE_DIR / 'data' / 'processed' / 'Attack_Dataset_Clean.parquet'
    ruta_cifrado = BASE_DIR /'data' / 'processed' / 'Encryption_Dataset_Clean.parquet'
    
    # Columnas exactas que espera el carril de ataques
    columnas_attack = [
        'flow_duration', 'fwd_packets', 'bwd_packets', 'fwd_bytes', 'bwd_bytes',
        'fwd_packet_len_mean', 'bwd_packet_len_mean', 'fwd_iat_mean', 'bwd_iat_mean',
        'fwd_tcp_window', 'bwd_tcp_window', 'flow_packets_per_sec', 'flow_bytes_per_sec'
    ]
    
    print("[-] Cargando subconjuntos para muestreo de producción...")
    df_at = pd.read_parquet(ruta_ataques, engine='pyarrow')
    df_enc = pd.read_parquet(ruta_cifrado, engine='pyarrow')
    
    # Extraemos las columnas de encriptación dinámicamente descartando el label
    columnas_enc = [col for col in df_enc.columns if col != 'Encryption_Label']
    
    # 2. SELECCIÓN DE FIRMAS DE TRÁFICO REALES
    print("[+] Extrayendo firmas criptográficas y vectoriales reales...")
    
    # Caso 1: Flujo Normal y Texto Plano Real
    row_normal_at = df_at[df_at['attack_vector'] == 'Malware_Exploits'].sample(n=1, random_state=60)[columnas_attack].values[0].tolist()
    row_normal_enc = df_enc[df_enc['Encryption_Label'] == 'No Cifrado'].sample(n=1, random_state=60)[columnas_enc].values[0].tolist()
    
    # Caso 2: Intrusión DoS Real en Texto Plano
    row_dos_at = df_at[df_at['attack_vector'] == 'Normal'].sample(n=1, random_state=10)[columnas_attack].values[0].tolist()
    row_dos_enc = df_enc[df_enc['Encryption_Label'] == 'Cifrado'].sample(n=1, random_state=10)[columnas_enc].values[0].tolist()
    
    # Caso 3: Ataque/Malware Real oculto dentro de un canal Cifrado (VPN/Tor)
    # Extraemos un vector de malware real de tus datos unificados
    row_malware_at = df_at[df_at['attack_vector'] == 'DoS'].sample(n=1, random_state=99)[columnas_attack].values[0].tolist()
    row_malware_enc = df_enc[df_enc['Encryption_Label'] == 'No Cifrado'].sample(n=1, random_state=99)[columnas_enc].values[0].tolist()

    # Estructuramos el set experimental de la demo
    perfiles = [
        ("001", row_normal_at, row_normal_enc, "Tráfico Benigno Estándar (Navegación Web Real)"),
        ("002", row_dos_at, row_dos_enc, "Ataque de Inundación de Denegación de Servicio (DoS Real)"),
        ("003", row_malware_at, row_malware_enc, "Infección de Malware Ofuscada sobre Canal Cifrado Real (VPN/Tor)")
    ]
    
    # Liberar memoria de los dataframes masivos antes de transmitir
    del df_at, df_enc
    
    # 3. TRANSMISIÓN END-TO-END
    for id_f, at_feat, enc_feat, desc in perfiles:
        print(f"\n[Sonda] Capturando datos del flujo corporativo: '{desc}'")
        print(f"[Sonda] Transmitiendo vectores métricos hacia el núcleo IDS...")
        
        # Enviar al servidor socket
        resp = enviar_flujo_a_nids(id_f, at_feat, enc_feat)
        
        print(f"[Sonda] Respuesta Central NIDS: Status={resp['status']} | Veredicto='{resp['veredicto']}'")
        time.sleep(2)
        
    print("\n[+] Demostración de ráfagas operacionales concluida con éxito.")