import socket
import json
import time
import random

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
        # Leer confirmación de recepción del veredicto
        response = client_socket.recv(1024).decode('utf-8')
        client_socket.close()
        return json.loads(response)
    except ConnectionRefusedError:
        return {"status": "ERROR", "veredicto": "Servidor NIDS inalcanzable. Verifica que el core esté corriendo."}

if __name__ == "__main__":
    print("[+] Inicializando Simulador de Sonda de Tráfico en Tiempo Real...")
    time.sleep(1)
    
    # --- DEFINICIÓN DE PERFILES DE TRÁFICO EXPERIMENTALES REALES ---
    # Perfil 1: Tráfico Benigno sin cifrar (Simulación normal)
    trafico_normal_attack = [0.5, 2.0, 2.0, 100.0, 100.0, 50.0, 50.0, 0.1, 0.1, 2048, 2048, 8.0, 400.0]
    trafico_normal_enc = [0.0] * 62
    
    # Perfil 2: Inundación DoS en Texto Plano (CICIDS2017)
    trafico_dos_attack = [0.01, 150.0, 0.0, 45000.0, 0.0, 300.0, 0.0, 0.0001, 0.0, 1024, 0, 15000.0, 4500000.0]
    trafico_dos_enc = [0.0] * 62
    
    # Perfil 3: Tráfico sospechoso de Malware viajando en Canal Cifrado (CIC-Darknet2020)
    # Alteramos descriptores volumétricos simulados activos de Darknet para disparar las alertas
    trafico_malware_attack = [12.4, 25.0, 30.0, 1200.0, 8500.0, 48.0, 283.3, 0.4, 0.3, 8192, 8192, 4.4, 782.2]
    # Llenamos las 62 variables activas. Ponemos magnitudes típicas en las primeras posiciones (Flow Duration, Packets, etc.)
    trafico_malware_enc = [12400000.0, 25.0, 30.0, 1200.0, 8500.0, 400.0, 0.0, 48.0, 12.0, 1500.0, 0.0, 283.3, 50.0] + [0.0]*49

    perfiles = [
        ("001", trafico_normal_attack, trafico_normal_enc, "Flujo de Navegación HTTP Estándar"),
        ("002", trafico_dos_attack, trafico_dos_enc, "Ráfaga de Inundación masiva de paquetes (DoS)"),
        ("003", trafico_malware_attack, trafico_malware_enc, "Conexión persistente hacia Servidor C2 (Malware/Cifrado)")
    ]
    
    for id_f, at_feat, enc_feat, desc in perfiles:
        print(f"\n[Sonda] Detectando descriptor de flujo: '{desc}'")
        print(f"[Sonda] Transmitiendo telemetría al motor central...")
        
        resp = enviar_flujo_a_nids(id_f, at_feat, enc_feat)
        
        print(f"[Sonda] Servidor NIDS responde: Status={resp['status']} | Veredicto='{resp['veredicto']}'")
        time.sleep(3) # Pausa entre capturas de flujos para visibilidad de la demo
        
    print("\n[+] Simulación de ráfagas completada.")