import socket
import json
import os
import sys
from pathlib import Path

# Asegurar que Python reconozca el directorio raíz para importar el motor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference_engine import InferenceAndCorrelationEngine

def iniciar_servidor_nids(base_dir, host='127.0.0.1', port=9999):
    # Instanciamos el motor de correlación cargando los modelos en memoria una sola vez
    engine = InferenceAndCorrelationEngine(base_dir=base_dir)
    
    # Configuración del socket de red
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    
    print(f"\n[+] Servidor NIDS Activo escuchando en {host}:{port}...")
    print("[-] Esperando ráfagas de tráfico desde la sonda empresarial simulada...\n")
    
    try:
        while True:
            client_conn, addr = server_socket.accept()
            # Recibir el flujo serializado en formato JSON
            data = client_conn.recv(4096).decode('utf-8')
            
            if data:
                try:
                    payload = json.loads(data)
                    features_attack = payload['features_attack']
                    features_encryption = payload['features_encryption']
                    id_flujo = payload.get('id_flujo', 'N/A')
                    
                    # Ejecutar el pipeline híbrido modular
                    resultado = engine.process_flow(features_attack, features_encryption)
                    
                    print(f"[FLUJO {id_flujo}] Inferencia Concluida con éxito:")
                    print(f" ├─ Origen: {addr[0]}:{addr[1]}")
                    print(f" ├─ Análisis Vectorial : {resultado['Carril_Aataques']}")
                    print(f" ├─ Análisis Cifrado   : {resultado['Carril_Bcifrado']}")
                    print(f" └─ VEREDICTO GENERAL  : {resultado['Veredicto_IDS']}")
                    print("-" * 50)
                    
                    # Responder a la sonda con el estado del veredicto
                    respuesta = {"status": "SUCCESS", "veredicto": resultado['Veredicto_IDS']}
                    client_conn.send(json.dumps(respuesta).encode('utf-8'))
                    
                except json.JSONDecodeError:
                    print("[-] Error: Payload corrupto o no estructurado en formato JSON.")
                except KeyError as ke:
                    print(f"[-] Error: Faltan llaves estructurales en el JSON recibido: {ke}")
                    
            client_conn.close()
    except KeyboardInterrupt:
        print("\n[+] Apagando Servidor Sonda Core por solicitud del administrador.")
    finally:
        server_socket.close()

if __name__ == "__main__":
    RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
    iniciar_servidor_nids(base_dir=RAIZ_PROYECTO)