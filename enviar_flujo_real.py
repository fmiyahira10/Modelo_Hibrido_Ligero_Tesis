import csv
import json
import math
import socket
import sys
import time

from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src/core"))

from flow_protocol import decode_message, MAX_MESSAGE_BYTES

RUTA_VECTORES = Path.home() / "tesis/flujos_csv/vectores_prueba.jsonl"
RUTA_CSV = Path.home() / "tesis/flujos_csv/flujos_lab.csv"


def convertir_fecha(valor):
    valor = str(valor).strip()

    try:
        fecha = datetime.fromisoformat(
            valor.replace("Z", "+00:00")
        )
    except ValueError:
        fecha = None

    if fecha is None:
        formatos = [
            "%d/%m/%Y %I:%M:%S %p",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M:%S.%f",
            "%m/%d/%Y %I:%M:%S %p",
            "%m/%d/%Y %H:%M:%S",
            "%Y-%m-%d %I:%M:%S %p",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
        ]
        for formato in formatos:
            try:
                fecha = datetime.strptime(valor, formato)
                break
            except ValueError:
                continue

    if fecha is None:
        raise ValueError(f"Formato de Timestamp desconocido: {valor}")

    if fecha.tzinfo is None:
        # CICFlowMeter se ejecutó en esta misma VM.
        # Utilizamos la zona horaria local de la VM.
        zona_local = datetime.now().astimezone().tzinfo
        fecha = fecha.replace(tzinfo=zona_local)

    return fecha


# 1. Leer primer flujo validado
with RUTA_VECTORES.open(encoding="utf-8") as archivo:
    flujo = json.loads(next(archivo))

vector_b = flujo["features_encryption"]

# 2. Encontrar su fila original en CICFlowMeter
# Se contrastan las IP y ocho características crudas.
columnas = [
    ("Protocol", 0),
    ("Flow Duration", 1),
    ("Total Fwd Packet", 2),
    ("Total Bwd packets", 3),
    ("Total Length of Fwd Packet", 4),
    ("Total Length of Bwd Packet", 5),
    ("Flow Bytes/s", 14),
    ("Flow Packets/s", 15),
]

coincidencias = []

with RUTA_CSV.open(encoding="utf-8-sig", newline="") as archivo:
    lector = csv.DictReader(archivo)

    for fila in lector:
        if fila["Src IP"].strip() != flujo["src_ip"]:
            continue

        if fila["Dst IP"].strip() != flujo["dst_ip"]:
            continue

        if all(
            math.isclose(
                float(fila[nombre]),
                float(vector_b[indice]),
                rel_tol=1e-7,
                abs_tol=1e-6
            )
            for nombre, indice in columnas
        ):
            coincidencias.append(fila)

if len(coincidencias) != 1:
    raise RuntimeError(
        "No se pudo identificar un flujo único en el CSV. "
        f"Coincidencias: {len(coincidencias)}. "
        "No se enviarán metadatos ambiguos."
    )

fila = coincidencias[0]

# 3. Reconstruir metadatos reales de la captura
inicio_flujo = convertir_fecha(fila["Timestamp"])

# CICFlowMeter expresa Flow Duration en microsegundos.
duracion_us = float(fila["Flow Duration"])

if not math.isfinite(duracion_us) or duracion_us < 0:
    raise ValueError("Duración del flujo inválida")

fin_flujo = inicio_flujo + timedelta(
    microseconds=duracion_us
)

# 4. Construir mensaje conforme al contrato
payload = {
    "schema_version": 1,
    "run_id": "lab-prueba-mirror-01",
    "flow_id": flujo["id_flujo"],
    "source_kind": "captured_flow",
    "extractor_id": "CICFlowMeter-PCAP-local",
    "flow": {
        "src_ip": fila["Src IP"].strip(),
        "dst_ip": fila["Dst IP"].strip(),
        "src_port": int(fila["Src Port"]),
        "dst_port": int(fila["Dst Port"]),
        "protocol": int(fila["Protocol"]),
        "started_at": inicio_flujo.isoformat(),
        "ended_at": fin_flujo.isoformat()
    },
    "features_attack": flujo["features_attack"],
    "features_encryption": flujo["features_encryption"]
}

mensaje = (
    json.dumps(payload, allow_nan=False) + "\n"
).encode("utf-8")

# 5. Validación local con el mismo protocolo del servidor
decode_message(mensaje)

print("=" * 55)
print("VALIDACION LOCAL DEL PROTOCOLO")
print("=" * 55)
print("[OK] Contrato validado con flow_protocol.py")
print("Run ID:", payload["run_id"])
print("Flow ID:", payload["flow_id"])
print("Source:", payload["source_kind"])
print("Extractor:", payload["extractor_id"])
print("Metadatos:", json.dumps(payload["flow"], indent=2))
print("Vector A:", len(payload["features_attack"]))
print("Vector B:", len(payload["features_encryption"]))
print("Bytes:", len(mensaje))

if len(mensaje) > MAX_MESSAGE_BYTES:
    raise RuntimeError("Mensaje mayor que el máximo permitido")

# 6. Enviar al servidor NIDS
print("\nEnviando flujo al servidor...")

inicio = time.perf_counter()

with socket.create_connection(
    ("127.0.0.1", 9999),
    timeout=10
) as cliente:

    cliente.settimeout(60)
    cliente.sendall(mensaje)

    with cliente.makefile("rb") as receptor:
        respuesta = receptor.readline(MAX_MESSAGE_BYTES + 1)

tiempo_ms = (time.perf_counter() - inicio) * 1000

if not respuesta:
    raise RuntimeError("El servidor no devolvió respuesta")

resultado = json.loads(respuesta.decode("utf-8"))

print("\n" + "=" * 55)
print("RESULTADO DEL NIDS")
print("=" * 55)
print(json.dumps(resultado, indent=2, ensure_ascii=False))
print(f"Tiempo total cliente-servidor: {tiempo_ms:.2f} ms")

if resultado.get("status") == "SUCCESS":
    print("\n[OK] COMUNICACION E INFERENCIA EXITOSAS")
else:
    print("\n[ERROR] Revisar respuesta del servidor")

print("=" * 55)
