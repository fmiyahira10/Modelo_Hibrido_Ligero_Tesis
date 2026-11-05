import csv
import os
import json
import math
import re
import socket
import statistics
import sys
import time

from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import Counter

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src/core"))

from flow_protocol import decode_message, MAX_MESSAGE_BYTES

CSV_ORIGEN = Path(os.environ.get("TESIS_CSV", str(Path.home() / "tesis/flujos_csv/flujos_lab.csv")))
CSV_RESULTADOS = Path(os.environ.get("TESIS_RESULTADOS", str(BASE / "runtime/results")))

MAPEO_A = [
    "Flow Duration",
    "Total Fwd Packet",
    "Total Bwd packets",
    "Total Length of Fwd Packet",
    "Total Length of Bwd Packet",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "FWD Init Win Bytes",
    "Bwd Init Win Bytes",
    "Flow Packets/s",
    "Flow Bytes/s"
]

def leer_caracteristicas():
    grupos = [[], []]
    carril = None

    ruta = BASE / "caracteristicas_modelos.txt"

    for linea in ruta.read_text(encoding="utf-8-sig").splitlines():
        titulo = re.match(r"^\s*([12])\.\s+MODELO\b", linea)

        if titulo:
            carril = int(titulo.group(1)) - 1
            continue

        m = re.match(r"^\s*(\d+)\.\s+(.+?)\s*$", linea)

        if carril is not None and m:
            if int(m.group(1)) == len(grupos[carril]) + 1:
                grupos[carril].append(m.group(2).strip())

    if list(map(len, grupos)) != [13, 62]:
        raise RuntimeError("No se encontraron las 13 y 62 características")

    return grupos

def convertir_fecha(valor):
    valor = str(valor).strip()
    valor = re.sub(
        r"\b([ap])\.\s*m\.",
        lambda m: m.group(1).upper() + "M",
        valor, flags=re.IGNORECASE,
    )

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
            "%m/%d/%Y %I:%M:%S %p",
            "%m/%d/%Y %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f"
        ]

        for formato in formatos:
            try:
                fecha = datetime.strptime(valor, formato)
                break
            except ValueError:
                continue

    if fecha is None:
        raise ValueError(f"Timestamp inválido: {valor}")

    # Mantiene la interpretación UTC de la prueba anterior.
    # Debe verificarse con la zona real de captura.
    if fecha.tzinfo is None:
        fecha = fecha.replace(tzinfo=timezone.utc)

    return fecha

def crear_vectores(fila, columnas_b):
    vector_a = [float(fila[c]) for c in MAPEO_A]
    vector_b = [float(fila[c]) for c in columnas_b]

    if not all(math.isfinite(x) for x in vector_a + vector_b):
        raise ValueError("Características no finitas")

    return vector_a, vector_b

caracteristicas_a, caracteristicas_b = leer_caracteristicas()

with CSV_ORIGEN.open(encoding="utf-8-sig", newline="") as f:
    filas = list(csv.DictReader(f))

if not filas:
    raise RuntimeError("El archivo flujos_lab.csv está vacío")

columnas = set(MAPEO_A + caracteristicas_b + [
    "Src IP", "Dst IP", "Src Port",
    "Dst Port", "Protocol", "Timestamp"
])

faltantes = columnas - set(filas[0])

if faltantes:
    raise RuntimeError(f"Columnas faltantes: {sorted(faltantes)}")

# Comprobar que las características coincidan
# con el flujo que ya validamos anteriormente.
ruta_prueba = Path(os.environ.get("TESIS_VECTORES", str(Path.home() / "tesis/flujos_csv/vectores_prueba.jsonl")))

with ruta_prueba.open(encoding="utf-8") as f:
    muestra = json.loads(next(f))

coincidencias = 0

for fila in filas:
    if fila["Src IP"] != muestra["src_ip"]:
        continue
    if fila["Dst IP"] != muestra["dst_ip"]:
        continue

    a, b = crear_vectores(fila, caracteristicas_b)

    anterior = (
        muestra["features_attack"]
        + muestra["features_encryption"]
    )

    actual = a + b

    if all(
        math.isclose(x, y, rel_tol=1e-7, abs_tol=1e-6)
        for x, y in zip(actual, anterior)
    ):
        coincidencias += 1

if coincidencias == 0:
    raise RuntimeError(
        "Los vectores no coinciden con la muestra validada. "
        "Deteniendo el lote para revisar el mapeo."
    )

print("[OK] Vectores contrastados con la prueba anterior")

# Permite probar un subconjunto con: python ejecutar_lote.py 5
if len(sys.argv) > 1:
    filas = filas[:int(sys.argv[1])]

run_id = "lab-lote-" + datetime.now(
    timezone.utc
).strftime("%Y%m%dT%H%M%SZ")

CSV_RESULTADOS.mkdir(parents=True, exist_ok=True)
salida = CSV_RESULTADOS / f"{run_id}.csv"

campos = [
    "run_id", "flow_id", "src_ip", "dst_ip",
    "src_port", "dst_port", "protocol",
    "attack", "encryption", "verdict", "alert",
    "score_attack", "score_encryption",
    "inference_ms", "rtt_ms", "status", "error"
]

conteo_a = Counter()
conteo_b = Counter()
tiempos = []
inicio_lote = time.perf_counter()

print(f"\nIniciando lote: {len(filas)} flujos")
print("Run ID:", run_id)

with salida.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=campos)
    writer.writeheader()

    for i, fila in enumerate(filas, 1):
        flow_id = f"lab-{i:06d}"

        registro = {
            "run_id": run_id,
            "flow_id": flow_id,
            "src_ip": fila["Src IP"],
            "dst_ip": fila["Dst IP"],
            "src_port": fila["Src Port"],
            "dst_port": fila["Dst Port"],
            "protocol": fila["Protocol"],
            "status": "ERROR",
            "error": ""
        }

        inicio = time.perf_counter()

        try:
            a, b = crear_vectores(fila, caracteristicas_b)

            fecha_inicio = convertir_fecha(fila["Timestamp"])
            duracion = float(fila["Flow Duration"])

            if not math.isfinite(duracion) or duracion < 0:
                raise ValueError("Duración inválida")

            fecha_fin = fecha_inicio + timedelta(
                microseconds=duracion
            )

            payload = {
                "schema_version": 1,
                "run_id": run_id,
                "flow_id": flow_id,
                "source_kind": "captured_flow",
                "extractor_id": "CICFlowMeter-PCAP-13A-62B-v1",
                "flow": {
                    "src_ip": fila["Src IP"],
                    "dst_ip": fila["Dst IP"],
                    "src_port": int(fila["Src Port"]),
                    "dst_port": int(fila["Dst Port"]),
                    "protocol": int(fila["Protocol"]),
                    "started_at": fecha_inicio.isoformat(),
                    "ended_at": fecha_fin.isoformat()
                },
                "features_attack": a,
                "features_encryption": b
            }

            mensaje = (
                json.dumps(payload, allow_nan=False) + "\n"
            ).encode("utf-8")

            decode_message(mensaje)

            with socket.create_connection(
                ("127.0.0.1", 9999), timeout=10
            ) as sock:

                sock.settimeout(60)
                sock.sendall(mensaje)

                respuesta = sock.makefile("rb").readline(
                    MAX_MESSAGE_BYTES + 1
                )

            if not respuesta:
                raise RuntimeError("Respuesta vacía")

            recibido = json.loads(respuesta)

            if recibido.get("status") != "SUCCESS":
                raise RuntimeError(
                    recibido.get("error", "Error NIDS")
                )

            if recibido.get("flow_id") != flow_id:
                raise RuntimeError("ID de respuesta incorrecto")

            resultado = recibido["result"]

            ataque = resultado["Carril_A_ataques"]
            cifrado = resultado["Carril_B_cifrado"]

            registro.update({
                "status": "SUCCESS",
                "attack": ataque,
                "encryption": cifrado,
                "verdict": resultado["Veredicto_IDS"],
                "alert": resultado.get("alert"),
                "score_attack": resultado.get(
                    "scores", {}
                ).get("attack", {}).get(ataque),
                "score_encryption": resultado.get(
                    "scores", {}
                ).get("encryption", {}).get(cifrado),
                "inference_ms": resultado.get("inference_ms")
            })

            conteo_a[ataque] += 1
            conteo_b[cifrado] += 1

            if isinstance(
                resultado.get("inference_ms"), (int, float)
            ):
                tiempos.append(resultado["inference_ms"])

        except Exception as e:
            registro["status"] = "ERROR"
            registro["error"] = str(e)

        registro["rtt_ms"] = round(
            (time.perf_counter() - inicio) * 1000, 3
        )

        writer.writerow(registro)

        if i % 25 == 0 or i == len(filas):
            f.flush()
            print(
                f"Procesados: {i}/{len(filas)} | "
                f"OK: {sum(conteo_a.values())} | "
                f"Error: {i - sum(conteo_a.values())}",
                flush=True
            )

duracion_total = time.perf_counter() - inicio_lote

print("\n========== RESUMEN ==========")
print("Archivo:", salida)
print("Carril A:", dict(conteo_a))
print("Carril B:", dict(conteo_b))

if tiempos:
    ordenados = sorted(tiempos)
    p95 = ordenados[math.ceil(len(ordenados) * .95) - 1]

    print(f"Latencia promedio: {statistics.mean(tiempos):.2f} ms")
    print(f"Latencia mediana: {statistics.median(tiempos):.2f} ms")
    print(f"Latencia P95: {p95:.2f} ms")

print(
    "Throughput:",
    f"{sum(conteo_a.values()) / duracion_total:.2f} flujos/s"
)
print("=============================")
