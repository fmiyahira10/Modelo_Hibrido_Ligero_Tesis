import csv
import json
import os
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

run = Path(os.environ["TESIS_RUN_DIR"])
results = Path(os.environ["TESIS_RESULTADOS"])

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

files = list(results.glob("lab-lote-*.csv"))
if len(files) != 1:
    raise RuntimeError(
        f"Se esperaba un CSV de inferencia; encontrados: {len(files)}"
    )

predictions = read_csv(files[0])
audit = {
    int(r["csv_row"]): r
    for r in read_csv(run / "flow_request_audit.csv")
}
requests = {
    r["request_id"]: r
    for r in read_csv(run / "requests.csv")
}

if not predictions or len(predictions) != len(audit):
    raise RuntimeError(
        "La cantidad de predicciones no coincide con la auditoria."
    )

joined = []
groups = defaultdict(list)

for number, prediction in enumerate(predictions, 1):
    if prediction["flow_id"] != f"lab-{number:06d}":
        raise RuntimeError(
            "El orden de los resultados cambio; revisar el emparejamiento."
        )

    row = audit[number + 1]

    for name in ("src_ip", "dst_ip"):
        if prediction[name].strip() != row[name].strip():
            raise RuntimeError(f"IPs diferentes en fila {number + 1}")

    for name in ("src_port", "dst_port", "protocol"):
        if int(float(prediction[name])) != int(row[name]):
            raise RuntimeError(
                f"Metadatos diferentes en fila {number + 1}"
            )

    request_id = row["request_ids"]
    if request_id not in requests:
        raise RuntimeError(
            "Correspondencia de solicitud ausente o ambigua."
        )

    request = requests[request_id]
    record = dict(prediction)
    record.update({
        "capture_run_id": run.name,
        "request_id": request_id,
        "csv_row": number + 1,
        "direction": row["direction"],
        "url": request["url"],
        "expected_attack": request["expected_attack"],
        "planned_encryption": request["planned_encryption"],
    })

    joined.append(record)
    groups[request_id].append(record)

with (results / "predicciones_asociadas.csv").open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(f, fieldnames=list(joined[0]))
    writer.writeheader()
    writer.writerows(joined)

success = sum(p["status"] == "SUCCESS" for p in predictions)

print(
    f"Segmentos procesados: {len(predictions)} | "
    f"OK: {success} | Error: {len(predictions) - success}"
)
print("\nResultados por solicitud, conservando todos sus segmentos:")

for request_id in sorted(requests, key=int):
    rows = groups[request_id]
    good = [r for r in rows if r["status"] == "SUCCESS"]
    attacks = dict(Counter(r["attack"] for r in good))
    encryption = dict(Counter(r["encryption"] for r in good))

    print(
        f"Solicitud {request_id}: segmentos={len(rows)}, "
        f"OK={len(good)}, A={attacks}, B={encryption}"
    )

manifest = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "capture_run_id": run.name,
    "inference_run_ids": sorted({p["run_id"] for p in predictions}),
    "source_csv": os.environ["TESIS_CSV"],
    "request_count": len(requests),
    "flow_segment_count": len(predictions),
    "successful_inferences": success,
    "failed_inferences": len(predictions) - success,
    "processing_complete": (
        success == len(predictions)
        and set(groups) == set(requests)
    ),
    "encryption_reference": (
        "planned_encryption pendiente de contrastar con PCAP"
    ),
    "scope": "piloto operativo con el carril B anterior",
}

with (results / "manifest.json").open("w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print("\nResultados guardados en:", results)

if not manifest["processing_complete"]:
    raise SystemExit(
        "Procesamiento incompleto: revisar errores y solicitudes faltantes."
    )
