import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

run = Path(os.environ["TESIS_RUN_DIR"])

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"Archivo sin encabezado: {path}")
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        return list(reader)

requests = read_csv(run / "requests.csv")
expected_ip = os.environ.get("TESIS_CLIENT_IP")
if expected_ip and any(r["src_ip"].strip() != expected_ip for r in requests):
    raise RuntimeError("El origen registrado no coincide con el cliente seleccionado.")
flows = read_csv(Path(os.environ["TESIS_CSV"]))

if len(requests) != 9 or len({r["request_id"] for r in requests}) != 9:
    raise RuntimeError("Este piloto requiere nueve solicitudes con IDs distintos.")
if any(r["status"] != "SUCCESS" for r in requests):
    raise RuntimeError("Hay solicitudes fallidas.")
if not flows:
    raise RuntimeError("CICFlowMeter no produjo filas.")

index = defaultdict(list)
for r in requests:
    key = (
        r["src_ip"].strip(), int(r["src_port"]),
        r["dst_ip"].strip(), int(r["dst_port"]), 6
    )
    index[key].append(r["request_id"])

audit = []
counts = Counter()

for number, row in enumerate(flows, 2):
    key = (
        row["Src IP"].strip(), int(float(row["Src Port"])),
        row["Dst IP"].strip(), int(float(row["Dst Port"])),
        int(float(row["Protocol"]))
    )
    reverse = (key[2], key[3], key[0], key[1], key[4])

    ids = index.get(key, [])
    direction = "cliente-servidor"
    if not ids:
        ids = index.get(reverse, [])
        direction = "servidor-cliente"

    if len(ids) != 1:
        raise RuntimeError(
            f"Fila {number}: correspondencia ausente o ambigua."
        )

    counts[ids[0]] += 1
    audit.append({
        "csv_row": number,
        "request_ids": ids[0],
        "direction": direction,
        "src_ip": key[0],
        "src_port": key[1],
        "dst_ip": key[2],
        "dst_port": key[3],
        "protocol": key[4],
        "timestamp": row.get("Timestamp", ""),
        "fwd_packets": row.get("Total Fwd Packet", ""),
        "bwd_packets": row.get("Total Bwd packets", ""),
        "duration_us": row.get("Flow Duration", ""),
    })

if set(counts) != {r["request_id"] for r in requests}:
    raise RuntimeError("Hay solicitudes sin segmentos capturados.")

with (run / "flow_request_audit.csv").open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(f, fieldnames=list(audit[0]))
    writer.writeheader()
    writer.writerows(audit)

print(
    f"Auditoria OK: {len(requests)} solicitudes, "
    f"{len(flows)} segmentos."
)
print(
    "Segmentos por solicitud:",
    dict(sorted(counts.items(), key=lambda x: int(x[0])))
)
