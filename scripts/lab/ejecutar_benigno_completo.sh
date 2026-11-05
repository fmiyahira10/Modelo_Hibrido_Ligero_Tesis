#!/usr/bin/env bash
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BASE"

PY="$BASE/.venv_openvino312/bin/python"
CFM="$HOME/herramientas/CICFlowMeter/build/install/CICFlowMeter/bin"
MARCADOR="$HOME/tesis/evidencia/ultima_corrida_benigna.txt"

export TESIS_CLIENT_IP="${TESIS_CLIENT_IP:-10.10.10.20}"
case "$TESIS_CLIENT_IP" in
    10.10.10.20) export TESIS_CLIENT_VLAN=10 ;;
    10.10.30.10) export TESIS_CLIENT_VLAN=30 ;;
    *) echo "Cliente no habilitado: $TESIS_CLIENT_IP" >&2; exit 2 ;;
esac
CLIENTE="tesis@$TESIS_CLIENT_IP"

mkdir -p "$BASE/runtime"
exec 9> "$BASE/runtime/benigno.lock"
flock -n 9 || {
    echo "Ya hay una corrida completa en ejecucion."
    exit 1
}

[[ -x "$PY" && -x "$CFM/cfm" ]] || {
    echo "Falta Python o CICFlowMeter."
    exit 1
}

ss -H -lnt 'sport = :9999' | grep -q . || {
    echo "Inicia el servidor OpenVINO en 9999."
    exit 1
}

ssh -i "$HOME/.ssh/id_ed25519_tesis_lab" \
    -o BatchMode=yes -o ConnectTimeout=5 \
    "$CLIENTE" \
    'if [ -f /etc/cron.d/tesis-lab ] || pgrep -f "[p]rogramador_v2.py" >/dev/null; then echo "Pausa el programador anterior en el cliente seleccionado." >&2; exit 1; fi'

ANTERIOR="$(cat "$MARCADOR" 2>/dev/null || true)"
bash scripts/lab/capturar_benigno.sh
RUN_DIR="$(cat "$MARCADOR")"

[[ "$RUN_DIR" != "$ANTERIOR" ]] || {
    echo "No se registro una captura nueva."
    exit 1
}

ETAPA="revision de captura"
trap 'printf "ERROR: %s\n" "$ETAPA" > "$RUN_DIR/estado.txt"' ERR
printf 'EN_CURSO\n' > "$RUN_DIR/estado.txt"

grep -Eq '^0 packets dropped by kernel$' "$RUN_DIR/tcpdump.log"

export TZ=UTC
export TESIS_RUN_DIR="$RUN_DIR"
export TESIS_CSV="$RUN_DIR/cfm/scenario.pcap_Flow.csv"
export TESIS_VECTORES="$RUN_DIR/vectores_prueba.jsonl"
export TESIS_RESULTADOS="$RUN_DIR/inferencia"
export TESIS_BASE_DIR="$BASE"

mkdir "$RUN_DIR/cfm" "$TESIS_RESULTADOS"

ETAPA="CICFlowMeter"
(
    cd "$CFM"
    timeout 180 ./cfm "$RUN_DIR/scenario.pcap" "$RUN_DIR/cfm"
) > "$RUN_DIR/cfm.log" 2>&1

ETAPA="auditoria"
"$PY" scripts/lab/auditar_benigno.py \
    | tee "$RUN_DIR/auditoria.log"

ETAPA="validacion"
"$PY" validar_vectores_lab.py \
    2>&1 | tee "$RUN_DIR/validacion.log"

grep -Fxq 'Flujos rechazados: 0' "$RUN_DIR/validacion.log"
grep -Fxq '[OK] Orden del carril A validado' "$RUN_DIR/validacion.log"
grep -Fxq '[OK] Orden del carril B validado' "$RUN_DIR/validacion.log"
[[ -s "$TESIS_VECTORES" ]]

ETAPA="inferencia"
"$PY" ejecutar_lote.py \
    2>&1 | tee "$TESIS_RESULTADOS/lote.log"

ETAPA="resumen"
"$PY" scripts/lab/resumir_benigno.py \
    2>&1 | tee "$TESIS_RESULTADOS/resumen.log"

ETAPA="exportacion de predicciones"
"$PY" - <<'PY'
import csv
import json
import os
from pathlib import Path

run = Path(os.environ["TESIS_RUN_DIR"])
out = Path(os.environ["TESIS_RESULTADOS"])
base = Path(os.environ["TESIS_BASE_DIR"])

manifest = json.loads((out / "manifest.json").read_text())
if not manifest["processing_complete"]:
    raise RuntimeError("El resumen informa procesamiento incompleto.")

files = list(out.glob("lab-lote-*.csv"))
if len(files) != 1:
    raise RuntimeError("Cantidad inesperada de archivos de resultados.")

with files[0].open(encoding="utf-8-sig", newline="") as f:
    expected = {r["flow_id"] for r in csv.DictReader(f)}

records = {}
for path in (base / "runtime/logs").glob("predictions.jsonl*"):
    if path.name != "predictions.jsonl" and not path.suffix[1:].isdigit():
        continue

    with path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("run_id") in manifest["inference_run_ids"]:
                fid = record.get("flow_id")
                if fid in records:
                    raise RuntimeError(f"Prediccion duplicada: {fid}")
                records[fid] = record

if set(records) != expected:
    raise RuntimeError(
        "Los registros del servidor no coinciden con los resultados."
    )

with (run / "predictions.jsonl").open("w", encoding="utf-8") as f:
    for fid in sorted(records):
        f.write(
            json.dumps(
                records[fid], ensure_ascii=False, allow_nan=False
            ) + "\n"
        )

manifest["pipeline_complete"] = True
manifest["server_prediction_count"] = len(records)
manifest["client_ip"] = os.environ["TESIS_CLIENT_IP"]
manifest["client_ssh_user"] = "tesis"
manifest["observation_policy"] = "client_vlan"
manifest["observation_vlan_id"] = int(os.environ["TESIS_CLIENT_VLAN"])

(run / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2)
)
print("Predicciones del servidor exportadas:", len(records))
PY

printf 'COMPLETO\n' > "$RUN_DIR/estado.txt"
printf '\nPILOTO COMPLETO\nEvidencia: %s\n' "$RUN_DIR"
