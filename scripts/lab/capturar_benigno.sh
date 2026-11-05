#!/usr/bin/env bash
set -euo pipefail

LLAVE="$HOME/.ssh/id_ed25519_tesis_lab"
export TESIS_CLIENT_IP="${TESIS_CLIENT_IP:-10.10.10.20}"
case "$TESIS_CLIENT_IP" in
    10.10.10.20) export TESIS_CLIENT_VLAN=10 ;;
    10.10.30.10) export TESIS_CLIENT_VLAN=30 ;;
    *) echo "Cliente no habilitado: $TESIS_CLIENT_IP" >&2; exit 2 ;;
esac
CLIENTE="tesis@$TESIS_CLIENT_IP"
sudo -v

RUN_ID="N-$(date -u +%Y%m%dT%H%M%S%NZ)"
OUT="$HOME/tesis/evidencia/$RUN_ID"
mkdir -p "$HOME/tesis/evidencia"
mkdir "$OUT"
CAP_JOB=""
CAP_RC=0

detener_captura() {
    if [[ -n "$CAP_JOB" ]]; then
        if [[ -s "$OUT/capture.pid" ]]; then
            sudo -n kill -INT "$(cat "$OUT/capture.pid")" 2>/dev/null || true
        fi
        if wait "$CAP_JOB"; then CAP_RC=0; else CAP_RC=$?; fi
        CAP_JOB=""
    fi
}
trap detener_captura EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

sudo -n bash -c 'echo "$$" > "$1"; shift; exec env LC_ALL=C /usr/bin/tcpdump "$@"' \
    bash "$OUT/capture.pid" \
    -i enp0s8 -nn -s 0 -U -Z "$(id -un)" \
    -w "$OUT/capture.pcap" > "$OUT/tcpdump.log" 2>&1 < /dev/null &
CAP_JOB=$!

LISTA=0
for intento in {1..50}; do
    if [[ -f "$OUT/tcpdump.log" ]] && grep -q 'listening on' "$OUT/tcpdump.log"; then
        LISTA=1
        break
    fi
    sleep 0.1
done

if (( LISTA == 0 )); then
    cat "$OUT/tcpdump.log"
    exit 1
fi

GEN_RC=0
if ssh -i "$LLAVE" -o BatchMode=yes -o ConnectTimeout=5 "$CLIENTE" \
    "bash ~/tesis-lab/generar_benigno.sh '$RUN_ID' 3" \
    | tee "$OUT/generador.log"; then
    GEN_RC=0
else
    GEN_RC=$?
fi

sleep 2
detener_captura

scp -i "$LLAVE" -o BatchMode=yes \
    "$CLIENTE:tesis-lab/runs/$RUN_ID/requests.csv" \
    "$CLIENTE:tesis-lab/runs/$RUN_ID/curl-errors.log" "$OUT/"

if (( GEN_RC != 0 || CAP_RC != 0 )); then
    echo "Corrida con errores. Revisar $OUT"
    exit 1
fi

FILTRO="vlan.id == ${TESIS_CLIENT_VLAN} && tcp && ((ip.addr == ${TESIS_CLIENT_IP} && ip.addr == 10.10.10.10 && (tcp.port == 80 || tcp.port == 443)) || (ip.addr == ${TESIS_CLIENT_IP} && ip.addr == 10.10.20.10 && tcp.port == 80))"

tshark -r "$OUT/capture.pcap" -Y "$FILTRO" -F pcap \
    -w "$OUT/scenario.pcap" 2> "$OUT/tshark.log"

PAQUETES=$(tshark -r "$OUT/scenario.pcap" -T fields -e frame.number \
    2>> "$OUT/tshark.log" | wc -l)

if (( PAQUETES == 0 )); then
    echo "El espejo no capturo el escenario."
    exit 1
fi

printf '%s\n' "$OUT" > "$HOME/tesis/evidencia/ultima_corrida_benigna.txt"

cat "$OUT/tcpdump.log"
printf '\nCorrida: %s\nPaquetes del escenario: %s\nCarpeta: %s\n' \
    "$RUN_ID" "$PAQUETES" "$OUT"
