#!/usr/bin/env bash
set -eu

RUN_ID="${1:-}"
CICLOS="${2:-3}"

if [[ ! "$RUN_ID" =~ ^N-[0-9TZ-]+$ ]]; then
    echo "Uso: bash generar_benigno.sh N-FECHAUTC ciclos"
    exit 2
fi

if [[ ! "$CICLOS" =~ ^[0-9]+$ ]]; then
    echo "Los ciclos deben ser un entero entre 1 y 120."
    exit 2
fi

if (( CICLOS < 1 || CICLOS > 120 )); then
    echo "Los ciclos deben estar entre 1 y 120."
    exit 2
fi

DIRECTORIO="$HOME/tesis-lab/runs/$RUN_ID"
mkdir -p "$HOME/tesis-lab/runs"

if ! mkdir "$DIRECTORIO"; then
    echo "No se creará una corrida sobre una carpeta existente."
    exit 2
fi

CSV="$DIRECTORIO/requests.csv"
ERRORES="$DIRECTORIO/curl-errors.log"
touch "$ERRORES"

printf '%s\n' \
'run_id,request_id,start_utc,end_utc,url,expected_attack,planned_encryption,src_ip,src_port,dst_ip,dst_port,http_code,download_bytes,http_seconds,curl_exit,status' \
> "$CSV"

URLS=(
    "http://10.10.10.10/health"
    "https://10.10.10.10/health"
    "http://10.10.20.10/health"
)

EXITOS=0
FALLOS=0
NUMERO=0

for (( ciclo=1; ciclo<=CICLOS; ciclo++ )); do
    for URL in "${URLS[@]}"; do
        NUMERO=$((NUMERO + 1))
        INICIO=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)

        OPCIONES=()
        CIFRADO="No Cifrado"

        if [[ "$URL" == https://* ]]; then
            OPCIONES+=(--insecure)
            CIFRADO="Cifrado"
        fi

        printf 'Solicitud %s: %s\n' "$NUMERO" "$URL"

        if DATOS=$(curl \
            --noproxy '*' \
            --http1.1 \
            --silent --show-error --fail \
            --connect-timeout 3 \
            --max-time 10 \
            --header 'Connection: close' \
            --output /dev/null \
            --write-out '%{local_ip},%{local_port},%{remote_ip},%{remote_port},%{http_code},%{size_download},%{time_total}' \
            "${OPCIONES[@]}" \
            "$URL" 2>> "$ERRORES"); then
            RC=0
        else
            RC=$?
        fi

        FIN=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)

        IFS=, read -r SRC SPORT DST DPORT CODE BYTES SEGUNDOS <<< "$DATOS"

        ESTADO="ERROR"
        if [[ "$RC" == 0 && "$CODE" =~ ^2[0-9][0-9]$ ]]; then
            ESTADO="SUCCESS"
            EXITOS=$((EXITOS + 1))
        else
            FALLOS=$((FALLOS + 1))
        fi

        printf '%s,%s,%s,%s,%s,Normal,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$RUN_ID" "$NUMERO" "$INICIO" "$FIN" "$URL" "$CIFRADO" \
            "$SRC" "$SPORT" "$DST" "$DPORT" "$CODE" \
            "$BYTES" "$SEGUNDOS" "$RC" "$ESTADO" >> "$CSV"

        sleep 1
    done
done

printf '\nSolicitudes: %s | Correctas: %s | Fallos: %s\n' \
    "$NUMERO" "$EXITOS" "$FALLOS"
printf 'Registro: %s\n' "$CSV"

if (( FALLOS > 0 )); then
    exit 1
fi
