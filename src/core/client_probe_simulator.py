"""Reproduce un JSONL de flujos ya extraídos. NO captura ni mezcla datasets."""
import argparse
import json
import socket
from pathlib import Path
from flow_protocol import decode_message, MAX_MESSAGE_BYTES


def enviar_payload(payload, port=9999):
    encoded = (json.dumps(payload, allow_nan=False) + '\n').encode()
    decode_message(encoded)
    with socket.create_connection(('127.0.0.1', port), timeout=30) as connection:
        connection.sendall(encoded)
        with connection.makefile('rb') as stream:
            raw = stream.readline(MAX_MESSAGE_BYTES + 1)
            if len(raw) > MAX_MESSAGE_BYTES or not raw.endswith(b'\n'):
                raise ValueError('Respuesta incompleta o demasiado grande')
            result = json.loads(raw)
    if result.get('status') != 'SUCCESS':
        raise RuntimeError(result.get('error', 'Inferencia fallida'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--port', type=int, default=9999)
    args = parser.parse_args()
    with args.input.open('rb') as stream:
        while True:
            line = stream.readline(MAX_MESSAGE_BYTES + 1)
            if not line:
                break
            payload = decode_message(line)
            print(json.dumps(enviar_payload(payload, args.port), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
