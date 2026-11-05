"""Servicio LOCAL de inferencia; requiere una sonda de características aparte."""
import argparse
import json
import logging
import socket
import socketserver
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from flow_protocol import decode_message, MAX_MESSAGE_BYTES


class Server(socketserver.TCPServer):
    allow_reuse_address = True


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(5)
        try:
            payload = decode_message(self.rfile.readline(MAX_MESSAGE_BYTES + 1))
            if payload['source_kind'] == 'test_fixture' and not self.server.allow_test_fixtures:
                raise ValueError('Las muestras sintéticas están desactivadas en este servicio')
            result = self.server.engine.process_flow(payload['features_attack'], payload['features_encryption'])
            record = {key: payload[key] for key in (
                'schema_version', 'run_id', 'flow_id', 'source_kind', 'extractor_id', 'flow')}
            record.update({'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                           'probe_peer': list(self.client_address), 'result': result})
            self.server.write_record(record)
            response = {'status': 'SUCCESS', 'flow_id': payload['flow_id'], 'result': result}
        except (ValueError, KeyError, TypeError, OverflowError, socket.timeout) as exc:
            response = {'status': 'ERROR', 'error': str(exc)}
        except Exception:
            logging.exception('Fallo de inferencia o escritura; no se confirma el flujo')
            response = {'status': 'ERROR', 'error': 'Fallo interno; revisar journalctl'}
        try:
            self.wfile.write((json.dumps(response, allow_nan=False) + '\n').encode())
        except OSError:
            logging.warning('Cliente desconectado antes de la respuesta')


def make_server(engine, write_record, port=9999, allow_test_fixtures=False):
    server = Server(('127.0.0.1', port), Handler)
    server.engine = engine
    server.write_record = write_record
    server.allow_test_fixtures = allow_test_fixtures
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--port', type=int, default=9999)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--allow-test-fixtures', action='store_true')
    args = parser.parse_args()
    from inference_engine_openvino import InferenceAndCorrelationEngine, inventory
    if args.inventory_only:
        report = inventory(args.base_dir)
        print(json.dumps(report, indent=2))
        return 0 if all(x['exists'] for x in report.values()) else 2
    engine = InferenceAndCorrelationEngine(args.base_dir)
    if args.check_only:
        print(json.dumps(engine.describe(), indent=2))
        return 0
    logs = args.base_dir / 'runtime' / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    class StrictRotatingHandler(RotatingFileHandler):
        def handleError(self, record):
            raise OSError('No se pudo guardar el registro de inferencia')
    output = logging.getLogger('nids.records')
    output.setLevel(logging.INFO)
    output.propagate = False
    handler = StrictRotatingHandler(logs / 'predictions.jsonl', maxBytes=20_000_000,
                                    backupCount=10, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(message)s'))
    output.addHandler(handler)
    def write_record(record):
        output.info(json.dumps(record, ensure_ascii=False, allow_nan=False))
    write_record({'event': 'startup', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                  'configuration': engine.describe(), 'python': sys.version})
    with make_server(engine, write_record, args.port, args.allow_test_fixtures) as server:
        print(f'NIDS preparado en 127.0.0.1:{args.port}; esperando características', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    handler.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
