"""Contrato de la sonda: un JSON por línea, vectores del MISMO flujo."""
import ipaddress
import json
import math
from datetime import datetime

MAX_MESSAGE_BYTES = 65536


def _constant(value):
    raise ValueError(f'Valor JSON no finito: {value}')


def decode_message(data):
    if len(data) > MAX_MESSAGE_BYTES or not data.endswith(b'\n'):
        raise ValueError('Mensaje demasiado grande o sin delimitador de fin de línea')
    payload = json.loads(data, parse_constant=_constant)
    validate_payload(payload)
    return payload


def validate_payload(payload):
    if not isinstance(payload, dict) or payload.get('schema_version') != 1:
        raise ValueError('schema_version debe ser 1')
    for name in ('run_id', 'flow_id'):
        if not isinstance(payload.get(name), str) or not 1 <= len(payload[name]) <= 128:
            raise ValueError(f'{name} ausente o inválido')
    if payload.get('source_kind') not in ('captured_flow', 'test_fixture'):
        raise ValueError('source_kind debe ser captured_flow o test_fixture')
    for key, size in [('features_attack', 13), ('features_encryption', 62)]:
        values = payload.get(key)
        if not isinstance(values, (list, dict)) or len(values) != size:
            raise ValueError(f'{key} requiere {size} valores')
        sequence = values.values() if isinstance(values, dict) else values
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in sequence):
            raise ValueError(f'{key} contiene datos no numéricos o no finitos')
    flow = payload.get('flow')
    if not isinstance(flow, dict):
        raise ValueError('Faltan metadatos del flujo')
    for key in ('src_ip', 'dst_ip'):
        ipaddress.ip_address(flow.get(key, ''))
    for key in ('src_port', 'dst_port'):
        value = flow.get(key)
        if type(value) is not int or not 0 <= value <= 65535:
            raise ValueError(f'{key} inválido')
    if flow.get('protocol') not in (6, 17):
        raise ValueError('Este contrato admite TCP=6 y UDP=17')
    times = []
    for key in ('started_at', 'ended_at'):
        value = flow.get(key)
        if not isinstance(value, str):
            raise ValueError(f'{key} ausente')
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if value.tzinfo is None:
            raise ValueError(f'{key} debe incluir zona horaria')
        times.append(value)
    if times[1] < times[0]:
        raise ValueError('El flujo termina antes de comenzar')
    if not isinstance(payload.get('extractor_id'), str) or not payload['extractor_id'].strip():
        raise ValueError('Identifica la versión/configuración del extractor en extractor_id')
    return payload
