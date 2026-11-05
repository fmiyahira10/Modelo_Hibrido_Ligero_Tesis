import copy
import json
import socket
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/core'))
from flow_protocol import decode_message, validate_payload, MAX_MESSAGE_BYTES
from server_nids_core import make_server
from inference_engine import inventory, raw_frame, InferenceAndCorrelationEngine
import joblib
import numpy as np


def fixture():
    return {'schema_version': 1, 'run_id': 'unit-test', 'flow_id': 'synthetic-1',
            'source_kind': 'test_fixture', 'extractor_id': 'unit-test-no-capture',
            'flow': {'src_ip': '10.10.10.10', 'dst_ip': '10.10.30.10',
                     'src_port': 12345, 'dst_port': 443, 'protocol': 6,
                     'started_at': '2026-10-18T18:00:00Z',
                     'ended_at': '2026-10-18T18:00:01Z'},
            'features_attack': [1.0] * 13, 'features_encryption': [1.0] * 62}


class ContractTests(unittest.TestCase):
    def test_dimensions_and_nonfinite(self):
        for key in ('features_attack', 'features_encryption'):
            p = fixture(); p[key].pop()
            with self.assertRaises(ValueError): validate_payload(p)
            p = fixture(); p[key][0] = float('nan')
            with self.assertRaises(ValueError): validate_payload(p)
            p = fixture(); p[key][0] = True
            with self.assertRaises(ValueError): validate_payload(p)

    def test_metadata_and_times(self):
        for edit in [('src_ip', 'not-an-ip'), ('src_port', 65536),
                     ('started_at', '2026-10-18T18:00:02Z'),
                     ('ended_at', '2026-10-18T18:00:01')]:
            p = fixture(); p['flow'][edit[0]] = edit[1]
            with self.assertRaises(ValueError): validate_payload(p)

    def test_frame_boundaries(self):
        raw = (json.dumps(fixture()) + '\n').encode()
        self.assertEqual(decode_message(raw)['flow_id'], 'synthetic-1')
        with self.assertRaises(ValueError): decode_message(raw[:-1])
        with self.assertRaises(ValueError): decode_message(b' ' * MAX_MESSAGE_BYTES + b'\n')
        with self.assertRaises(ValueError): decode_message(raw.replace(b'1.0', b'NaN', 1))

    def test_fragmented_tcp_larger_than_4096(self):
        records = []
        engine = Mock()
        engine.process_flow.return_value = {'alert': False}
        with make_server(engine, records.append, port=0, allow_test_fixtures=True) as server:
            thread = threading.Thread(target=server.handle_request)
            thread.start()
            p = fixture(); p['extractor_id'] = 'unit-test-' + 'x' * 6000
            raw = (json.dumps(p) + '\n').encode()
            with socket.create_connection(server.server_address, timeout=3) as conn:
                for offset in range(0, len(raw), 71):
                    conn.sendall(raw[offset:offset+71])
                response = json.loads(conn.makefile('rb').readline())
            thread.join(3)
        self.assertEqual(response['status'], 'SUCCESS')
        self.assertEqual(records[0]['flow']['src_ip'], '10.10.10.10')
        self.assertEqual(records[0]['probe_peer'][0], '127.0.0.1')
        engine.process_flow.assert_called_once()

    def test_production_rejects_synthetic(self):
        engine = Mock()
        with make_server(engine, Mock(), port=0) as server:
            thread = threading.Thread(target=server.handle_request); thread.start()
            with socket.create_connection(server.server_address, timeout=3) as conn:
                conn.sendall((json.dumps(fixture()) + '\n').encode())
                result = json.loads(conn.makefile('rb').readline())
            thread.join(3)
        self.assertEqual(result['status'], 'ERROR')
        engine.process_flow.assert_not_called()

    def test_actual_scaler_order(self):
        for lane, n in [('attack', 13), ('encryption', 62)]:
            scaler = joblib.load(ROOT / f'models/scalers_encoders/robust_scaler_{lane}.pkl')
            names = list(scaler.feature_names_in_)
            self.assertEqual(len(names), n)
            raw = dict(reversed(list(zip(names, range(n)))))
            frame = raw_frame(raw, scaler)
            np.testing.assert_equal(frame.to_numpy()[0], np.arange(n))
            raw.pop(names[0]); raw['unknown'] = 0
            with self.assertRaises(ValueError): raw_frame(raw, scaler)

    def test_real_encoder_mapping(self):
        encoder = joblib.load(ROOT / 'models/scalers_encoders/label_encoder_encryption.pkl')
        self.assertEqual(encoder.inverse_transform([0, 1]).tolist(), ['Cifrado', 'No Cifrado'])

    def test_missing_artifacts_reported(self):
        report = inventory(ROOT)
        self.assertEqual(len(report), 8)
        self.assertTrue(report['scaler_attack']['exists'])

    def test_scaling_once_and_decoding_with_mock_classifiers(self):
        engine = InferenceAndCorrelationEngine.__new__(InferenceAndCorrelationEngine)
        for lane in ('attack', 'encryption'):
            scaler = joblib.load(ROOT / f'models/scalers_encoders/robust_scaler_{lane}.pkl')
            wrapper = Mock(wraps=scaler)
            wrapper.feature_names_in_ = scaler.feature_names_in_
            setattr(engine, 'scaler_' + lane, wrapper)
            encoder = joblib.load(ROOT / f'models/scalers_encoders/label_encoder_{lane}.pkl')
            setattr(engine, 'le_' + lane, encoder)
            clf = Mock(); clf.classes_ = np.arange(len(encoder.classes_))
            label = 'Normal' if lane == 'attack' else 'Cifrado'
            code = encoder.transform([label])[0]
            clf.predict.return_value = [code]
            clf.predict_proba.return_value = np.eye(len(encoder.classes_))[[code]]
            setattr(engine, 'clf_rf_' + lane, clf)
            setattr(engine, 'extractor_' + lane, Mock(return_value=np.zeros((1,16))))
        result = engine.process_flow([1.] * 13, [1.] * 62)
        self.assertFalse(result['alert'])
        self.assertEqual(result['Carril_B_cifrado'], 'Cifrado')
        self.assertEqual(result['action_executed'], 'none')
        engine.scaler_attack.transform.assert_called_once()
        engine.scaler_encryption.transform.assert_called_once()
        self.assertEqual(engine.extractor_attack.call_args.args[0].shape, (1,13,1))
        self.assertEqual(engine.extractor_encryption.call_args.args[0].shape, (1,62,1))


if __name__ == '__main__':
    unittest.main()
