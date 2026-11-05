import os
os.environ["KERAS_BACKEND"] = "openvino"

from keras.models import Model
from keras.models import load_model as _keras_load_model

def load_model(*args, **kwargs):
    kwargs["compile"] = False
    return _keras_load_model(*args, **kwargs)

"""Inferencia de vectores crudos. No captura paquetes ni crea características."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

import hashlib
from importlib.metadata import version
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

ARTIFACTS = {
    'scaler_attack': 'models/scalers_encoders/robust_scaler_attack.pkl',
    'le_attack': 'models/scalers_encoders/label_encoder_attack.pkl',
    'clf_rf_attack': 'models/classifiers/attack_classifier_rf.pkl',
    'scaler_encryption': 'models/scalers_encoders/robust_scaler_encryption.pkl',
    'le_encryption': 'models/scalers_encoders/label_encoder_encryption.pkl',
    'clf_rf_encryption': 'models/classifiers/encryption_classifier_rf.pkl',
    'cnn_full_attack': 'models/deep_learning/best_cnn_attack_model.keras',
    'cnn_full_encryption': 'models/deep_learning/best_cnn_encryption_model.keras',
}


def inventory(base_dir):
    result = {}
    for key, relative in ARTIFACTS.items():
        path = Path(base_dir) / relative
        record = {'path': relative, 'exists': path.is_file()}
        if path.is_file():
            with path.open('rb') as stream:
                record['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
            record['bytes'] = path.stat().st_size
        result[key] = record
    return result


def raw_frame(values, scaler):
    names = list(scaler.feature_names_in_)
    if isinstance(values, dict):
        if set(values) != set(names):
            raise ValueError('Características ausentes o desconocidas; no se rellenan con ceros')
        values = [values[name] for name in names]
    a = np.asarray(values)
    if a.ndim != 1 or len(a) != len(names) or a.dtype.kind not in 'iuf':
        raise ValueError(f'Se requieren {len(names)} valores numéricos crudos en una dimensión')
    a = a.astype(np.float64)
    if not np.isfinite(a).all():
        raise ValueError('Hay NaN o infinito en las características')
    return pd.DataFrame([a], columns=names)


class InferenceAndCorrelationEngine:
    def __init__(self, base_dir):
        self.tesis_dir = Path(base_dir).resolve()
        self.artifacts = inventory(self.tesis_dir)
        missing = [v['path'] for v in self.artifacts.values() if not v['exists']]
        if missing:
            raise FileNotFoundError('Faltan artefactos: ' + ', '.join(missing))
        with warnings.catch_warnings():
            warnings.simplefilter('error', InconsistentVersionWarning)
            for attr, relative in ARTIFACTS.items():
                path = self.tesis_dir / relative
                obj = load_model(path, compile=False) if path.suffix == '.keras' else joblib.load(path)
                setattr(self, attr, obj)
        for lane, count, layer in [('attack', 13, 'Embedding_Attack'),
                                   ('encryption', 62, 'Embedding_Encryption')]:
            scaler = getattr(self, 'scaler_' + lane)
            model = getattr(self, 'cnn_full_' + lane)
            clf = getattr(self, 'clf_rf_' + lane)
            encoder = getattr(self, 'le_' + lane)
            if getattr(scaler, 'n_features_in_', None) != count:
                raise ValueError(f'Escalador incompatible: {lane}')
            if len(getattr(scaler, 'feature_names_in_', [])) != count:
                raise ValueError(f'Falta el orden de características de {lane}')
            if tuple(model.input_shape[1:]) != (count, 1):
                raise ValueError(f'Entrada CNN incompatible: {lane}: {model.input_shape}')
            extractor = Model(inputs=model.inputs, outputs=model.get_layer(layer).output)
            if extractor.output_shape[-1] != getattr(clf, 'n_features_in_', None):
                raise ValueError(f'Embedding y clasificador incompatibles: {lane}')
            if not np.array_equal(np.sort(clf.classes_), np.arange(len(encoder.classes_))):
                raise ValueError(f'Codificador y clases incompatibles: {lane}')
            if hasattr(clf, 'n_jobs'):
                clf.n_jobs = int(os.environ.get('NIDS_RF_THREADS', '2'))
            if hasattr(clf, 'verbose'):
                clf.verbose = 0
            setattr(self, 'extractor_' + lane, extractor)
        if set(self.le_attack.classes_) != {'Normal', 'DoS', 'Generic', 'Malware_Exploits',
                                            'Access_Attacks', 'Reconnaissance'}:
            raise ValueError('Las macroclases no coinciden con el contrato')
        if set(self.le_encryption.classes_) != {'Cifrado', 'No Cifrado'}:
            raise ValueError('Las clases de cifrado no coinciden con el contrato')

    def describe(self):
        return {'versions': {name: version(name) for name in
                ('keras', 'keras', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib')},
                'artifacts': self.artifacts, 'features': {
            lane: getattr(self, 'scaler_' + lane).feature_names_in_.tolist()
            for lane in ('attack', 'encryption')}, 'classes': {
            lane: getattr(self, 'le_' + lane).classes_.tolist()
            for lane in ('attack', 'encryption')}}

    def process_flow(self, features_attack_raw, features_encryption_raw):
        start = time.perf_counter()
        labels, scores = {}, {}
        for lane, raw in [('attack', features_attack_raw), ('encryption', features_encryption_raw)]:
            scaler = getattr(self, 'scaler_' + lane)
            scaled = scaler.transform(raw_frame(raw, scaler))
            if not np.isfinite(scaled).all():
                raise ValueError('El escalado produjo valores no finitos')
            embedding = np.asarray(getattr(self, 'extractor_' + lane)(scaled[..., None], training=False))
            if not np.isfinite(embedding).all():
                raise ValueError('El embedding contiene valores no finitos')
            clf = getattr(self, 'clf_rf_' + lane)
            encoder = getattr(self, 'le_' + lane)
            code = clf.predict(embedding)[0]
            labels[lane] = str(encoder.inverse_transform([code])[0])
            probabilities = clf.predict_proba(embedding)[0]
            scores[lane] = {str(label): float(p) for label, p in zip(
                encoder.inverse_transform(clf.classes_), probabilities)}
        attack, encryption = labels['attack'], labels['encryption']
        return {
            'Carril_A_ataques': attack, 'Carril_B_cifrado': encryption,
            'Veredicto_IDS': self._correlate_verdicts(attack, encryption),
            'alert': attack != 'Normal', 'action_executed': 'none', 'scores': scores,
            'inference_ms': (time.perf_counter() - start) * 1000,
            'encryption_label_scope': 'VPN/Tor frente a NonVPN/Non-Tor según entrenamiento',
        }

    @staticmethod
    def _correlate_verdicts(attack, encryption):
        if attack == 'Normal':
            return f'Sin ataque predicho; carril B: {encryption}'
        return f'Ataque predicho: {attack}; carril B: {encryption}; pendiente de evaluación'
