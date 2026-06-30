import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, matthews_corrcoef, classification_report
import os
import joblib
from pathlib import Path

# ===================================================================
# 1. RUTA RELATIVA Y CARGA DE DATOS DE CIFRADO
# ===================================================================
BASE_DIR = Path(__file__).resolve().parents[2]

print("[CARRIL CIFRADO] Cargando matrices de la Fase 3...")
X_train_raw = np.load(os.path.join(BASE_DIR,'data','final', 'X_train_encryption.npy'))
X_test_raw = np.load(os.path.join(BASE_DIR,'data','final', 'X_test_encryption.npy'))
y_train = np.load(os.path.join(BASE_DIR,'data','final', 'y_train_encryption.npy'))
y_test = np.load(os.path.join(BASE_DIR,'data','final', 'y_test_encryption.npy'))

le_encryption = joblib.load(os.path.join(BASE_DIR,'models', 'label_encoder_encryption.pkl'))
# ===================================================================
# 2. FASE 5: EXTRACCIÓN DE EMBEDDINGS DESDE LA CNN_ENCRYPTION
# ===================================================================
print("\n[Fase 5] Cargando CNN_Encryption y extrayendo espacio latente (16D)...")
modelo_cnn = load_model(os.path.join(BASE_DIR,'src','Embedding', 'best_cnn_encryption_model.keras'))

extractor_embeddings = Model(inputs=modelo_cnn.input, outputs=modelo_cnn.get_layer('Embedding_Encryption').output)

X_train_reshaped = np.expand_dims(X_train_raw, axis=-1)
X_test_reshaped = np.expand_dims(X_test_raw, axis=-1)

X_train_embeddings = extractor_embeddings.predict(X_train_reshaped)
X_test_embeddings = extractor_embeddings.predict(X_test_reshaped)

# ===================================================================
# 3. FASE 6: ENTRENAMIENTO DE LIGHTGBM BINARIO
# ===================================================================
print("\n[Fase 6] Entrenando LightGBM Binario para Estado de Cifrado...")
lgb_encryption = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    objective='binary',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_encryption.fit(X_train_embeddings, y_train)
joblib.dump(lgb_encryption, os.path.join(BASE_DIR, 'src', 'Results', 'encryption_classifier_lgb.pkl'))
print("-> Clasificador de cifrado guardado con éxito.")

# ===================================================================
# 4. FASE 7: EVALUACIÓN EXPERIMENTAL (MÉTRICAS TESIS)
# ===================================================================
print("\n[Fase 7] Ejecutando inferencia sobre el conjunto de prueba...")
y_pred = lgb_encryption.predict(X_test_embeddings)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
f2 = fbeta_score(y_test, y_pred, beta=2.0, average='binary', zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

print("\n" + "="*60)
print("   RENDIMIENTO EXPERIMENTAL: MODELO HÍBRIDO DE ENCRIPTIÓN")
print("="*60)
print(f" - Exactitud (Accuracy)                : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f" - Precisión (Precision)               : {precision:.4f} ({precision*100:.2f}%)")
print(f" - Sensibilidad (Recall)               : {recall:.4f} ({recall*100:.2f}%)")
print(f" - Puntuación F1 (F1-Score)            : {f1:.4f} ({f1*100:.2f}%)")
print(f" - Puntuación F2 (F2-Score Cripto)     : {f2:.4f} ({f2*100:.2f}%)")
print(f" - Coeficiente de Matthews (MCC)       : {mcc:.4f}")
print("="*60)

print("\n=== REPORTE DE CLASIFICACIÓN DETALLADO ===")
print(classification_report(y_test, y_pred, target_names=le_encryption.classes_, zero_division=0))
print("="*60)