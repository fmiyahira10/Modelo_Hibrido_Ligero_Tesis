import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, matthews_corrcoef, classification_report
import os
import joblib
from pathlib import Path

# ===================================================================
# 1. RUTA RELATIVA Y CARGA DE DATOS DE ATAQUE
# ===================================================================
BASE_DIR = Path(__file__).resolve().parents[2]

print("[CARRIL ATAQUES] Cargando matrices de la Fase 3...")
X_train_raw = np.load(os.path.join(BASE_DIR,'data','final', 'X_train_attack.npy'))
X_test_raw = np.load(os.path.join(BASE_DIR,'data','final', 'X_test_attack.npy'))
y_train = np.load(os.path.join(BASE_DIR,'data','final', 'y_train_attack.npy'))
y_test = np.load(os.path.join(BASE_DIR,'data','final', 'y_test_attack.npy'))

le_attack = joblib.load(os.path.join(BASE_DIR,'models', 'label_encoder_attack.pkl'))

# ===================================================================
# 2. FASE 5: EXTRACCIÓN DE EMBEDDINGS DESDE LA CNN_ATTACK
# ===================================================================
print("\n[Fase 5] Cargando CNN_Attack y extrayendo representaciones latentes (16D)...")
modelo_cnn = load_model(os.path.join(BASE_DIR,'src','Embedding', 'best_cnn_attack_model.keras'))

extractor_embeddings = Model(inputs=modelo_cnn.input, outputs=modelo_cnn.get_layer('Embedding_Attack').output)

X_train_reshaped = np.expand_dims(X_train_raw, axis=-1)
X_test_reshaped = np.expand_dims(X_test_raw, axis=-1)

X_train_embeddings = extractor_embeddings.predict(X_train_reshaped)
X_test_embeddings = extractor_embeddings.predict(X_test_reshaped)

# ===================================================================
# 3. FASE 6: ENTRENAMIENTO DE LIGHTGBM MULTICLASE
# ===================================================================
print("\n[Fase 6] Entrenando LightGBM Multiclase sobre el espacio latente...")
lgb_attack = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,          # Parámetro clave de LightGBM para controlar el crecimiento leaf-wise
    objective='multiclass',
    random_state=42,
    n_jobs=-1,
    verbose=-1              # Desactiva advertencias innecesarias en consola
)

lgb_attack.fit(X_train_embeddings, y_train)
joblib.dump(lgb_attack, os.path.join(BASE_DIR, 'src', 'Results', 'attack_classifier_lgb.pkl'))
print("-> Clasificador LightGBM de ataques guardado con éxito.")

# ===================================================================
# 4. FASE 7: EVALUACIÓN EXPERIMENTAL (MÉTRICAS TESIS)
# ===================================================================
print("\n[Fase 7] Ejecutando inferencia sobre el conjunto de prueba...")
y_pred = lgb_attack.predict(X_test_embeddings)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
f2 = fbeta_score(y_test, y_pred, beta=2.0, average='macro', zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

print("\n" + "="*60)
print("   RENDIMIENTO EXPERIMENTAL: MODELO HÍBRIDO (CNN + LIGHTGBM)")
print("="*60)
print(f" - Exactitud (Accuracy)                : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f" - Precisión (Precision Macro)         : {precision:.4f} ({precision*100:.2f}%)")
print(f" - Sensibilidad (Recall Macro)         : {recall:.4f} ({recall*100:.2f}%)")
print(f" - Puntuación F1 (F1-Score Macro)      : {f1:.4f} ({f1*100:.2f}%)")
print(f" - Puntuación F2 (F2-Score Defensivo)  : {f2:.4f} ({f2*100:.2f}%)")
print(f" - Coeficiente de Matthews (MCC)       : {mcc:.4f}")
print("="*60)

print("\n=== DESGLOSE DETALLADO DE DESEMPEÑO POR MACRO-CLASE ===")
print(classification_report(y_test, y_pred, target_names=le_attack.classes_, zero_division=0))
print("="*60)