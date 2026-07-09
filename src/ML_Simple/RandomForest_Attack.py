import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, fbeta_score, matthews_corrcoef, classification_report
)
from pathlib import Path
import joblib

# ===================================================================
# 1. CONFIGURACIÓN Y CARGA
# ===================================================================
BASE_DIR = Path(__file__).resolve().parents[2]

print("Cargando matrices de la Fase 3...")
X_train = np.load(os.path.join(BASE_DIR, 'data', 'final', 'X_train_attack.npy'))
X_test = np.load(os.path.join(BASE_DIR, 'data', 'final', 'X_test_attack.npy'))
y_train = np.load(os.path.join(BASE_DIR, 'data', 'final', 'y_train_attack.npy'))
y_test = np.load(os.path.join(BASE_DIR, 'data', 'final', 'y_test_attack.npy'))

if len(X_train.shape) == 3:
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

# ===================================================================
# 2. ENTRENAMIENTO CON REGULARIZACIÓN (EVITANDO OVERFITTING)
# ===================================================================
print("Entrenando Baseline con restricciones de complejidad...")

# Limitamos la profundidad (max_depth) y aumentamos el requisito de 
# muestras en hojas (min_samples_leaf) para evitar el overfitting puro.
rf_baseline = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,             # Fuerza al modelo a no aprender reglas demasiado específicas
    min_samples_leaf=5,       # Evita que el modelo aprenda ruido de grupos muy pequeños
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_baseline.fit(X_train, y_train)

joblib.dump(rf_baseline, os.path.join(BASE_DIR, 'src', 'ML_Simple','Resultados', 'attack_classifier_rf_baseline.pkl'))


# ===================================================================
# 3. EVALUACIÓN Y VALIDACIÓN
# ===================================================================
y_pred = rf_baseline.predict(X_test)

# Métricas
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
f2 = fbeta_score(y_test, y_pred, beta=2, average='macro')
mcc = matthews_corrcoef(y_test, y_pred)

print("\n" + "="*60)
print(" RENDIMIENTO BASELINE REGULARIZADO: RANDOM FOREST")
print("="*60)
print(f"- Exactitud (Accuracy)       : {acc:.4f}")
print(f"- Precisión (Macro)          : {prec:.4f}")
print(f"- Sensibilidad (Recall Macro): {rec:.4f}")
print(f"- F1-Score (Macro)           : {f1:.4f}")
print(f"- F2-Score (Defensivo)       : {f2:.4f}")
print(f"- Coeficiente Matthews (MCC) : {mcc:.4f}")
print("="*60)
print(classification_report(y_test, y_pred, digits=4))