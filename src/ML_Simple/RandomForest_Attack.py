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
# 1. CONFIGURACIÓN DE RUTAS RELATIVAS
# ===================================================================
BASE_DIR = Path(__file__).resolve().parents[2]

print("Cargando matrices de la Fase 3 para el carril de Ataques...")
X_train = np.load(os.path.join(BASE_DIR, 'data','final', 'X_train_attack.npy'))
X_test = np.load(os.path.join(BASE_DIR, 'data','final', 'X_test_attack.npy'))
y_train = np.load(os.path.join(BASE_DIR, 'data','final', 'y_train_attack.npy'))
y_test = np.load(os.path.join(BASE_DIR, 'data','final', 'y_test_attack.npy'))

# Asegurar dimensiones correctas para Machine Learning tradicional (2D)
if len(X_train.shape) == 3:
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

print(f"Dimensiones de entrenamiento: {X_train.shape}")
print(f"Dimensiones de prueba: {X_test.shape}\n")

# ===================================================================
# 2. ENTRENAMIENTO DE LÍNEA BASE (SIN EMBEDDING)
# ===================================================================
print("Entrenando Modelo Baseline: Random Forest (Características Originales)...")
rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_baseline.fit(X_train, y_train)
print("Entrenamiento completado exitosamente.")

joblib.dump(rf_baseline, os.path.join(BASE_DIR, 'src', 'ML_Simple','Resultados', 'attack_classifier_rf_baseline.pkl'))
# ===================================================================
# 3. EVALUACIÓN EXPERIMENTAL FORMAL
# ===================================================================
print("Generando inferencia sobre el conjunto de prueba...")
y_pred = rf_baseline.predict(X_test)

# Cálculo de las 6 métricas reglamentarias
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
f2 = fbeta_score(y_test, y_pred, beta=2, average='macro')
mcc = matthews_corrcoef(y_test, y_pred)

print("\n" + "="*60)
print("   RENDIMIENTO BASELINE: RANDOM FOREST DIRECTO (SIN CNN)")
print("="*60)
print(f"- Exactitud (Accuracy)                : {acc:.4f} ({acc*100:.2f}%)")
print(f"- Precisión (Precision Macro)         : {prec:.4f} ({prec*100:.2f}%)")
print(f"- Sensibilidad (Recall Macro)         : {rec:.4f} ({rec*100:.2f}%)")
print(f"- Puntuación F1 (F1-Score Macro)      : {f1:.4f} ({f1*100:.2f}%)")
print(f"- Puntuación F2 (F2-Score Defensivo)  : {f2:.4f} ({f2*100:.2f}%)")
print(f"- Coeficiente de Matthews (MCC)       : {mcc:.4f}")
print("="*60)

print("\n=== DESGLOSE DETALLADO DE DESEMPEÑO POR MACRO-CLASE ===")
# Nota: Si cuentas con la lista de etiquetas mapeadas, puedes pasarla en target_names
print(classification_report(y_test, y_pred, digits=4))