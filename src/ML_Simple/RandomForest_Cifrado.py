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
# 1. CONFIGURACIÓN Y CARGA DE MATRICES (CARRIL B: CIFRADO)
# ===================================================================
BASE_DIR = Path(__file__).resolve().parents[2]

print("Cargando matrices de la Fase 3 para el carril de Cifrado...")
# Nota: Ajusta los nombres de archivo si en tu Fase 3 los guardaste como '_cifrado.npy'
X_train = np.load(os.path.join(BASE_DIR, 'data', 'final', 'X_train_encryption.npy'))
X_test = np.load(os.path.join(BASE_DIR, 'data', 'final', 'X_test_encryption.npy'))
y_train = np.load(os.path.join(BASE_DIR, 'data', 'final', 'y_train_encryption.npy'))
y_test = np.load(os.path.join(BASE_DIR, 'data', 'final', 'y_test_encryption.npy'))

# Asegurar dimensiones correctas para Machine Learning tradicional (2D)
if len(X_train.shape) == 3:
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

print(f"Dimensiones de entrenamiento: {X_train.shape}")
print(f"Dimensiones de prueba: {X_test.shape}\n")

# ===================================================================
# 2. ENTRENAMIENTO CON REGULARIZACIÓN ESTRICTA (ANTI-OVERFITTING)
# ===================================================================
print("Entrenando Modelo Baseline SImétrico: Random Forest Regularizado...")

# Usamos la misma configuración del Carril A para una comparación metodológicamente justa
rf_baseline_crypto = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,             # Controla el crecimiento desmedido del árbol
    min_samples_leaf=5,       # Evita que el modelo memorice muestras aisladas (ruido)
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_baseline_crypto.fit(X_train, y_train)
print("Entrenamiento completado exitosamente.")

# Guardar el artefacto del modelo entrenado
output_dir = os.path.join(BASE_DIR, 'src', 'ML_Simple', 'Resultados')
os.makedirs(output_dir, exist_ok=True)
joblib.dump(rf_baseline_crypto, os.path.join(output_dir, 'encryption_classifier_rf_baseline.pkl'))

# ===================================================================
# 3. EVALUACIÓN EXPERIMENTAL FORMAL
# ===================================================================
print("Generando inferencia sobre el conjunto de prueba...")
y_pred = rf_baseline_crypto.predict(X_test)

# Cálculo de las 6 métricas reglamentarias de tu metodología
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
f2 = fbeta_score(y_test, y_pred, beta=2, average='macro') # F2-Score enfocado en mitigar Falsos Negativos
mcc = matthews_corrcoef(y_test, y_pred)

print("\n" + "="*60)
print(" RENDIMIENTO BASELINE REGULARIZADO: RANDOM FOREST (CIFRADO)")
print("="*60)
print(f"- Exactitud (Accuracy)       : {acc:.4f} ({acc*100:.2f}%)")
print(f"- Precisión (Precision Macro) : {prec:.4f} ({prec*100:.2f}%)")
print(f"- Sensibilidad (Recall Macro) : {rec:.4f} ({rec*100:.2f}%)")
print(f"- Puntuación F1 (F1-Score Macro): {f1:.4f} ({f1*100:.2f}%)")
print(f"- Puntuación F2 (F2-Score Cripto): {f2:.4f} ({f2*100:.2f}%)")
print(f"- Coeficiente de Matthews (MCC) : {mcc:.4f}")
print("="*60)

print("\n=== REPORTE DE CLASIFICACIÓN DETALLADO ===")
print(classification_report(y_test, y_pred, digits=4))