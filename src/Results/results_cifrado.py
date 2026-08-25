import numpy as np
from pathlib import Path
import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, fbeta_score, matthews_corrcoef, 
                             classification_report, confusion_matrix)
import joblib

# ===================================================================
# 1. CONFIGURACIÓN DE RUTA RELATIVA Y CARGA DE DATOS
# ===================================================================
BASE_DIR = Path(__file__).resolve().parents[2]

print("Cargando matrices NumPy y codificadores...")
X_test_raw = np.load(os.path.join(BASE_DIR, 'data', 'final', 'X_test_encryption.npy'))
y_test = np.load(os.path.join(BASE_DIR, 'data', 'final', 'y_test_encryption.npy'))

# Cargar Label Encoder para ver los nombres reales de los ataques
le_attack = joblib.load(os.path.join(BASE_DIR, 'models', 'scalers_encoders', 'label_encoder_encryption.pkl'))

# ===================================================================
# 2. CARGAR CNN Y EXTRAER EMBEDDINGS (Paso Crucial)
# ===================================================================
print("\nCargando arquitectura profunda para extraer representaciones latentes (16D)...")
modelo_cnn = load_model(os.path.join(BASE_DIR, 'models', 'deep_learning', 'best_cnn_encryption_model.keras'))

# Truncar la red en la capa intermedia 'Embedding_Encryption'
extractor_embeddings = Model(inputs=modelo_cnn.input, outputs=modelo_cnn.get_layer('Embedding_Encryption').output)

# Redimensionar para Conv1D (muestras, variables, 1)
X_test_reshaped = np.expand_dims(X_test_raw, axis=-1)

# Generar la matriz de 16 características que el Random Forest espera
print("Transformando características crudas a espacio latente de 16D...")
X_test_embeddings = extractor_embeddings.predict(X_test_reshaped)

# ===================================================================
# 3. CARGAR RANDOM FOREST Y EJECUTAR INFERENCIA
# ===================================================================
print("\nCargando clasificador Random Forest guardado...")
modelo_rf = joblib.load(os.path.join(BASE_DIR, 'models', 'classifiers', 'encryption_classifier_rf.pkl'))

print("Ejecutando inferencia sobre los embeddings de prueba...")
y_pred = modelo_rf.predict(X_test_embeddings)

# ===================================================================
# 4. REPORTE TÉCNICO FORMAL (Métricas de Tesis)
# ===================================================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
f2 = fbeta_score(y_test, y_pred, beta=2.0, average='macro', zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

print("\n" + "="*60)
print("   RENDIMIENTO EXPERIMENTAL DESDE MODELO GUARDADO (.PKL)")
print("="*60)
print(f" - Exactitud (Accuracy)            : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f" - Precisión (Precision Macro)     : {precision:.4f} ({precision*100:.2f}%)")
print(f" - Sensibilidad (Recall Macro)     : {recall:.4f} ({recall*100:.2f}%)")
print(f" - Puntuación F1 (F1-Score Macro)  : {f1:.4f} ({f1*100:.2f}%)")
print(f" - Puntuación F2 (F2-Score Defensivo)  : {f2:.4f} ({f2*100:.2f}%)")
print(f" - Coeficiente de Matthews (MCC)   : {mcc:.4f}")
print("="*60)

print("\n=== DESGLOSE DETALLADO DE DESEMPEÑO POR MACRO-CLASE ===")
print(classification_report(y_test, y_pred, target_names=le_attack.classes_, zero_division=0))
print("="*60)
