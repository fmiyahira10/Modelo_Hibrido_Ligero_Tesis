import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, matthews_corrcoef, classification_report
import os
import joblib

# ===================================================================
# 1. CONFIGURACIÓN DE RUTA RELATIVA Y CARGA DE DATOS
# ===================================================================
BASE_DIR = r'C:\Users\i21327\Desktop\Tesis\Modelo_Hibrido_Ligero_Tesis'

print("Cargando matrices NumPy y codificadores del carril de cifrado...")
X_train_raw = np.load(os.path.join(BASE_DIR,'data','final', 'X_train_encryption.npy'))
X_test_raw = np.load(os.path.join(BASE_DIR,'data','final', 'X_test_encryption.npy'))
y_train = np.load(os.path.join(BASE_DIR,'data','final', 'y_train_encryption.npy'))
y_test = np.load(os.path.join(BASE_DIR,'data','final', 'y_test_encryption.npy'))

le_encryption = joblib.load(os.path.join(BASE_DIR,'models', 'label_encoder_encryption.pkl'))

# ===================================================================
# 2. FASE 5: EXTRACCIÓN DE EMBEDDINGS BINARIOS
# ===================================================================
print("\n[Fase 5] Cargando CNN_Encryption y extrayendo espacio latente de 16 dimensiones...")
modelo_cnn = load_model(os.path.join(BASE_DIR,'src','Embedding', 'best_cnn_encryption_model.keras'))

# Truncamos el clasificador de la CNN para apuntar a la capa de salida latente
extractor_embeddings = Model(inputs=modelo_cnn.input, outputs=modelo_cnn.get_layer('Embedding_Encryption').output)

# Formateo tridimensional exigido por la Conv1D
X_train_reshaped = np.expand_dims(X_train_raw, axis=-1)
X_test_reshaped = np.expand_dims(X_test_raw, axis=-1)

# Extracción masiva de características optimizadas
X_train_embeddings = extractor_embeddings.predict(X_train_reshaped)
X_test_embeddings = extractor_embeddings.predict(X_test_reshaped)

# ===================================================================
# 3. FASE 6: ENTRENAMIENTO DE RANDOM FOREST (ESTADO DE CIFRADO)
# ===================================================================
print("\n[Fase 6] Entrenando clasificador Random Forest sobre el espacio latente binario...")
rf_encryption = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1, # Paralelismo de hilos masivo para optimizar tiempos
    verbose=1
)

rf_encryption.fit(X_train_embeddings, y_train)
print("-> Clasificador de encriptación entrenado exitosamente.")

# Guardamos el clasificador binario final para el bloque de inferencia paralela
joblib.dump(rf_encryption, os.path.join(BASE_DIR, 'encryption_classifier_rf.pkl'))

# ===================================================================
# 4. FASE 7: EVALUACIÓN EXPERIMENTAL MEDIANTE MÉTRICAS DE TESIS
# ===================================================================
print("\n[Fase 7] Ejecutando evaluación estricta sobre el set de prueba...")
y_pred = rf_encryption.predict(X_test_embeddings)

# Cálculo de métricas de control (ponderación binaria directa)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

# F2-Score: Prioriza penalizar los falsos negativos criptográficos (Beta=2)
f2 = fbeta_score(y_test, y_pred, beta=2.0, average='binary', zero_division=0)

# MCC: Coeficiente de correlación de Matthews puro para verificar consistencia binaria
mcc = matthews_corrcoef(y_test, y_pred)

# ===================================================================
# 5. REPORTE TÉCNICO DE RENDIMIENTO EN CONSOLA
# ===================================================================
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