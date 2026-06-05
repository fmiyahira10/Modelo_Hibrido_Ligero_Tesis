import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, matthews_corrcoef, classification_report
import os
import joblib
from pathlib import Path

# ===================================================================
# 1. CONFIGURACIÓN DE RUTA RELATIVA Y CARGA DE DATOS
# ===================================================================
# Raíz del proyecto: sube 2 niveles desde src/machine_learning/
BASE_DIR = Path(__file__).resolve().parents[2]

print("Cargando matrices NumPy y codificadores...")
X_train_raw = np.load(os.path.join(BASE_DIR,'data','final', 'X_train_attack.npy'))
X_test_raw = np.load(os.path.join(BASE_DIR,'data','final', 'X_test_attack.npy'))
y_train = np.load(os.path.join(BASE_DIR,'data','final', 'y_train_attack.npy'))
y_test = np.load(os.path.join(BASE_DIR,'data','final', 'y_test_attack.npy'))

le_attack = joblib.load(os.path.join(BASE_DIR,'models', 'label_encoder_attack.pkl'))

# ===================================================================
# 2. FASE 5: EXTRACCIÓN DE EMBEDDINGS DESDE LA CNN OPERACIONAL
# ===================================================================
print("\n[Fase 5] Cargando arquitectura profunda y extrayendo representaciones latentes (16D)...")
modelo_cnn = load_model(os.path.join(BASE_DIR,'src','Embedding', 'best_cnn_attack_model.keras'))

# Truncar la red para usar la capa intermedia 'Embedding_Attack' como salida
extractor_embeddings = Model(inputs=modelo_cnn.input, outputs=modelo_cnn.get_layer('Embedding_Attack').output)

# Adecuar dimensiones para la Conv1D (muestras, variables, 1)
X_train_reshaped = np.expand_dims(X_train_raw, axis=-1)
X_test_reshaped = np.expand_dims(X_test_raw, axis=-1)

# Generar matrices de embeddings reducidas y potenciadas
X_train_embeddings = extractor_embeddings.predict(X_train_reshaped)
X_test_embeddings = extractor_embeddings.predict(X_test_reshaped)

print(f"-> Matriz de entrenamiento compactada para ML : {X_train_embeddings.shape}")
print(f"-> Matriz de testeo independiente para ML       : {X_test_embeddings.shape}")

# ===================================================================
# 3. FASE 6: ENTRENAMIENTO ROBUSTO DE RANDOM FOREST
# ===================================================================
print("\n[Fase 6] Entrenando clasificador Random Forest sobre el espacio latente...")
# Instanciamos el modelo con hiperparámetros balanceados y eficientes
rf_classifier = RandomForestClassifier(
    n_estimators=100,       # Cantidad óptima de árboles para evitar sobreajuste
    random_state=42, 
    n_jobs=-1,              # Utilizar todos los núcleos del procesador en paralelo
    verbose=1
)

rf_classifier.fit(X_train_embeddings, y_train)
print("-> Modelo Random Forest entrenado exitosamente.")

# Guardar el clasificador entrenado para el motor de correlación final
joblib.dump(rf_classifier, os.path.join(BASE_DIR, 'src', 'Results', 'attack_classifier_rf.pkl'))

# ===================================================================
# 4. FASE 7: EVALUACIÓN EXPERIMENTAL MEDIANTE MÉTRICAS DE TESIS
# ===================================================================
print("\n[Fase 7] Ejecutando inferencia sobre el conjunto de prueba independiente...")
y_pred = rf_classifier.predict(X_test_embeddings)

# Cálculo de métricas globales (utilizando promedio 'macro' para dar el mismo peso a cada tipo de ataque)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

# F2-Score: Ponderamos beta=2.0 para priorizar la sensibilidad defensiva (Recall)
f2 = fbeta_score(y_test, y_pred, beta=2.0, average='macro', zero_division=0)

# MCC: El coeficiente de Matthews extendido para evaluación multiclase completa
mcc = matthews_corrcoef(y_test, y_pred)

# ===================================================================
# 5. REPORTE TÉCNICO FORMAL PARA ARTÍCULO CIENTÍFICO
# ===================================================================
print("\n" + "="*60)
# Las citas bibliográficas de tu marco teórico validan la elección de estas métricas ante el jurado
print("   RENDIMIENTO EXPERIMENTAL: MODELO HÍBRIDO (CNN + RANDOM FOREST)")
print("="*60)
print(f" - Exactitud (Accuracy)                : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f" - Precisión (Precision Macro)         : {precision:.4f} ({precision*100:.2f}%)")
print(f" - Sensibilidad (Recall Macro)         : {recall:.4f} ({recall*100:.2f}%)")
print(f" - Puntuación F1 (F1-Score Macro)      : {f1:.4f} ({f1*100:.2f}%)")
print(f" - Puntuación F2 (F2-Score Defensivo)  : {f2:.4f} ({f2*100:.2f}%)")
print(f" - Coeficiente de Matthews (MCC)       : {mcc:.4f}")
print("="*60)

print("\n=== DESGLOSE DETALLADO DE DESEMPEÑO POR MACRO-CLASE ===")
# Mapea los códigos a los nombres reales de tus categorías de ataque
print(classification_report(y_test, y_pred, target_names=le_attack.classes_, zero_division=0))
print("="*60)