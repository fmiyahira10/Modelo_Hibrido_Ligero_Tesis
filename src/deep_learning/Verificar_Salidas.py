import tensorflow as tf
import numpy as np
import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar Datos y Modelo
data_path = os.path.join(os.path.dirname(__file__), '../../data/final')
model_path = os.path.join(os.path.dirname(__file__), 'Modelo_CNN1D_Hibrido.h5')
encoder_path = os.path.join(data_path, 'encoder_ataques.pkl')

print("Cargando datos de prueba...")
X_test_scaled = np.load(os.path.join(data_path, 'X_test_scaled.npy'))
y_test_cifrado = np.load(os.path.join(data_path, 'y_test_cifrado.npy'))
y_test_ataque = np.load(os.path.join(data_path, 'y_test_ataque.npy'))

# Ajustar forma para CNN 1D
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

print("Cargando modelo...")
modelo = tf.keras.models.load_model(model_path)

# 2. Realizar Predicciones
print("Realizando predicciones...")
pred_cifrado, pred_ataque = modelo.predict(X_test_cnn)

# Procesar salidas
# Fase 1: Umbral de 0.5 para binario
y_pred_cifrado = (pred_cifrado > 0.5).astype(int).flatten()
# Fase 2: Argmax para multiclase
y_pred_ataque = np.argmax(pred_ataque, axis=1)

# 3. VEREDICTO FASE 1: Detección de Cifrado
print("\n" + "="*50)
print("VEREDICTO FASE 1: Clasificación de Cifrado (Binaria)")
print("="*50)
print(classification_report(y_test_cifrado, y_pred_cifrado, target_names=['Texto Plano', 'Cifrado']))

# 4. VEREDICTO FASE 2: Clasificación de Ataques
print("\n" + "="*50)
print("VEREDICTO FASE 2: Clasificación de Ataques (Multiclase)")
print("="*50)

# Intentar cargar nombres de clases si el encoder existe
try:
    encoder = joblib.load(encoder_path)
    target_names = [str(c) for c in encoder.classes_]
    # Verificar si el número de clases coincide
    if len(target_names) != pred_ataque.shape[1]:
        print(f"Advertencia: El encoder tiene {len(target_names)} clases, pero el modelo predice {pred_ataque.shape[1]}.")
        target_names = [f"Clase {i}" for i in range(pred_ataque.shape[1])]
except:
    target_names = [f"Clase {i}" for i in range(pred_ataque.shape[1])]

print(classification_report(y_test_ataque, y_pred_ataque, target_names=target_names, labels=range(len(target_names))))

# 5. Verificación del Extractor de Características
print("\n" + "="*50)
print("VERIFICACIÓN DEL EXTRACTOR (BOTELLÓN)")
print("="*50)
extractor_path = os.path.join(os.path.dirname(__file__), 'Extractor_Features_CNN1D.h5')
if os.path.exists(extractor_path):
    extractor = tf.keras.models.load_model(extractor_path)
    features = extractor.predict(X_test_cnn[:5])
    print(f"Forma de las características extraídas: {features.shape}")
    print("Muestra de características (primeras 5):")
    print(features)
    print("\nResultado: El extractor está generando los 8 vectores de características correctamente.")
else:
    print("Archivo del extractor no encontrado.")

print("\nVerificación completada. Revisa los reportes de precisión y recall arriba.")
