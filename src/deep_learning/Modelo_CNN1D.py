import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import joblib
import os


# 1. Ajuste de Forma (Reshape) para la CNN 1D
# Las redes convolucionales 1D necesitan que los datos tengan la forma: (filas, características, canales)

data_path = os.path.join(os.path.dirname(__file__), '../../data/final')

X_train_scaled = np.load(os.path.join(data_path, 'X_train_scaled.npy'))
X_test_scaled = np.load(os.path.join(data_path, 'X_test_scaled.npy'))
y_train_ataque = np.load(os.path.join(data_path, 'y_train_ataque.npy'))
y_train_cifrado = np.load(os.path.join(data_path, 'y_train_cifrado.npy'))
y_test_cifrado = np.load(os.path.join(data_path, 'y_test_cifrado.npy'))
y_test_ataque = np.load(os.path.join(data_path, 'y_test_ataque.npy'))

print("Ajustando dimensiones para la CNN 1D...")
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# 2. Diseño de la Arquitectura Ligera
input_layer = Input(shape=(X_train_cnn.shape[1], 1), name='Entrada_13_Features')

# Primera capa convolucional (Extracción de patrones locales)
x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_layer)
x = MaxPooling1D(pool_size=2)(x) # Reduce dimensiones para mantener el modelo ligero y rápido

# Segunda capa convolucional (Extracción de patrones más profundos)
x = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(x)
x = MaxPooling1D(pool_size=2)(x)

x = Flatten()(x)

# 3. EL CUELLO DE BOTELLA (¡El corazón de tu tesis!)
# Esta capa extrae los 8 "features" de alto nivel que usarás en tu segunda fase
bottleneck = Dense(8, activation='relu', name='Capa_Extractora_Features')(x)

# 4. Capas de Salida Multitarea (Fase 1 y Fase 2)
# Salida Binaria (0 = Normal/Texto plano, 1 = Cifrado)
out_cifrado = Dense(1, activation='sigmoid', name='Salida_Fase1_Cifrado')(bottleneck)

# Salida Multiclase (Las 6 Macro-clases de Ataque)
out_ataque = Dense(6, activation='softmax', name='Salida_Fase2_Ataque')(bottleneck)

# 5. Construcción y Compilación del Modelo
modelo_cnn = Model(inputs=input_layer, outputs=[out_cifrado, out_ataque])

modelo_cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    # Dos funciones de pérdida porque hace dos tareas distintas a la vez
    loss={
        'Salida_Fase1_Cifrado': 'binary_crossentropy',
        'Salida_Fase2_Ataque': 'sparse_categorical_crossentropy' 
    },
    metrics={
        'Salida_Fase1_Cifrado': ['accuracy'],
        'Salida_Fase2_Ataque': ['accuracy']
    }
)

print("\nResumen de la arquitectura de la red:")
modelo_cnn.summary()

# ---------------------------------------------------------
# 6. Cálculo de Pesos de Clase para proteger el Tráfico Cifrado
# ---------------------------------------------------------
print("\nCalculando pesos matemáticos para equilibrar la Fase 1...")
pesos_cifrado = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_cifrado),
    y=y_train_cifrado
)

# Convertimos los pesos a un diccionario para inyectarlo en Keras
# Al asignarle el nombre de la capa, Keras sabe a qué salida aplicar estos pesos
dict_pesos = {
    'Salida_Fase1_Cifrado': {0: pesos_cifrado[0], 1: pesos_cifrado[1]}
}

print(f"Peso para Tráfico Normal (Clase 0): {pesos_cifrado[0]:.4f}")
print(f"Peso para Tráfico Cifrado (Clase 1): {pesos_cifrado[1]:.4f} <- ¡El modelo le prestará muchísima más atención!")

# ---------------------------------------------------------
# 7. ENTRENAMIENTO DEL MODELO
# ---------------------------------------------------------
print("\nIniciando el entrenamiento...")
historia = modelo_cnn.fit(
    X_train_cnn, 
    # Le pasamos las etiquetas reales para las dos salidas
    {'Salida_Fase1_Cifrado': y_train_cifrado, 'Salida_Fase2_Ataque': y_train_ataque},
    
    # Validation data nos permite ver en tiempo real si el modelo está memorizando (overfitting) o generalizando bien
    validation_data=(
        X_test_cnn, 
        {'Salida_Fase1_Cifrado': y_test_cifrado, 'Salida_Fase2_Ataque': y_test_ataque}
    ),
    
    epochs=15, # Empezamos con 15 iteraciones (épocas) para probar
    batch_size=512, # Un lote grande hace que el entrenamiento sea veloz
    class_weight=dict_pesos, # Inyectamos los pesos para proteger la clase minoritaria
    verbose=1
)

# Al terminar, guardamos el modelo entrenado completo
modelo_cnn.save("Modelo_CNN1D_Hibrido.h5")
print("\n¡Entrenamiento finalizado y modelo guardado!")