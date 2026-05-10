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

# En Keras, class_weight no soporta un diccionario de diccionarios para modelos multi-salida.
# La solución correcta es usar sample_weight. Creamos un arreglo con los pesos de muestra.
print("Convirtiendo pesos de clase a pesos de muestra (sample_weight)...")
sample_weights_cifrado = np.ones(shape=(len(y_train_cifrado),))
sample_weights_cifrado[y_train_cifrado == 0] = pesos_cifrado[0]
sample_weights_cifrado[y_train_cifrado == 1] = pesos_cifrado[1]

# Como tenemos dos salidas, Keras nos pide un arreglo de pesos para cada una.
# Para el ataque (Fase 2) le damos peso de 1.0 a todo para no alterarlo.
sample_weights_ataque = np.ones(shape=(len(y_train_ataque),))

print(f"Peso para Tráfico Normal (Clase 0): {pesos_cifrado[0]:.4f}")
print(f"Peso para Tráfico Cifrado (Clase 1): {pesos_cifrado[1]:.4f} <- ¡El modelo le prestará muchísima más atención!")

# ---------------------------------------------------------
# 7. ENTRENAMIENTO DEL MODELO
# ---------------------------------------------------------
print("\nIniciando el entrenamiento...")
historia = modelo_cnn.fit(
    X_train_cnn, 
    # Le pasamos las etiquetas reales para las dos salidas en forma de lista [cifrado, ataque]
    [y_train_cifrado, y_train_ataque],
    
    # Validation data nos permite ver en tiempo real si el modelo está memorizando (overfitting) o generalizando bien
    validation_data=(
        X_test_cnn, 
        [y_test_cifrado, y_test_ataque]
    ),
    
    # Inyectamos los pesos de muestra correspondientes a cada salida en el mismo orden
    sample_weight=[sample_weights_cifrado, sample_weights_ataque],
    
    epochs=15, # Empezamos con 15 iteraciones (épocas) para probar
    batch_size=512, # Un lote grande hace que el entrenamiento sea veloz
    verbose=1
)

# Al terminar, guardamos el modelo entrenado completo
modelo_cnn.save("Modelo_CNN1D_Hibrido.h5")
print("\n¡Entrenamiento finalizado y modelo multitarea guardado!")

# ---------------------------------------------------------
# 8. EXTRACCIÓN DE CARACTERÍSTICAS PARA MACHINE LEARNING
# ---------------------------------------------------------
# Para que tu modelo devuelva las mejores características a Machine Learning,
# creamos un "submodelo" que va desde la entrada hasta la capa de cuello de botella (bottleneck)
print("\nCreando el extractor de características...")
extractor_features = Model(
    inputs=modelo_cnn.input, 
    outputs=modelo_cnn.get_layer('Capa_Extractora_Features').output,
    name='Extractor_Features'
)

# Guardamos el extractor para cargarlo fácilmente en la fase de Machine Learning
extractor_features.save("Extractor_Features_CNN1D.h5")
print("¡Modelo extractor de características guardado con éxito como 'Extractor_Features_CNN1D.h5'!")