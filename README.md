# Modelo Híbrido Ligero NIDS: Deep Learning & Machine Learning

Este proyecto implementa un **Sistema de Detección de Intrusiones en Red (NIDS)** de arquitectura híbrida y ligera, diseñado para la detección avanzada de ciberamenazas y clasificación de tráfico cifrado en tiempo real. 

El sistema utiliza **Redes Neuronales Convolucionales 1D (CNN 1D)** para la reducción de dimensionalidad y extracción de representaciones latentes (*embeddings* de 16 dimensiones), combinadas con clasificadores de **Machine Learning (Random Forest, LightGBM y XGBoost)** para la decisión final y un **Motor de Inferencia y Correlación Lógica** para la mitigación de falsos positivos.

---

## 🚀 Arquitectura del Sistema (Carril Dual)

El sistema opera mediante un esquema de procesamiento en dos carriles independientes que se correlacionan en el motor central:

1. **Carril A (Vector de Ataques):**
   - **Propósito:** Detección y clasificación de categorías de intrusión (DoS, Malware/Exploits, Access Attacks, Normal).
   - **Datos de Entrada:** 13 características métricas de flujo (CICIDS2017 + UNSW-NB15).
   - **Extractor Latente:** CNN 1D truncada en la capa `Embedding_Attack` (16D).
   - **Clasificador Final:** Random Forest / LightGBM / XGBoost.

2. **Carril B (Análisis de Tráfico Cifrado):**
   - **Propósito:** Identificación de tráfico cifrado vs. no cifrado (Darknet).
   - **Datos de Entrada:** 62 características estadísticas de flujo.
   - **Extractor Latente:** CNN 1D truncada en la capa `Embedding_Encryption` (16D).
   - **Clasificador Final:** Random Forest / LightGBM / XGBoost.

3. **Motor de Correlación y Decisión Lógica:**
   - Evaluador central (`src/core/inference_engine.py`) que aplica reglas institucionales para detectar amenazas complejas (ej. *Malware/Exploits encapsulados en túneles VPN/Tor*) y disipar falsos positivos.

---

## 📊 Dimensiones de Datasets y Matrices Finales

### Carril A: Vector de Ataques (13 Características)
* **Entrenamiento (`X_train_attack`):** `(870,778, 13)`
* **Validación (`X_val_attack`):** `(610,168, 13)`
* **Prueba Final (`X_test_attack`):** `(610,168, 13)`

### Carril B: Tráfico Cifrado (62 Características)
* **Entrenamiento (`X_train_encryption`):** `(31,871, 62)`
* **Validación (`X_val_encryption`):** `(6,829, 62)`
* **Prueba Final (`X_test_encryption`):** `(6,830, 62)`

---

## 📁 Estructura del Repositorio

```text
├── data/
│   ├── raw/                # Datasets originales (CICIDS2017, Darknet, UNSW-NB15)
│   ├── processed/          # Datasets unificados y limpios (formato Parquet)
│   ├── final/              # Matrices NumPy (.npy) preprocesadas y escaladas
│   └── scripts/            # Scripts de limpieza, homologación y unificación de labels
├── models/                 # Artefactos y modelos entrenados
│   ├── scalers_encoders/   # Escaladores (RobustScaler) y codificadores (.pkl)
│   ├── deep_learning/      # Extractores de embeddings Keras (.keras)
│   └── classifiers/        # Clasificadores finales de ML (RF, LGBM, XGBoost en .pkl)
├── reports/
│   └── figures/            # Curvas de aprendizaje, t-SNE y diagnósticos de overfitting (.png)
├── src/
│   ├── core/               # Módulos centrales de inferencia y despliegue en tiempo real
│   │   ├── inference_engine.py      # Motor de correlación y reglas institucionales
│   │   ├── server_nids_core.py      # Servidor TCP Socket NIDS
│   │   └── client_probe_simulator.py# Sonda cliente simuladora de tráfico real
│   ├── preprocessing/      # Módulos de limpieza y estandarización
│   ├── EDA/                # Exploración de datos y gráficos exploratorios
│   ├── deep_learning/      # Scripts de entrenamiento de CNNs 1D
│   ├── machine_learning/   # Scripts de entrenamiento de clasificadores en espacio latente
│   ├── ML_Simple/          # Modelos de Machine Learning baseline directos
│   └── Results/            # Scripts de evaluación y métricas formales
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Documentación general del repositorio
```

---

## 📥 Descarga de Datos y Artefactos

Debido al tamaño de los conjuntos de datos masivos, las matrices finales y modelos no se almacenan completamente en el repositorio de Git.

1. **Datos Originales:** Coloque los archivos descargados en `data/raw/`.
2. **Matrices `.npy` Procesadas:** Los archivos escalados necesarios para entrenamiento e inferencia se pueden descargar en:
   - 🔗 [Google Drive - Datasets del Proyecto](https://drive.google.com/drive/folders/1rMrunG7TCJuxXgy-Yn6GJ_4IdAychIyz?usp=sharing)
3. **Ubicación:** Guarde las matrices descargadas (`X_train_attack.npy`, `X_train_encryption.npy`, etc.) en `data/final/`.

---

## ⚙️ Instalación y Requisitos

1. **Clonar el repositorio:**
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd Modelo_Hibrido_Ligero_Tesis
   ```

2. **Crear y activar entorno virtual:**
   ```bash
   python -m venv .venv
   # En Windows PowerShell:
   \.venv\Scripts\Activate.ps1
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃‍♂️ Ejecución y Despliegue en Tiempo Real

Para probar la inferencia del sistema híbrido en tiempo real mediante sockets TCP:

1. **Iniciar el Servidor NIDS (Core):**
   ```bash
   python src/core/server_nids_core.py
   ```
   *El servidor cargará en memoria los artefactos desde `models/`, extractores Keras y clasificadores, quedando listo en `127.0.0.1:9999`.*

2. **Ejecutar la Sonda Simuladora de Tráfico:**
   ```bash
   python src/core/client_probe_simulator.py
   ```
   *La sonda transmitirá ráfagas de tráfico extraídas de los datasets reales y el servidor emitirá los veredictos integrados.*
=
