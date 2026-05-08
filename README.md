# Proyecto Híbrido de Ciberseguridad: Deep Learning & Machine Learning

Este proyecto implementa una arquitectura híbrida para la detección de intrusiones y ataques en red, utilizando modelos de **Deep Learning** para la selección de características y algoritmos de **Machine Learning** para la clasificación final.

## Estructura del Proyecto

La organización del repositorio es la siguiente:

- **data/**: Contiene los conjuntos de datos en sus diferentes etapas (Ver sección de Descarga).
    - `raw/`: Datos originales (CSV, ZIP) tal como se descargaron.
    - `processed/`: Datos intermedios procesados (formato Parquet).
    - `final/`: Archivos finales para entrenamiento (.npy escalados).
- **models/**: Almacena artefactos como encoders (.pkl), scalers y modelos.
- **notebooks/**: Jupyter Notebooks para análisis (EDA) y experimentación.
- **src/**: Código fuente.
    - `preprocessing/`: Scripts para limpieza y estandarización.
    - `deep_learning/`: Modelos para extracción de características (CNN1D).
    - `machine_learning/`: Clasificadores finales.
- **requirements.txt**: Dependencias del proyecto.

## Descarga de Datos

Debido al tamaño de los archivos, los datasets procesados y finales no se encuentran directamente en este repositorio. Para reproducir los experimentos, siga estos pasos:

1. **Datos Originales:** Descargue los datasets CICIDS2017, Darknet y UNSW-NB15 de sus fuentes oficiales y colóquelos en `data/raw/`.
2. **Datos Finales (Splits):** Los archivos `.npy` necesarios para el entrenamiento inmediato pueden ser descargados desde el siguiente enlace:
   - [(https://drive.google.com/drive/folders/1rMrunG7TCJuxXgy-Yn6GJ_4IdAychIyz?usp=sharing)]
3. **Ubicación:** Coloque los archivos descargados (`X_train_scaled.npy`, `y_train_ataque.npy`, etc.) en la carpeta `data/final/`.

## Requisitos

Para instalar las dependencias necesarias, ejecuta:

```bash
python -m venv -venv
pip install -r requirements.txt
```

## Notas sobre los Datos

Debido al gran tamaño de los datasets originales (CICIDS2017, Darknet, UNSW-NB15), los archivos dentro de la carpeta `data/` están excluidos del repositorio de Git mediante el archivo `.gitignore`. Asegúrate de colocar los archivos correspondientes en `data/raw/` antes de ejecutar los scripts de procesamiento.
