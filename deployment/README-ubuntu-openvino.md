# Versión de laboratorio: Ubuntu + OpenVINO

Rama: lab/ubuntu-openvino
Fecha del respaldo: 2026-11-05
VM: vm-modelo

## Entorno utilizado

- Ubuntu 24.04 LTS.
- Python 3.12.15.
- Entorno virtual local: .venv_openvino312.
- Keras 3.14.1 con backend OpenVINO.
- OpenVINO 2025.4.1.
- NumPy 2.3.5.
- pandas 3.1.0.
- scikit-learn 1.8.0.
- joblib 1.5.3.

Las dependencias exactas del entorno funcional se registran en
requirements-ubuntu-openvino.txt. La comprobación pip check no encontró
incompatibilidades.

Esta adaptación se utiliza porque la CPU expuesta a la VM del laboratorio
no presenta AVX y TensorFlow produjo un error de instrucción ilegal.

## Arranque manual en la VM existente

Desde la raíz del repositorio:

```bash
source .venv_openvino312/bin/activate
export KERAS_BACKEND=openvino
python src/core/server_nids_openvino.py
```

Motor asociado: src/core/inference_engine_openvino.py.
Servidor local utilizado en las pruebas: 127.0.0.1:9999.

## Recreación del entorno

Con Python 3.12 disponible:

```bash
python3.12 -m venv .venv_openvino312
.venv_openvino312/bin/python -m pip install -r requirements-ubuntu-openvino.txt
.venv_openvino312/bin/python -m pip check
```

La instalación en una VM nueva requiere validación funcional y los mismos
artefactos de modelos, escaladores y encoders.

## Estado experimental

- Carril A: clasificación de tráfico normal y categorías de ataque.
- Carril B: versión anterior basada en etiquetas VPN/Tor frente a NonVPN/Non-Tor.
- Las pruebas de laboratorio mostraron que el carril B no identifica
  adecuadamente HTTPS/TLS como cifrado.
- Se decidió reentrenar el carril B con CICIDS reetiquetado.
- Esta rama conserva el estado previo a incorporar ese nuevo modelo.

La ejecución del pipeline no demuestra por sí sola la validez de las
predicciones ni constituye la evaluación experimental final.

## Despliegue y archivos externos

Los archivos de servicio, contratos, dependencias anteriores y pruebas
se conservan como parte del respaldo. El servicio nids-inference.service
debe revisarse antes de habilitarlo para confirmar que utiliza el entorno
.venv_openvino312 y el servidor OpenVINO.

Los entornos virtuales, capturas, datasets y modelos excluidos por
.gitignore no se incluyen en este respaldo de Git. Los artefactos externos
deben conservarse por separado para reproducir la ejecución.
