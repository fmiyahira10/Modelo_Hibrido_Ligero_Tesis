from pathlib import Path
import re
import os
import json
import numpy as np
import pandas as pd
import joblib

BASE = Path(__file__).resolve().parent
DATOS = Path.home() / "tesis" / "flujos_csv"
CSV = Path(os.environ.get("TESIS_CSV", str(DATOS / "flujos_lab.csv")))

# Leer características separadas por carril
archivo = BASE / "caracteristicas_modelos.txt"
contenido = archivo.read_text(encoding="utf-8-sig")

grupos = [[], []]
carril = None

for linea in contenido.splitlines():
    linea = linea.strip()
    titulo = linea.upper()

    # Identificar las secciones del archivo
    if "MODELO DE VECTOR DE ATAQUE" in titulo:
        carril = 0
        continue

    if (
        "MODELO DE ENCRIPTACIÓN" in titulo
        or "MODELO DE ENCRIPTACION" in titulo
    ):
        carril = 1
        continue

    if carril is None:
        continue

    # Identificar las características numeradas
    m = re.match(r"^(\d+)\.\s+(.+)$", linea)

    if m:
        numero = int(m.group(1))
        nombre = m.group(2).strip()

        # Verificar numeración consecutiva
        if numero == len(grupos[carril]) + 1:
            grupos[carril].append(nombre)

A, B = grupos

print(f"Carril A: {len(A)} características encontradas")
print(f"Carril B: {len(B)} características encontradas")

if len(A) != 13 or len(B) != 62:
    raise RuntimeError(
        "Error al leer las características. "
        f"Se encontraron A={len(A)} y B={len(B)}"
    )

print("[OK] Listas de características cargadas correctamente")

# Equivalencias entre carril A y CICFlowMeter
MAP = dict(zip(A, [
    "Flow Duration",
    "Total Fwd Packet",
    "Total Bwd packets",
    "Total Length of Fwd Packet",
    "Total Length of Bwd Packet",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "FWD Init Win Bytes",
    "Bwd Init Win Bytes",
    "Flow Packets/s",
    "Flow Bytes/s"
]))

df = pd.read_csv(CSV)
df.columns = df.columns.str.strip()

faltantes = set(B) | set(MAP.values())
faltantes -= set(df.columns)

if faltantes:
    raise RuntimeError(f"Faltan columnas: {sorted(faltantes)}")

# Dos vectores procedentes de las mismas filas
xa = pd.DataFrame({
    nombre: pd.to_numeric(df[MAP[nombre]], errors="coerce")
    for nombre in A
})
xb = df[B].apply(pd.to_numeric, errors="coerce")

# Verificar compatibilidad de orden con los escaladores
for carril, matriz, archivo in [
    ("A", xa, "robust_scaler_attack.pkl"),
    ("B", xb, "robust_scaler_encryption.pkl")
]:
    ruta = BASE / "models/scalers_encoders" / archivo
    if ruta.exists():
        scaler = joblib.load(ruta)
        nombres = getattr(scaler, "feature_names_in_", None)
        if nombres is None:
            print(f"[AVISO] Carril {carril}: escalador sin nombres")
        elif list(nombres) != list(matriz.columns):
            raise RuntimeError(
                f"Carril {carril}: el orden difiere del entrenamiento.\n"
                f"Esperado: {list(nombres)}\n"
                f"Recibido: {list(matriz.columns)}"
            )
        else:
            print(f"[OK] Orden del carril {carril} validado")
    else:
        print(f"[AVISO] Falta el escalador del carril {carril}")

# Rechazar filas con valores inválidos
validos = (
    np.isfinite(xa.to_numpy(dtype=float)).all(axis=1) &
    np.isfinite(xb.to_numpy(dtype=float)).all(axis=1)
)

indices = np.flatnonzero(validos)

print(f"Flujos originales: {len(df)}")
print(f"Flujos válidos: {len(indices)}")
print(f"Flujos rechazados: {len(df) - len(indices)}")
print(f"Dimensiones: A={xa.shape[1]}, B={xb.shape[1]}")

# Exportar hasta 10 flujos de prueba, sin ejecutar inferencia
salida = Path(os.environ.get("TESIS_VECTORES", str(DATOS / "vectores_prueba.jsonl")))
salida.parent.mkdir(parents=True, exist_ok=True)

with salida.open("w", encoding="utf-8") as f:
    for i in indices[:10]:
        registro = {
            "id_flujo": f"fila-{i+2}",
            "src_ip": str(df.iloc[i]["Src IP"]),
            "dst_ip": str(df.iloc[i]["Dst IP"]),
            "features_attack": xa.iloc[i].astype(float).tolist(),
            "features_encryption": xb.iloc[i].astype(float).tolist()
        }
        f.write(json.dumps(registro, allow_nan=False) + "\n")

print(f"Vectores exportados: {salida}")
