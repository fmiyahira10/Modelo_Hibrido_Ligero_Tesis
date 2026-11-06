"""Clasificador sobre embeddings de la mejor CNN v2. Clase positiva: Cifrado."""
from pathlib import Path
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            f1_score, fbeta_score, matthews_corrcoef, precision_score, recall_score)
from tensorflow.keras.models import Model, load_model

BASE = Path(__file__).resolve().parents[2]
DATOS = BASE / "data/cifrado_cicids2017_v2"
MATRICES = DATOS / "matrices"
CNN = BASE / "models/deep_learning/best_cnn_encryption_cicids2017_v2.keras"
RF = BASE / "models/classifiers/encryption_classifier_rf_cicids2017_v2.pkl"
INFORMES = BASE / "reports/figures/cifrado_cicids2017_v2"
CLASES = ["No Cifrado", "Cifrado"]


def embeddings(extractor, split):
    x = np.load(MATRICES / f"X_{split}_encryption.npy", mmap_mode="r")
    y = np.load(MATRICES / f"y_{split}_encryption.npy")
    if x.shape != (len(y), 62) or y.ndim != 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Dimensiones o etiquetas inesperadas en {split}")
    z = extractor.predict(x[..., None], batch_size=512, verbose=1)
    if z.shape != (len(y), 16) or not np.isfinite(z).all():
        raise ValueError(f"Embeddings invalidos en {split}")
    return z, y


def evaluar(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0), F1=f1_score(y, pred, zero_division=0),
                F2=fbeta_score(y, pred, beta=2, zero_division=0), MCC=matthews_corrcoef(y, pred),
                FPR=fp/(fp+tn), TN=tn, FP=fp, FN=fn, TP=tp,
                por_clase=classification_report(y, pred, labels=[0, 1], target_names=CLASES,
                                                zero_division=0, output_dict=True))


def main():
    if RF.exists():
        raise FileExistsError(f"Archiva el RF v2 antes de repetir la corrida: {RF}")
    RF.parent.mkdir(parents=True, exist_ok=True)
    INFORMES.mkdir(parents=True, exist_ok=True)
    cnn = load_model(CNN, compile=False)
    extractor = Model(cnn.inputs, cnn.get_layer("Embedding_Encryption").output)
    z, y = embeddings(extractor, "train")
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=10,
                                min_samples_leaf=4, max_features="sqrt", random_state=42,
                                n_jobs=-1, verbose=1)
    rf.fit(z, y)
    resultados = {"train": evaluar(y, rf.predict(z))}
    for split in ["val", "test"]:
        z, y = embeddings(extractor, split)
        pred = rf.predict(z)
        resultados[split] = evaluar(y, pred)
        if split == "test":
            meta = pd.read_csv(DATOS / "prueba_metadatos.csv", usecols=["dia", "protocolo", "etiqueta"])
            esperadas = meta["etiqueta"].map({"No Cifrado": 0, "Cifrado": 1}).to_numpy()
            if not np.array_equal(y, esperadas):
                raise ValueError("Las etiquetas de prueba y los metadatos no coinciden en orden.")
            meta["acierto"] = y == pred
            for campos, nombre in [(["protocolo"], "protocolo"), (["dia", "protocolo"], "dia_protocolo")]:
                tabla = meta.groupby(campos)["acierto"].agg(flujos="size", aciertos="sum", tasa_acierto="mean")
                tabla.to_csv(INFORMES / f"prueba_por_{nombre}.csv")
                print(tabla)
    (INFORMES / "metricas_rf.json").write_text(json.dumps(resultados, indent=2), encoding="utf-8")
    metricas = ["accuracy", "precision", "recall", "F1", "F2", "MCC"]
    tabla = pd.DataFrame(resultados).loc[metricas].astype(float)
    ax = tabla.plot.bar(figsize=(10, 5), rot=0)
    ax.set_ylabel("Valor de la métrica")
    ax.figure.tight_layout()
    ax.figure.savefig(INFORMES / "comparacion_rf.png", dpi=300)
    plt.close(ax.figure)
    joblib.dump(rf, RF)
    print(tabla.to_string())
    print("Clasificador:", RF)
    print("Resultados:", INFORMES)


if __name__ == "__main__":
    main()
