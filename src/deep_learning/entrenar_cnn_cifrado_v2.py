"""CNN del carril B. Ejecutar despues de generar las matrices escaladas."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from sklearn.manifold import TSNE

BASE = Path(__file__).resolve().parents[2]
MATRICES = BASE / "data/cifrado_cicids2017_v2/matrices"
MODELO = BASE / "models/deep_learning/best_cnn_encryption_cicids2017_v2.keras"
FIG = BASE / "reports/figures/cifrado_cicids2017_v2"
CLASES = ["No Cifrado", "Cifrado"]


def cargar(split):
    x = np.load(MATRICES / f"X_{split}_encryption.npy", allow_pickle=False)
    y = np.load(MATRICES / f"y_{split}_encryption.npy", allow_pickle=False)
    if x.shape != (len(y), 62) or y.ndim != 1 or not np.isfinite(x).all():
        raise ValueError(f"Matriz o dimensiones invalidas en {split}")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Etiquetas inesperadas en {split}")
    return x[..., None], y


def construir():
    entrada = layers.Input(shape=(62, 1), name="Input_Descriptores_Crudos")
    x = layers.Conv1D(32, 3, activation="relu", padding="same", name="Conv1D_Cifrado_1")(entrada)
    x = layers.BatchNormalization(name="BatchNorm_Cifrado_1")(x)
    x = layers.MaxPooling1D(2, name="MaxPool_Cifrado_1")(x)
    x = layers.Dropout(0.3, seed=42, name="Dropout_Cifrado_1")(x)
    x = layers.Conv1D(64, 3, activation="relu", padding="same", name="Conv1D_Cifrado_2")(x)
    x = layers.BatchNormalization(name="BatchNorm_Cifrado_2")(x)
    x = layers.MaxPooling1D(2, name="MaxPool_Cifrado_2")(x)
    x = layers.Flatten(name="Flatten_Cifrado")(x)
    x = layers.Dropout(0.4, seed=42, name="Dropout_Pre_Bottleneck")(x)
    x = layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(.001),
                     name="Embedding_Encryption")(x)
    x = layers.Dropout(.2, seed=42, name="Dropout_Post_Bottleneck")(x)
    salida = layers.Dense(1, activation="sigmoid", name="Salida_Clasificador_Binario")(x)
    modelo = Model(entrada, salida, name="CNN_Encryption_Full")
    modelo.compile(optimizer=tf.keras.optimizers.Adam(.001), loss="binary_crossentropy",
                   metrics=["accuracy"])
    return modelo


def main():
    tf.keras.utils.set_random_seed(42)
    if MODELO.exists():
        raise FileExistsError(f"Ya existe el checkpoint. Archiva la corrida antes de repetir: {MODELO}")
    MODELO.parent.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    x, y = cargar("train")
    xv, yv = cargar("val")
    modelo = construir()
    modelo.summary()
    h = modelo.fit(x, y, validation_data=(xv, yv), batch_size=128, epochs=20,
                   callbacks=[tf.keras.callbacks.ModelCheckpoint(str(MODELO), monitor="val_loss",
                              save_best_only=True, verbose=1),
                              tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                              restore_best_weights=True, verbose=1)]).history
    (FIG / "historial_cnn.json").write_text(json.dumps(h, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, titulo in zip(axes, ["loss", "accuracy"], ["Pérdida", "Exactitud"]):
        ax.plot(range(1, len(h[key])+1), h[key], label="Entrenamiento")
        ax.plot(range(1, len(h[key])+1), h["val_"+key], label="Validación")
        ax.set(xlabel="Época", ylabel=titulo)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "curvas_cnn.png", dpi=300)
    plt.close(fig)

    mejor = tf.keras.models.load_model(MODELO, compile=False)
    extractor = Model(mejor.inputs, mejor.get_layer("Embedding_Encryption").output)
    ids = np.random.default_rng(42).choice(len(yv), min(4000, len(yv)), replace=False)
    z = extractor.predict(xv[ids], batch_size=512, verbose=0)
    coords = TSNE(n_components=2, perplexity=min(30, len(ids)-1), random_state=42,
                  init="pca", learning_rate="auto").fit_transform(z)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, nombre in enumerate(CLASES):
        subset = coords[yv[ids] == i]
        ax.scatter(subset[:, 0], subset[:, 1], s=8, alpha=.6, label=nombre)
    ax.set(title="Embeddings de validación", xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "tsne_validacion.png", dpi=300)
    plt.close(fig)
    print("Mejor checkpoint:", MODELO)


if __name__ == "__main__":
    main()
