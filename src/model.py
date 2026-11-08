"""
model.py
--------
Arquitectura de la red neuronal (Perceptrón Multicapa) siguiendo la guía:
  Capa 1: 128 neuronas, ReLU + BatchNormalization + Dropout(0.3)
  Capa 2:  64 neuronas, ReLU + Dropout(0.2)
  Salida:   1 neurona, Sigmoide -> probabilidad de fuga entre 0 y 1
Optimizador Adam, pérdida Binary Cross-Entropy.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from tensorflow import keras  # noqa: E402

from . import config  # noqa: E402


def build_mlp(n_features: int) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="deep_retention_mlp",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="roc_auc"),
                 keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model
