"""
preprocessing.py
----------------
Carga del dataset y construcción del preprocesador:
  - StandardScaler para variables numéricas (las redes neuronales son
    sensibles a magnitudes: Balance ~100.000 vs. Tenure ~5).
  - OneHotEncoder para variables categóricas (país, género, tipo de tarjeta).

IMPORTANTE: el preprocesador se ajusta (fit) SOLO con el conjunto de
entrenamiento, para no filtrar información de validación/prueba.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    """Lee el CSV y elimina columnas que no deben usarse."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns])
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             config.CAT_COLS),
        ]
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Nombres de columnas después de transformar (ej. Geography_Germany)."""
    names = list(config.NUM_COLS)
    ohe = preprocessor.named_transformers_["cat"]
    for col, cats in zip(config.CAT_COLS, ohe.categories_):
        names += [f"{col}_{c}" for c in cats]
    return names


def original_feature_of(transformed_name: str) -> str:
    """Geography_Germany -> Geography (para agrupar importancias SHAP)."""
    for col in config.CAT_COLS:
        if transformed_name.startswith(col + "_"):
            return col
    return transformed_name


def validate_input(df: pd.DataFrame) -> list[str]:
    """Revisa que un cliente nuevo tenga columnas y valores válidos.
    Devuelve una lista de errores (vacía si todo está bien)."""
    errors = []
    missing = [c for c in config.FEATURES if c not in df.columns]
    if missing:
        return [f"Faltan columnas: {missing}"]
    for col, (lo, hi) in config.NUM_RANGES.items():
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any():
            errors.append(f"'{col}' tiene valores no numéricos o vacíos.")
        elif ((vals < lo) | (vals > hi)).any():
            errors.append(f"'{col}' fuera de rango [{lo}, {hi}].")
    for col, allowed in config.CATEGORIES.items():
        bad = set(df[col].astype(str)) - set(allowed)
        if bad:
            errors.append(f"'{col}' tiene valores no válidos: {sorted(bad)}.")
    return errors


def to_matrix(preprocessor, df: pd.DataFrame) -> np.ndarray:
    return preprocessor.transform(df[config.FEATURES]).astype("float32")
