"""
predict.py
----------
Canal de INFERENCIA (separado del entrenamiento). Carga los artefactos
guardados en models/ y calcula la probabilidad de fuga de cualquier cliente.
La app de Streamlit solo usa este módulo: nunca vuelve a entrenar.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tensorflow import keras  # noqa: E402

from . import config  # noqa: E402
from .preprocessing import (feature_names, original_feature_of,  # noqa: E402
                            to_matrix, validate_input)


class ChurnPredictor:
    def __init__(self):
        missing = [p for p in (config.MODEL_PATH, config.PREPROCESSOR_PATH,
                               config.CALIBRATOR_PATH) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"No se encontraron los artefactos del modelo: {missing}. "
                "Ejecuta primero: python -m src.train")
        self.model = keras.models.load_model(config.MODEL_PATH)
        self.preprocessor = joblib.load(config.PREPROCESSOR_PATH)
        self.calibrator = joblib.load(config.CALIBRATOR_PATH)
        self.feature_names = feature_names(self.preprocessor)
        self._explainer = None

    # ---------------------------------------------------------------- core
    def _raw_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0, batch_size=1024).ravel()

    def _calibrate(self, raw: np.ndarray) -> np.ndarray:
        return calibrate(self.calibrator, raw)

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        return self._calibrate(self._raw_proba(X))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        errors = validate_input(df)
        if errors:
            raise ValueError("; ".join(errors))
        return self.predict_matrix(to_matrix(self.preprocessor, df))

    # ------------------------------------------------------ explicabilidad
    def explain(self, df: pd.DataFrame, max_evals: int = 200) -> pd.DataFrame:
        """Valores SHAP por cliente, agrupados a la variable original
        (Geography_Germany + Geography_Spain + ... -> Geography)."""
        import shap
        if self._explainer is None:
            background = np.load(config.SHAP_BACKGROUND_PATH)
            masker = shap.maskers.Independent(background)
            self._explainer = shap.PermutationExplainer(
                self.predict_matrix, masker, seed=config.SEED)
        X = to_matrix(self.preprocessor, df)
        sv = self._explainer(X, max_evals=max_evals, silent=True).values
        contrib = pd.DataFrame(sv, columns=self.feature_names)
        grouped = contrib.T.groupby(
            [original_feature_of(c) for c in contrib.columns]).sum().T
        return grouped


def calibrate(calibrator, raw: np.ndarray) -> np.ndarray:
    """Calibración Platt: regresión logística sobre el logit de la red.
    Corrige que la ponderación de clases infla las probabilidades."""
    eps = 1e-6
    raw = np.clip(raw, eps, 1 - eps)
    logit = np.log(raw / (1 - raw)).reshape(-1, 1)
    return calibrator.predict_proba(logit)[:, 1]
