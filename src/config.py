"""
config.py
---------
Configuración central del proyecto: rutas, columnas del dataset e
hiperparámetros. Todo lo que "se puede cambiar" vive aquí, para no tener
números mágicos regados por el código.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Customer-Churn-Records.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODEL_PATH = MODELS_DIR / "deep_retention_mlp.keras"      # pesos de la red
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"    # scaler + one-hot
CALIBRATOR_PATH = MODELS_DIR / "calibrator.joblib"        # calibración Platt
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background.npy" # muestra base SHAP
METRICS_PATH = REPORTS_DIR / "metrics.json"

# ---------------------------------------------------------------------------
# Columnas
# ---------------------------------------------------------------------------
TARGET = "Exited"
ID_COL = "CustomerId"

# Columnas que NO entran al modelo:
#  - RowNumber, Surname: identificadores sin poder predictivo.
#  - Complain: FUGA DE DATOS (data leakage). En este dataset 2034 de 2038
#    clientes que se fueron tienen Complain=1 y casi ningún cliente que se
#    quedó la tiene. Es un dato que se registra junto con (o después de) la
#    salida, así que en producción no estaría disponible ANTES de que el
#    cliente se vaya. Si la usáramos, el modelo "haría trampa" (AUC ~0.999).
DROP_COLS = ["RowNumber", "Surname", "Complain"]

NUM_COLS = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "Satisfaction Score", "Point Earned",
]
CAT_COLS = ["Geography", "Gender", "Card Type"]
FEATURES = NUM_COLS + CAT_COLS

# Valores válidos (se usan para validar datos nuevos en la app)
CATEGORIES = {
    "Geography": ["France", "Germany", "Spain"],
    "Gender": ["Female", "Male"],
    "Card Type": ["SILVER", "GOLD", "PLATINUM", "DIAMOND"],
}
NUM_RANGES = {  # (mínimo, máximo) aceptados
    "CreditScore": (300, 900),
    "Age": (18, 100),
    "Tenure": (0, 10),
    "Balance": (0.0, 1_000_000.0),
    "NumOfProducts": (1, 4),
    "HasCrCard": (0, 1),
    "IsActiveMember": (0, 1),
    "EstimatedSalary": (0.0, 1_000_000.0),
    "Satisfaction Score": (1, 5),
    "Point Earned": (0, 1000),
}

# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
EPOCHS = 300
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE = 15
