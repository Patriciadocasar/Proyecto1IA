"""
train.py
--------
Canal de ENTRENAMIENTO. Ejecutar desde la raíz del proyecto:

    python -m src.train

Pasos:
  1. Cargar datos y separar en entrenamiento / validación / prueba (estratificado).
  2. Ajustar el preprocesador SOLO con entrenamiento (evita data leakage).
  3. Calcular pesos de clase (manejo del desbalance ~80/20).
  4. Entrenar el MLP con Early Stopping sobre val_loss.
  5. Calibrar las probabilidades (Platt) con el conjunto de validación.
  6. Evaluar en prueba: ROC-AUC, PR-AUC, Brier, ECE.
  7. Explicabilidad con SHAP.
  8. Guardar artefactos (modelo, preprocesador, calibrador) y reportes.
  9. Puntuar la cartera completa con la regla financiera.
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss,  # noqa: E402
                             classification_report, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402
from tensorflow import keras  # noqa: E402

from . import config  # noqa: E402
from .finance import (ParametrosFinancieros, evaluar_cartera,  # noqa: E402
                      matriz_decision, resumen_cartera, estrategia_ingenua)
from .model import build_mlp  # noqa: E402
from .predict import calibrate  # noqa: E402
from .preprocessing import (build_preprocessor, feature_names,  # noqa: E402
                            load_data, original_feature_of, to_matrix)


def expected_calibration_error(y, p, n_bins=10):
    """ECE: diferencia promedio entre la probabilidad predicha y la tasa real
    de fuga dentro de cada 'cajita' de probabilidad. 0 = calibración perfecta."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def main():
    keras.utils.set_random_seed(config.SEED)
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)

    # 1. Datos ---------------------------------------------------------------
    df = load_data()
    y = df[config.TARGET].to_numpy()
    print(f"[1] Datos: {df.shape[0]} clientes | tasa de fuga = {y.mean():.2%}")

    df_trainval, df_test = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED)
    val_rel = config.VAL_SIZE / (1 - config.TEST_SIZE)
    df_train, df_val = train_test_split(
        df_trainval, test_size=val_rel,
        stratify=df_trainval[config.TARGET], random_state=config.SEED)
    print(f"    train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")

    # 2. Preprocesamiento (fit solo en train) -------------------------------
    pre = build_preprocessor().fit(df_train[config.FEATURES])
    X_tr, X_va, X_te = (to_matrix(pre, d) for d in (df_train, df_val, df_test))
    y_tr, y_va, y_te = (d[config.TARGET].to_numpy() for d in (df_train, df_val, df_test))
    names = feature_names(pre)
    print(f"[2] Variables tras escalado + one-hot: {len(names)}")

    # 3. Desbalance ----------------------------------------------------------
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    class_weight = {0: float(w[0]), 1: float(w[1])}
    print(f"[3] Pesos de clase: {class_weight}")

    # 4. Entrenamiento -------------------------------------------------------
    model = build_mlp(X_tr.shape[1])
    model.summary()
    early = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=config.PATIENCE, restore_best_weights=True)
    hist = model.fit(
        X_tr, y_tr, validation_data=(X_va, y_va),
        epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
        class_weight=class_weight, callbacks=[early], verbose=2)
    best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
    print(f"[4] Early stopping: mejor época = {best_epoch} "
          f"de {len(hist.history['loss'])}")

    # 5. Calibración (Platt) con validación ---------------------------------
    raw_va = model.predict(X_va, verbose=0).ravel()
    logit_va = np.log(np.clip(raw_va, 1e-6, 1 - 1e-6) /
                      (1 - np.clip(raw_va, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    calibrator = LogisticRegression().fit(logit_va, y_va)

    # 6. Evaluación en prueba -----------------------------------------------
    raw_te = model.predict(X_te, verbose=0).ravel()
    p_te = calibrate(calibrator, raw_te)
    metrics = {
        "roc_auc": float(roc_auc_score(y_te, p_te)),
        "pr_auc": float(average_precision_score(y_te, p_te)),
        "brier_sin_calibrar": float(brier_score_loss(y_te, raw_te)),
        "brier_calibrado": float(brier_score_loss(y_te, p_te)),
        "ece_sin_calibrar": expected_calibration_error(y_te, raw_te),
        "ece_calibrado": expected_calibration_error(y_te, p_te),
        "tasa_fuga_real_test": float(y_te.mean()),
        "prob_media_predicha_test": float(p_te.mean()),
        "mejor_epoca": best_epoch,
        "epocas_entrenadas": len(hist.history["loss"]),
        "class_weight": class_weight,
        "n_train": len(df_train), "n_val": len(df_val), "n_test": len(df_test),
        "variables_excluidas": config.DROP_COLS,
        # referencia con umbral 0.5 (solo diagnóstico; la app NO lo usa)
        "reporte_umbral_05": classification_report(
            y_te, (raw_te >= 0.5).astype(int), output_dict=True),
        "matriz_confusion_umbral_05": confusion_matrix(
            y_te, (raw_te >= 0.5).astype(int)).tolist(),
    }
    print(f"[6] ROC-AUC={metrics['roc_auc']:.4f} | PR-AUC={metrics['pr_auc']:.4f} | "
          f"Brier={metrics['brier_calibrado']:.4f} | ECE={metrics['ece_calibrado']:.4f}")

    # Gráficas de diagnóstico ----------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    ax[0].plot(hist.history["loss"], label="train")
    ax[0].plot(hist.history["val_loss"], label="validación")
    ax[0].axvline(best_epoch - 1, ls="--", c="gray", label="mejor época")
    ax[0].set_title("Curva de pérdida"); ax[0].set_xlabel("Época"); ax[0].legend()
    fpr, tpr, _ = roc_curve(y_te, p_te)
    ax[1].plot(fpr, tpr); ax[1].plot([0, 1], [0, 1], "--", c="gray")
    ax[1].set_title(f"ROC (AUC={metrics['roc_auc']:.3f})")
    ax[1].set_xlabel("Falsos positivos"); ax[1].set_ylabel("Verdaderos positivos")
    pr, rc, _ = precision_recall_curve(y_te, p_te)
    ax[2].plot(rc, pr); ax[2].axhline(y_te.mean(), ls="--", c="gray")
    ax[2].set_title(f"Precision-Recall (AUC={metrics['pr_auc']:.3f})")
    ax[2].set_xlabel("Recall"); ax[2].set_ylabel("Precision")
    for probs, lab in ((raw_te, "sin calibrar"), (p_te, "calibrada")):
        fy, fx = calibration_curve(y_te, probs, n_bins=10)
        ax[3].plot(fx, fy, "o-", label=lab)
    ax[3].plot([0, 1], [0, 1], "--", c="gray", label="perfecta")
    ax[3].set_title("Curva de calibración"); ax[3].set_xlabel("Prob. predicha")
    ax[3].set_ylabel("Fuga real"); ax[3].legend()
    plt.tight_layout(); plt.savefig(config.REPORTS_DIR / "diagnostico_modelo.png", dpi=130)
    plt.close()

    # 7. Explicabilidad (SHAP) ----------------------------------------------
    import shap
    rng = np.random.default_rng(config.SEED)
    background = X_tr[rng.choice(len(X_tr), 100, replace=False)]
    sample = X_te[rng.choice(len(X_te), 400, replace=False)]

    def f(X):
        return calibrate(calibrator, model.predict(X, verbose=0, batch_size=2048).ravel())

    explainer = shap.PermutationExplainer(f, shap.maskers.Independent(background),
                                          seed=config.SEED)
    sv = explainer(sample, max_evals=2 * len(names) + 1, silent=True)
    sv.feature_names = names
    imp = (pd.DataFrame(np.abs(sv.values), columns=names).T
           .groupby([original_feature_of(c) for c in names]).sum().T.mean()
           .sort_values(ascending=False))
    imp.rename("mean_abs_shap").to_csv(config.REPORTS_DIR / "shap_importancia.csv")
    print("[7] Factores determinantes (SHAP):")
    print(imp.head(5).to_string())

    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("SHAP: impacto de cada variable en la probabilidad de fuga")
    plt.tight_layout(); plt.savefig(config.REPORTS_DIR / "shap_beeswarm.png", dpi=130)
    plt.close("all")
    plt.figure(figsize=(7, 4.5))
    imp.iloc[::-1].plot.barh(color="#4C72B0")
    plt.title("Importancia media |SHAP| por variable original")
    plt.tight_layout(); plt.savefig(config.REPORTS_DIR / "shap_importancia.png", dpi=130)
    plt.close()

    # 8. Guardar artefactos -------------------------------------------------
    model.save(config.MODEL_PATH)
    joblib.dump(pre, config.PREPROCESSOR_PATH)
    joblib.dump(calibrator, config.CALIBRATOR_PATH)
    np.save(config.SHAP_BACKGROUND_PATH, background)
    metrics["factores_shap"] = imp.to_dict()
    with open(config.METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    print(f"[8] Artefactos guardados en {config.MODELS_DIR}")

    # 9. Puntuar la cartera completa ----------------------------------------
    params = ParametrosFinancieros()
    prob_all = f(to_matrix(pre, df))
    ev = evaluar_cartera(df, prob_all, params)
    ev.to_csv(config.REPORTS_DIR / "cartera_puntuada.csv", index=False)
    resumen = resumen_cartera(ev)
    resumen["estrategia_ingenua_p05"] = estrategia_ingenua(ev, params)
    # La cartera completa incluye los 7000 clientes con los que se entrenó la
    # red (sus probabilidades son algo optimistas). Como control, el mismo
    # resumen solo con el conjunto de prueba, que la red nunca vio:
    ev_te = evaluar_cartera(df_test, p_te, params)
    resumen["solo_conjunto_prueba"] = resumen_cartera(ev_te)
    with open(config.REPORTS_DIR / "resumen_financiero.json", "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, indent=2, ensure_ascii=False)
    matriz_decision(ev).to_csv(config.REPORTS_DIR / "matriz_decision.csv", index=False)
    print("[9] Resumen financiero:")
    for k, v in resumen.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
