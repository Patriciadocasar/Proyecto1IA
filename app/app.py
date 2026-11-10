"""
app.py — Aplicación interactiva "Deep Retention"
Ejecutar desde la raíz del proyecto:

    streamlit run app/app.py

Vistas:
   Buscador por cliente   -> puntaje de la red, impacto monetario y decisión
   Cliente nuevo          -> mismo análisis para datos ingresados a mano
   Panel de cartera       -> balance en riesgo, inversión, dinero a rescatar
   Simulador             -> ajustar costos/efectividad y ver el impacto
   Modelo                 -> métricas técnicas y explicabilidad (SHAP)
"""
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.finance import (ParametrosFinancieros, estrategia_ingenua,  # noqa: E402
                         evaluar_cartera, ganancia_neta_esperada,
                         resumen_cartera)
from src.predict import ChurnPredictor  # noqa: E402
from src.preprocessing import load_data, validate_input  # noqa: E402

st.set_page_config(page_title="Deep Retention", page_icon="🏦", layout="wide")

COLORES = {
    "Retención Prioritaria": "#2E7D32",
    "Retención Automatizada": "#66BB6A",
    "Salida Permitida": "#E53935",
    "Sin Acción Requerida": "#90A4AE",
}


# ---------------------------------------------------------------------------
# Carga (cacheada: el modelo se carga una sola vez)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando red neuronal...")
def get_predictor():
    return ChurnPredictor()


@st.cache_data(show_spinner="Calculando probabilidades de la cartera...")
def get_portfolio():
    df = load_data()
    df["prob_fuga"] = get_predictor().predict_proba(df)
    return df


def money(x):
    return f"${x:,.0f}"


# ---------------------------------------------------------------------------
# Barra lateral: parámetros financieros (alimentan TODAS las vistas)
# ---------------------------------------------------------------------------
def sidebar_params() -> ParametrosFinancieros:
    st.sidebar.title("🏦 Deep Retention")
    st.sidebar.caption("Parámetros de negocio (simulador global)")
    p = ParametrosFinancieros()

    with st.sidebar.expander("Margen del cliente", expanded=False):
        p.spread_neto = st.slider("Spread neto sobre saldo (%)", 0.0, 5.0,
                                  p.spread_neto * 100, 0.1) / 100
        p.comision_por_producto = st.number_input(
            "Comisión anual por producto ($)", 0.0, 1000.0, p.comision_por_producto, 5.0)
        p.ingreso_tarjeta = st.number_input(
            "Ingreso anual por tarjeta ($)", 0.0, 1000.0, p.ingreso_tarjeta, 5.0)
        p.costo_operativo = st.number_input(
            "Costo operativo anual por cuenta ($)", 0.0, 2000.0, p.costo_operativo, 10.0)
        p.horizonte_anios = st.slider("Horizonte de valor (años)", 0.5, 5.0,
                                      p.horizonte_anios, 0.5)

    with st.sidebar.expander("Catálogo de incentivos", expanded=True):
        for key, a in p.catalogo.items():
            st.markdown(f"**{a.nombre}**  \n_niveles: {', '.join(a.niveles)}_")
            c1, c2 = st.columns(2)
            a.costo = c1.number_input("Costo ($)", 0.0, 5000.0, a.costo, 5.0,
                                      key=f"c_{key}")
            a.efectividad = c2.slider("Efectividad", 0.0, 1.0, a.efectividad,
                                      0.05, key=f"e_{key}")

    with st.sidebar.expander("Segmentación", expanded=False):
        p.umbral_patrimonial = st.number_input(
            "Saldo mínimo nivel Patrimonial ($)", 0.0, 500_000.0,
            p.umbral_patrimonial, 5_000.0)
        p.umbral_medio = st.number_input(
            "Saldo mínimo nivel Medio ($)", 0.0, 500_000.0, p.umbral_medio, 5_000.0)
        p.umbral_estable = st.slider("Prob. bajo la cual es 'Estable'", 0.0, 1.0,
                                     p.umbral_estable, 0.05)
    st.sidebar.info("Regla: se interviene solo si  \n"
                    "**P(fuga) × Efectividad × Margen − Costo > 0**  \n"
                    f"y P(fuga) ≥ {p.umbral_estable:.2f} (debajo es 'Estable').")
    return p


# ---------------------------------------------------------------------------
# Ficha de un cliente (usada por Buscador y Cliente nuevo)
# ---------------------------------------------------------------------------
def gauge(prob):
    return go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Probabilidad de fuga (red neuronal)"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#263238"},
               "steps": [{"range": [0, 30], "color": "#C8E6C9"},
                         {"range": [30, 65], "color": "#FFF59D"},
                         {"range": [65, 100], "color": "#FFCDD2"}]},
    )).update_layout(height=260, margin=dict(t=60, b=10, l=30, r=30))


def ficha_cliente(row_df: pd.DataFrame, prob: float, p: ParametrosFinancieros):
    ev = evaluar_cartera(row_df, np.array([prob]), p).iloc[0]
    color = COLORES.get(ev["decision"], "#555")

    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.plotly_chart(gauge(prob), width="stretch")
    with c2:
        st.markdown(
            f"<div style='background:{color};padding:18px;border-radius:12px;color:white'>"
            f"<div style='font-size:14px;opacity:.85'>Decisión del sistema</div>"
            f"<div style='font-size:30px;font-weight:700'>{ev['decision']}</div>"
            f"<div style='font-size:15px'>{ev['segmento']} · {ev['motivo']}</div>"
            f"</div>", unsafe_allow_html=True)
        st.write("")
        m1, m2, m3 = st.columns(3)
        m1.metric("Margen del cliente", money(ev["margen_cliente"]))
        m2.metric("Nivel patrimonial", ev["nivel_patrimonial"])
        m3.metric("Saldo", money(row_df["Balance"].iloc[0]))
        m4, m5, m6 = st.columns(3)
        m4.metric("Incentivo asignado", ev["incentivo"])
        m5.metric("Costo", money(ev["costo_incentivo"]))
        m6.metric("Ganancia neta esperada", money(ev["ganancia_mejor_accion"]))

    # Tabla con TODAS las acciones evaluadas -> transparencia de la decisión
    st.markdown("##### Cálculo de la regla de retención para cada incentivo")
    filas = []
    for a in p.catalogo.values():
        eleg = ev["nivel_patrimonial"] in a.niveles
        g = ganancia_neta_esperada(prob, a.efectividad, ev["margen_cliente"], a.costo)
        filas.append({
            "Incentivo": a.nombre, "Elegible": "✅" if eleg else "—",
            "Fórmula": f"{prob:.3f} × {a.efectividad:.2f} × {ev['margen_cliente']:,.0f} − {a.costo:,.0f}",
            "Ganancia neta esperada": g if eleg else np.nan,
            "¿Se ejecuta?": "Sí" if a.nombre == ev["incentivo"] else "No",
        })
    st.dataframe(pd.DataFrame(filas).style.format(
        {"Ganancia neta esperada": "${:,.2f}"}, na_rep="no elegible"),
        hide_index=True, width="stretch")

    # Explicación SHAP del cliente
    with st.expander("¿Por qué la red da esta probabilidad? (SHAP)", expanded=True):
        with st.spinner("Calculando contribuciones SHAP..."):
            shap_row = get_predictor().explain(row_df).iloc[0].sort_values()
        fig = px.bar(
            x=shap_row.values * 100, y=shap_row.index, orientation="h",
            color=np.where(shap_row.values > 0, "Aumenta la fuga", "Reduce la fuga"),
            color_discrete_map={"Aumenta la fuga": "#E53935", "Reduce la fuga": "#43A047"},
            labels={"x": "Contribución (puntos porcentuales)", "y": "", "color": ""})
        fig.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Barras rojas empujan la probabilidad hacia arriba; verdes hacia abajo, "
                   "respecto al cliente promedio.")


# ---------------------------------------------------------------------------
# Vistas
# ---------------------------------------------------------------------------
def vista_buscador(df, p):
    st.header("Buscador por cliente")
    c1, c2 = st.columns([1, 2])
    modo = c1.radio("Buscar por", ["ID de cliente", "Apellido"], horizontal=True)
    if modo == "ID de cliente":
        cid = c2.selectbox("CustomerId", df[config.ID_COL].tolist(),
                           index=0, placeholder="Escribe un ID...")
    else:
        raw = pd.read_csv(config.DATA_PATH, usecols=[config.ID_COL, "Surname"])
        apellido = c2.text_input("Apellido", "Hargrave")
        hits = raw[raw["Surname"].str.contains(apellido, case=False, na=False)]
        if hits.empty:
            st.warning("No se encontraron clientes con ese apellido.")
            return
        cid = st.selectbox(f"{len(hits)} coincidencias",
                           hits[config.ID_COL].tolist(),
                           format_func=lambda x: f"{x} – {hits.set_index(config.ID_COL).loc[x, 'Surname']}")
    row = df[df[config.ID_COL] == cid]
    with st.expander("Datos del cliente"):
        st.dataframe(row[config.FEATURES].T.rename(columns=lambda _: "valor").astype(str),
                     width="stretch")
    ficha_cliente(row[config.FEATURES + [config.ID_COL]], float(row["prob_fuga"].iloc[0]), p)


def vista_nuevo(p):
    st.header("Evaluar un cliente nuevo")
    st.caption("Ingresa los datos de un cliente que no está en la base para "
               "obtener su puntaje y la recomendación.")
    with st.form("nuevo"):
        c1, c2, c3 = st.columns(3)
        datos = {
            "CreditScore": c1.number_input("Score crediticio", 300, 900, 650),
            "Age": c1.number_input("Edad", 18, 100, 45),
            "Tenure": c1.number_input("Antigüedad (años)", 0, 10, 3),
            "Balance": c2.number_input("Saldo ($)", 0.0, 1_000_000.0, 125_000.0, 1000.0),
            "NumOfProducts": c2.selectbox("N.º de productos", [1, 2, 3, 4]),
            "EstimatedSalary": c2.number_input("Salario estimado ($)", 0.0, 1_000_000.0,
                                               90_000.0, 1000.0),
            "HasCrCard": int(c3.checkbox("Tiene tarjeta de crédito", True)),
            "IsActiveMember": int(c3.checkbox("Es miembro activo", False)),
            "Satisfaction Score": c3.slider("Satisfacción (1-5)", 1, 5, 3),
            "Point Earned": c3.number_input("Puntos acumulados", 0, 1000, 500),
            "Geography": c1.selectbox("País", config.CATEGORIES["Geography"], index=1),
            "Gender": c2.selectbox("Género", config.CATEGORIES["Gender"]),
            "Card Type": c3.selectbox("Tipo de tarjeta", config.CATEGORIES["Card Type"]),
        }
        enviado = st.form_submit_button("Evaluar cliente", type="primary")
    if enviado:
        row = pd.DataFrame([datos])
        errores = validate_input(row)
        if errores:
            for e in errores:
                st.error(e)
            return
        prob = float(get_predictor().predict_proba(row)[0])
        ficha_cliente(row, prob, p)

    st.divider()
    st.subheader("Carga masiva (CSV)")
    st.caption(f"Columnas requeridas: {', '.join(config.FEATURES)}")
    up = st.file_uploader("Sube un CSV con clientes nuevos", type="csv")
    if up is not None:
        try:
            nuevos = pd.read_csv(up)
            errores = validate_input(nuevos)
            if errores:
                for e in errores:
                    st.error(e)
                return
            prob = get_predictor().predict_proba(nuevos)
            ev = evaluar_cartera(nuevos, prob, p)
            st.dataframe(ev, width="stretch")
            st.download_button("Descargar resultados", ev.to_csv(index=False),
                               "clientes_evaluados.csv", "text/csv")
        except Exception as exc:  # archivo corrupto, separador raro, etc.
            st.error(f"No se pudo procesar el archivo: {exc}")


def vista_panel(df, p):
    st.header("Panel de cartera")
    ev = evaluar_cartera(df.drop(columns="prob_fuga"), df["prob_fuga"].to_numpy(), p)

    f1, f2, f3 = st.columns(3)
    paises = f1.multiselect("País", config.CATEGORIES["Geography"],
                            default=config.CATEGORIES["Geography"])
    niveles = f2.multiselect("Nivel patrimonial", ["Patrimonial", "Medio", "Básico"],
                             default=["Patrimonial", "Medio", "Básico"])
    umbral = f3.slider("Umbral de 'riesgo crítico' (solo para reportar)", 0.3, 0.9, 0.5, 0.05)
    ev = ev[ev["Geography"].isin(paises) & ev["nivel_patrimonial"].isin(niveles)]
    if ev.empty:
        st.warning("No hay clientes con esos filtros.")
        return
    r = resumen_cartera(ev, umbral)

    k = st.columns(5)
    k[0].metric("Clientes analizados", f"{r['clientes_analizados']:,}")
    k[1].metric("Capital en riesgo crítico", money(r["capital_riesgo_critico"]),
                f"{r['clientes_riesgo_critico']:,} clientes", delta_color="off")
    k[2].metric("Capital rescatable", money(r["capital_rescatable"]),
                f"{r['clientes_a_intervenir']:,} a intervenir", delta_color="off")
    k[3].metric("Inversión en campañas", money(r["inversion_total"]))
    k[4].metric("Ganancia neta / ROI", money(r["ganancia_neta_total"]),
                f"ROI {r['roi']:.0%}")

    c1, c2 = st.columns(2)
    seg = ev.groupby("decision").agg(clientes=("prob_fuga", "size"),
                                     saldo=("Balance", "sum")).reset_index()
    c1.plotly_chart(px.pie(seg, names="decision", values="clientes", hole=.45,
                           color="decision", color_discrete_map=COLORES,
                           title="Clientes por decisión"), width="stretch")
    c2.plotly_chart(px.bar(seg, x="decision", y="saldo", color="decision",
                           color_discrete_map=COLORES, title="Saldo total por decisión",
                           labels={"saldo": "Saldo ($)", "decision": ""}),
                    width="stretch")

    c3, c4 = st.columns(2)
    muestra = ev.sample(min(3000, len(ev)), random_state=0)
    c3.plotly_chart(px.scatter(
        muestra, x="prob_fuga", y="margen_cliente", color="decision",
        color_discrete_map=COLORES, opacity=.6,
        hover_data=[config.ID_COL, "Balance", "incentivo", "ganancia_neta_esperada"],
        title="Mapa de decisión: riesgo vs. margen",
        labels={"prob_fuga": "Probabilidad de fuga", "margen_cliente": "Margen ($)"}),
        width="stretch")
    c4.plotly_chart(px.histogram(
        ev, x="prob_fuga", color="decision", nbins=40, color_discrete_map=COLORES,
        title="Distribución de probabilidades", labels={"prob_fuga": "Probabilidad de fuga"}),
        width="stretch")

    st.subheader("Matriz de decisión comercial")
    mat = ev.groupby(["segmento", "decision"]).agg(
        Clientes=("prob_fuga", "size"), **{"Prob. media": ("prob_fuga", "mean")},
        **{"Saldo medio": ("Balance", "mean")},
        **{"Costo total": ("costo_incentivo", "sum")},
        **{"Ganancia neta": ("ganancia_neta_esperada", "sum")}).reset_index()
    st.dataframe(mat.style.format({"Prob. media": "{:.2f}", "Saldo medio": "${:,.0f}",
                                   "Costo total": "${:,.0f}", "Ganancia neta": "${:,.0f}"}),
                 hide_index=True, width="stretch")

    st.subheader("Clientes prioritarios a intervenir")
    top = ev[ev["costo_incentivo"] > 0].sort_values("ganancia_neta_esperada", ascending=False)
    cols = [config.ID_COL, "Geography", "Age", "Balance", "prob_fuga", "margen_cliente",
            "incentivo", "costo_incentivo", "ganancia_neta_esperada"]
    st.dataframe(top[cols].head(100).style.format(
        {"Balance": "${:,.0f}", "prob_fuga": "{:.2%}", "margen_cliente": "${:,.0f}",
         "costo_incentivo": "${:,.0f}", "ganancia_neta_esperada": "${:,.0f}"}),
        hide_index=True, width="stretch")
    st.download_button("📥 Descargar cartera evaluada (CSV)", ev.to_csv(index=False),
                       "cartera_evaluada.csv", "text/csv")


def vista_simulador(df, p):
    st.header("Simulador de parámetros")
    st.caption("Modifica costos y efectividades en la barra lateral. Aquí ves el "
               "efecto frente a los parámetros por defecto y frente a una estrategia "
               "ingenua (dar incentivo a todo cliente con prob ≥ 0.5).")
    base_p = ParametrosFinancieros()
    X = df.drop(columns="prob_fuga")
    prob = df["prob_fuga"].to_numpy()
    ev_actual = evaluar_cartera(X, prob, p)
    r_act = resumen_cartera(ev_actual)
    r_base = resumen_cartera(evaluar_cartera(X, prob, base_p))
    ing = estrategia_ingenua(ev_actual, p)

    comp = pd.DataFrame({
        "Escenario": ["Parámetros por defecto", "Escenario simulado",
                      "Estrategia ingenua (p ≥ 0.5)"],
        "Clientes intervenidos": [r_base["clientes_a_intervenir"],
                                  r_act["clientes_a_intervenir"], ing["clientes"]],
        "Inversión": [r_base["inversion_total"], r_act["inversion_total"], ing["inversion"]],
        "Ganancia neta esperada": [r_base["ganancia_neta_total"],
                                   r_act["ganancia_neta_total"], ing["ganancia_neta"]],
        "ROI": [r_base["roi"], r_act["roi"], ing["roi"]],
        "Clientes con ganancia negativa": [0, 0, ing["clientes_con_perdida"]],
    })
    k = st.columns(4)
    k[0].metric("Clientes a intervenir", f"{r_act['clientes_a_intervenir']:,}",
                f"{r_act['clientes_a_intervenir'] - r_base['clientes_a_intervenir']:+,}")
    k[1].metric("Inversión", money(r_act["inversion_total"]),
                f"{r_act['inversion_total'] - r_base['inversion_total']:+,.0f}",
                delta_color="inverse")
    k[2].metric("Ganancia neta esperada", money(r_act["ganancia_neta_total"]),
                f"{r_act['ganancia_neta_total'] - r_base['ganancia_neta_total']:+,.0f}")
    k[3].metric("ROI", f"{r_act['roi']:.0%}", f"{(r_act['roi'] - r_base['roi']) * 100:+.0f} pp")

    st.dataframe(comp.style.format({"Inversión": "${:,.0f}",
                                    "Ganancia neta esperada": "${:,.0f}", "ROI": "{:.0%}"}),
                 hide_index=True, width="stretch")
    st.caption("La estrategia ingenua puede mostrar un ROI más alto porque gasta en "
               "pocos clientes, pero deja de ganar dinero: el criterio de decisión es "
               "la **ganancia neta total**, y además incluye clientes con ganancia "
               "esperada negativa (gasto que no se recupera).")
    st.plotly_chart(px.bar(comp, x="Escenario", y="Ganancia neta esperada",
                           color="Escenario", text_auto=".3s",
                           title="Ganancia neta esperada por escenario"),
                    width="stretch")

    st.subheader("Sensibilidad al costo de las campañas")
    st.caption("Multiplica TODOS los costos del catálogo por un factor y observa "
               "cómo cambian la ganancia y el número de clientes intervenidos.")
    factores = np.linspace(0.25, 4, 16)
    filas = []
    for fct in factores:
        q = deepcopy(p)
        for a in q.catalogo.values():
            a.costo *= fct
        r = resumen_cartera(evaluar_cartera(X, prob, q))
        filas.append({"Factor de costo": fct, "Ganancia neta": r["ganancia_neta_total"],
                      "Clientes intervenidos": r["clientes_a_intervenir"],
                      "Inversión": r["inversion_total"]})
    sens = pd.DataFrame(filas)
    fig = go.Figure()
    fig.add_bar(x=sens["Factor de costo"], y=sens["Clientes intervenidos"],
                name="Clientes intervenidos", yaxis="y2", opacity=.35)
    fig.add_scatter(x=sens["Factor de costo"], y=sens["Ganancia neta"],
                    name="Ganancia neta ($)", mode="lines+markers")
    fig.update_layout(xaxis_title="Factor sobre el costo actual (1 = actual)",
                      yaxis=dict(title="Ganancia neta ($)"),
                      yaxis2=dict(title="Clientes", overlaying="y", side="right"),
                      legend=dict(orientation="h"), height=420)
    st.plotly_chart(fig, width="stretch")


def vista_modelo():
    st.header("Modelo de red neuronal")
    if not config.METRICS_PATH.exists():
        st.warning("Ejecuta `python -m src.train` para generar las métricas.")
        return
    m = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    k = st.columns(4)
    k[0].metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    k[1].metric("PR-AUC", f"{m['pr_auc']:.3f}", f"base {m['tasa_fuga_real_test']:.2f}",
                delta_color="off")
    k[2].metric("Brier (calibrado)", f"{m['brier_calibrado']:.3f}",
                f"{m['brier_calibrado'] - m['brier_sin_calibrar']:+.3f}", delta_color="inverse")
    k[3].metric("ECE (calibrado)", f"{m['ece_calibrado']:.3f}",
                f"{m['ece_calibrado'] - m['ece_sin_calibrar']:+.3f}", delta_color="inverse")
    st.markdown(
        "**Arquitectura:** MLP 128 (ReLU + BatchNorm + Dropout 0.3) → 64 (ReLU + Dropout 0.2) "
        "→ 1 (Sigmoide) · Adam (lr=0.001) · Binary Cross-Entropy · `class_weight` balanceado · "
        f"Early Stopping (paciencia {config.PATIENCE}, mejor época {m['mejor_epoca']}) · "
        "Calibración Platt sobre validación.")
    st.info(f"Variables excluidas: {', '.join(m['variables_excluidas'])}. "
            "`Complain` se excluye por **fuga de datos**: casi todos los clientes que se "
            "fueron tienen queja registrada, por lo que es una consecuencia de la salida, "
            "no una causa observable antes.")
    for img, cap in [("diagnostico_modelo.png", "Pérdida, ROC, Precision-Recall y calibración"),
                     ("shap_importancia.png", "Factores determinantes de abandono (SHAP)"),
                     ("shap_beeswarm.png", "Dirección del efecto de cada variable (SHAP)")]:
        path = config.REPORTS_DIR / img
        if path.exists():
            st.image(str(path), caption=cap)


# ---------------------------------------------------------------------------
def main():
    p = sidebar_params()
    try:
        df = get_portfolio()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    tabs = st.tabs(["Buscador", "Cliente nuevo", "Panel de cartera",
                    "Simulador", "Modelo"])
    with tabs[0]:
        vista_buscador(df, p)
    with tabs[1]:
        vista_nuevo(p)
    with tabs[2]:
        vista_panel(df, p)
    with tabs[3]:
        vista_simulador(df, p)
    with tabs[4]:
        vista_modelo()


if __name__ == "__main__":
    main()
