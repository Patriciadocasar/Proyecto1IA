"""
finance.py
----------
Lógica financiera del sistema "Deep Retention".

1) Margen del cliente (lo que el banco gana con él en el horizonte):
       Margen = (Balance × spread_neto
                 + NumOfProducts × comisión_por_producto
                 + HasCrCard × ingreso_tarjeta
                 - costo_operativo_cuenta) × horizonte_años

   El dataset no trae comisiones ni volumen transaccional, así que se
   ESTIMAN con supuestos de negocio parametrizables (se ajustan en el
   simulador de la app). Un cliente con saldo bajo y pocos productos puede
   tener margen negativo => es DEFICITARIO.

2) Regla de decisión (la que exige el enunciado; NO se usa umbral fijo 0.5):
       Ganancia Neta Esperada = P(fuga) × Efectividad × Margen − Costo

   Para cada cliente se evalúan todas las acciones del catálogo a las que
   es elegible, se elige la de mayor ganancia, y SOLO se ejecuta si es > 0.

3) Segmento D (Estable) de la matriz del informe: si P(fuga) < umbral_estable
   (0.30) el cliente no recibe incentivo aunque la fórmula dé positivo. Con
   margen alto, la bonificación de $25 sale "rentable" incluso con P(fuga)
   de 0.05, pero es gasto en clientes que casi seguro se quedan.
"""
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Catálogo de acciones de retención
# ---------------------------------------------------------------------------
@dataclass
class Accion:
    nombre: str
    costo: float          # costo unitario por cliente ($)
    efectividad: float    # prob. de que el incentivo evite la fuga (0-1)
    niveles: tuple        # niveles patrimoniales elegibles


def catalogo_por_defecto() -> dict:
    return {
        "vip": Accion("Gestor VIP y Tarifa Plana", 250.0, 0.45, ("Patrimonial",)),
        "exoneracion": Accion("Exoneración de comisiones", 80.0, 0.35,
                              ("Patrimonial", "Medio")),
        "bonificacion": Accion("Bonificación transaccional", 25.0, 0.25,
                               ("Patrimonial", "Medio", "Básico")),
    }


@dataclass
class ParametrosFinancieros:
    spread_neto: float = 0.015            # 1.5% anual sobre el saldo
    comision_por_producto: float = 60.0   # $ al año por producto contratado
    ingreso_tarjeta: float = 30.0         # $ al año si tiene tarjeta de crédito
    costo_operativo: float = 120.0        # $ al año por mantener la cuenta
    horizonte_anios: float = 1.0          # años de margen que se pierden si se va
    umbral_patrimonial: float = 120_000.0 # saldo para nivel "Patrimonial"
    umbral_medio: float = 50_000.0        # saldo para nivel "Medio"
    umbral_estable: float = 0.30          # prob. bajo la cual el cliente es estable
    catalogo: dict = field(default_factory=catalogo_por_defecto)

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Cálculos
# ---------------------------------------------------------------------------
def margen_cliente(df: pd.DataFrame, p: ParametrosFinancieros) -> np.ndarray:
    anual = (df["Balance"].to_numpy() * p.spread_neto
             + df["NumOfProducts"].to_numpy() * p.comision_por_producto
             + df["HasCrCard"].to_numpy() * p.ingreso_tarjeta
             - p.costo_operativo)
    return anual * p.horizonte_anios


def nivel_patrimonial(df: pd.DataFrame, p: ParametrosFinancieros) -> np.ndarray:
    b = df["Balance"].to_numpy()
    return np.where(b >= p.umbral_patrimonial, "Patrimonial",
                    np.where(b >= p.umbral_medio, "Medio", "Básico"))


def ganancia_neta_esperada(prob, efectividad, margen, costo):
    """Fórmula central del enunciado."""
    return prob * efectividad * margen - costo


def evaluar_cartera(df: pd.DataFrame, prob: np.ndarray,
                    p: ParametrosFinancieros | None = None) -> pd.DataFrame:
    """Recibe clientes + probabilidad de fuga y devuelve la decisión comercial
    de cada uno con todos los montos."""
    p = p or ParametrosFinancieros()
    out = df.copy()
    prob = np.asarray(prob, dtype=float)
    margen = margen_cliente(df, p)
    nivel = nivel_patrimonial(df, p)

    # Ganancia de cada acción del catálogo (−inf si el cliente no es elegible)
    claves = list(p.catalogo.keys())
    ganancias = np.full((len(df), len(claves)), -np.inf)
    for j, k in enumerate(claves):
        a = p.catalogo[k]
        elegible = np.isin(nivel, a.niveles)
        g = ganancia_neta_esperada(prob, a.efectividad, margen, a.costo)
        ganancias[:, j] = np.where(elegible, g, -np.inf)

    mejor = ganancias.argmax(axis=1)
    mejor_ganancia = ganancias[np.arange(len(df)), mejor]
    accion = np.array([p.catalogo[claves[i]].nombre for i in mejor])
    costo = np.array([p.catalogo[claves[i]].costo for i in mejor])
    efect = np.array([p.catalogo[claves[i]].efectividad for i in mejor])

    estable = prob < p.umbral_estable
    retener = (~estable) & (mejor_ganancia > 0)
    salida = (~retener) & (~estable)

    segmento = np.select(
        [retener & (nivel == "Patrimonial"), retener, salida, estable],
        ["A: Alto Valor / Alto Riesgo", "B: Medio / Riesgo Moderado",
         "C: No Rentable", "D: Estable"], default="")
    decision = np.select(
        [retener & (nivel == "Patrimonial"), retener, salida, estable],
        ["Retención Prioritaria", "Retención Automatizada",
         "Salida Permitida", "Sin Acción Requerida"], default="")
    motivo = np.select(
        [estable, retener, salida & (margen <= 0), salida],
        ["Riesgo bajo de fuga",
         "Ganancia neta esperada positiva",
         "Cliente deficitario (margen ≤ 0)",
         "El incentivo cuesta más de lo que rescata"], default="")

    out["prob_fuga"] = prob
    out["margen_cliente"] = margen
    out["nivel_patrimonial"] = nivel
    out["segmento"] = segmento
    out["decision"] = decision
    out["motivo"] = motivo
    out["incentivo"] = np.where(retener, accion, "Sin intervención")
    out["costo_incentivo"] = np.where(retener, costo, 0.0)
    out["efectividad"] = np.where(retener, efect, 0.0)
    out["ganancia_neta_esperada"] = np.where(retener, mejor_ganancia, 0.0)
    # ganancia de la mejor acción aunque no se ejecute (útil para explicar)
    out["ganancia_mejor_accion"] = mejor_ganancia
    out["mejor_accion_posible"] = accion
    # balance esperado que se pierde y que se rescata
    out["balance_en_riesgo"] = prob * df["Balance"].to_numpy()
    out["balance_rescatado"] = np.where(
        retener, prob * efect * df["Balance"].to_numpy(), 0.0)
    return out


def resumen_cartera(ev: pd.DataFrame, umbral_critico: float = 0.5) -> dict:
    """KPIs para el panel y para el informe ejecutivo."""
    inter = ev[ev["costo_incentivo"] > 0]
    inversion = float(inter["costo_incentivo"].sum())
    ganancia = float(inter["ganancia_neta_esperada"].sum())
    criticos = ev[ev["prob_fuga"] >= umbral_critico]
    return {
        "clientes_analizados": int(len(ev)),
        "clientes_riesgo_critico": int(len(criticos)),
        "capital_riesgo_critico": float(criticos["Balance"].sum()),
        "balance_esperado_en_riesgo": float(ev["balance_en_riesgo"].sum()),
        "clientes_a_intervenir": int(len(inter)),
        "capital_rescatable": float(inter["balance_rescatado"].sum()),
        "margen_rescatado": float((inter["ganancia_neta_esperada"]
                                   + inter["costo_incentivo"]).sum()),
        "inversion_total": inversion,
        "ganancia_neta_total": ganancia,
        "roi": ganancia / inversion if inversion > 0 else 0.0,
        "clientes_salida_permitida": int((ev["decision"] == "Salida Permitida").sum()),
    }


def matriz_decision(ev: pd.DataFrame) -> pd.DataFrame:
    """Tabla de la sección 3 del informe (Matriz de Decisión Comercial)."""
    g = ev.groupby(["segmento", "decision", "incentivo"]).agg(
        clientes=("prob_fuga", "size"),
        prob_media=("prob_fuga", "mean"),
        saldo_medio=("Balance", "mean"),
        costo_total=("costo_incentivo", "sum"),
        ganancia_neta=("ganancia_neta_esperada", "sum"),
        capital_rescatable=("balance_rescatado", "sum"),
    ).reset_index()
    return g


def estrategia_ingenua(ev: pd.DataFrame, p: ParametrosFinancieros,
                       umbral: float = 0.5) -> dict:
    """Comparación: dar la acción más barata elegible a TODO cliente con
    prob ≥ 0.5, sin mirar rentabilidad (lo que hace hoy el banco)."""
    sel = ev[ev["prob_fuga"] >= umbral]
    total_costo, total_gan, perdida, n_perdida = 0.0, 0.0, 0.0, 0
    for nivel, grupo in sel.groupby("nivel_patrimonial"):
        acciones = [a for a in p.catalogo.values() if nivel in a.niveles]
        a = min(acciones, key=lambda x: x.costo)
        g = ganancia_neta_esperada(grupo["prob_fuga"], a.efectividad,
                                   grupo["margen_cliente"], a.costo)
        total_costo += a.costo * len(grupo)
        total_gan += float(g.sum())
        perdida += float(g[g < 0].sum())
        n_perdida += int((g < 0).sum())
    # Nota: el ROI de esta estrategia puede salir MAYOR que el del sistema,
    # porque solo gasta en pocos clientes con P(fuga) muy alta. El criterio
    # de negocio es la ganancia neta total, no el ROI: el sistema invierte en
    # más clientes mientras cada peso adicional siga dejando ganancia.
    return {"clientes": int(len(sel)), "inversion": total_costo,
            "ganancia_neta": total_gan,
            "roi": total_gan / total_costo if total_costo else 0.0,
            "clientes_con_perdida": n_perdida,
            "perdida_en_clientes_no_rentables": perdida}
