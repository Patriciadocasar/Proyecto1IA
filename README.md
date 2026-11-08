# 🏦 Deep Retention: Sistema de Retención Selectiva de Clientes

Motor predictivo basado en **redes neuronales** que estima la probabilidad de fuga de cada cliente bancario y decide, con una **regla financiera**, si vale la pena gastar dinero en retenerlo.

📄 **Informe ejecutivo (PDF):** [Informe_Ejecutivo_Deep_Retention.pdf](docs/Informe_Ejecutivo_Deep_Retention.pdf) <!-- ← colocar aquí el PDF -->

---

## ¿Qué hace?

1. Una **red neuronal (MLP)** calcula la probabilidad de fuga (0.00 a 1.00) de cada cliente.
2. Se estima el **margen** que el banco gana con el cliente (saldo, productos y tarjeta, menos el costo operativo).
3. Para cada incentivo del catálogo se calcula:

   **Ganancia Neta Esperada = P(fuga) × Efectividad del incentivo × Margen del cliente − Costo del incentivo**

4. Se elige el incentivo con mayor ganancia y **solo se ejecuta si la ganancia es mayor a 0**. Los clientes deficitarios quedan como **"Salida Permitida"**.

| Segmento | Decisión | Incentivo |
|---|---|---|
| A: Alto Valor | Retención Prioritaria | Gestor VIP y Tarifa Plana / Exoneración |
| B: Medio | Retención Automatizada | Exoneración de comisiones / Bonificación |
| C: No Rentable | Salida Permitida | Sin intervención |
| D: Estable | Sin Acción Requerida | Mantenimiento ordinario |

## Estructura del repositorio

```
deep-retention/
├── app/
│   └── app.py              # Aplicación Streamlit (solo inferencia)
├── data/
│   └── Customer-Churn-Records.csv
├── models/                 # Artefactos entrenados
│   ├── deep_retention_mlp.keras   # pesos de la red
│   ├── preprocessor.joblib        # StandardScaler + OneHotEncoder
│   ├── calibrator.joblib          # calibración de probabilidades (Platt)
│   └── shap_background.npy        # muestra de referencia para SHAP
├── reports/                # Métricas, gráficas y cartera puntuada
├── src/
│   ├── config.py           # rutas, columnas, hiperparámetros
│   ├── preprocessing.py    # carga, escalado, codificación y validación
│   ├── model.py            # arquitectura de la red neuronal
│   ├── train.py            # canal de ENTRENAMIENTO
│   ├── predict.py          # canal de INFERENCIA + explicabilidad SHAP
│   └── finance.py          # margen, regla de ganancia neta, catálogo
├── requirements.txt
└── README.md
```

## Instalación y ejecución

Requiere Python 3.10 o superior.

```bash
git clone https://github.com/<usuario>/deep-retention.git
cd deep-retention
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

# (Opcional) reentrenar la red. Los pesos ya vienen en models/
python -m src.train

# Lanzar la aplicación
streamlit run app/app.py
```

## La aplicación

| Pestaña | Para qué sirve |
|---|---|
| 🔎 **Buscador** | Busca un cliente por ID o apellido: probabilidad de la red, margen, decisión, incentivo, cálculo de la fórmula y explicación SHAP |
| ➕ **Cliente nuevo** | Formulario (con validación de datos) y carga masiva por CSV |
| 📊 **Panel de cartera** | Capital en riesgo, capital rescatable, inversión, ROI, matriz de decisión y lista de clientes prioritarios |
| 🎛️ **Simulador** | Cambia costos y efectividades en la barra lateral y compara contra los valores por defecto y contra la estrategia ingenua (p ≥ 0.5) |
| 🧠 **Modelo** | Métricas técnicas, curvas y factores de abandono |

## Modelo

- **Preprocesamiento:** `StandardScaler` para variables numéricas y `OneHotEncoder` para `Geography`, `Gender` y `Card Type`. Se ajusta **solo con entrenamiento**.
- **División:** 70 % entrenamiento, 15 % validación y 15 % prueba, estratificada.
- **Arquitectura:** Dense 128 (ReLU) + BatchNorm + Dropout 0.3 → Dense 64 (ReLU) + Dropout 0.2 → Dense 1 (Sigmoide). Optimizador Adam (lr = 0.001), pérdida binary cross-entropy.
- **Desbalance (20 % de fuga):** `class_weight="balanced"`.
- **Sobreajuste:** Early Stopping sobre `val_loss` con paciencia de 15 épocas y `restore_best_weights`.
- **Calibración:** como `class_weight` infla las probabilidades, se aplica Platt scaling con el conjunto de validación. Esto es necesario porque la probabilidad entra en una fórmula de dinero.
- **Fuga de datos evitada:** se excluye `Complain`, porque 2034 de los 2038 clientes que se fueron tienen queja registrada. Es una consecuencia de la salida y no estaría disponible antes. Con ella el AUC sería de aproximadamente 0.999, un valor artificial.

### Resultados (conjunto de prueba, 1500 clientes)

| ROC-AUC | PR-AUC | Brier | ECE |
|---|---|---|---|
| 0.851 | 0.685 | 0.106 | 0.019 |

**Factores principales de abandono (SHAP):** edad, número de productos, si es miembro activo y país (Alemania).

## Supuestos financieros

El dataset no trae comisiones ni volumen transaccional, así que el margen se **estima** con estos supuestos, todos ajustables en la app:

| Parámetro | Valor por defecto |
|---|---|
| Spread neto sobre saldo | 1.5 % anual |
| Comisión por producto | $60 al año |
| Ingreso por tarjeta de crédito | $30 al año |
| Costo operativo por cuenta | $120 al año |
| Gestor VIP y Tarifa Plana | $250, efectividad 45 %, solo nivel Patrimonial (saldo ≥ $120k) |
| Exoneración de comisiones | $80, efectividad 35 %, niveles Patrimonial y Medio (saldo ≥ $50k) |
| Bonificación transaccional | $25, efectividad 25 %, todos los niveles |
