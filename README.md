# lstm-streamflow-forecasting
Multi-horizon LSTM streamflow forecasting on 85 years of Slovenian Alpine river discharge data.
# LSTM Streamflow Forecasting on 85 Years of Alpine Discharge Data

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/license-MIT-green)

Multi-horizon (1–7 day) daily streamflow forecasting for a Slovenian Alpine river using a Long Short-Term Memory (LSTM) deep learning model, trained on 85 years of daily discharge observations.

---

## Summary

Built and evaluated an end-to-end LSTM forecasting pipeline for daily river discharge using the Slovenian Environment Agency (ARSO) hydrological archive for station 8500 (1940–2024, ~31,000 daily observations). The model produces 1- to 7-day-ahead forecasts using only the discharge time series and seasonal features — no exogenous meteorological inputs.

## Key Results

Evaluated on a **2018–2024 hold-out test set** (7 years, never seen during training):

| Horizon | LSTM NSE | Persistence NSE | Seasonal naïve NSE |
|--------:|---------:|----------------:|-------------------:|
| Day +1  | **0.59** | 0.55            | –1.00              |
| Day +2  | **0.39** | 0.22            | –1.00              |
| Day +3  | **0.29** | 0.05            | –1.00              |
| Day +4  | **0.24** | –0.03           | –1.00              |
| Day +5  | **0.20** | –0.11           | –1.00              |
| Day +6  | **0.16** | –0.18           | –1.00              |
| Day +7  | **0.13** | –0.23           | –1.00              |

*NSE = Nash–Sutcliffe Efficiency. 1.0 = perfect prediction, 0 = no better than the long-term mean, < 0 = worse than the mean.*

The LSTM is the only model with **positive forecast skill at multi-day horizons**. Storm peak timing is captured correctly; peak magnitudes on extreme events (> 60 m³/s) are under-predicted, consistent with the absence of precipitation forcing.

## Dataset

- **Source:** Slovenian Environment Agency (ARSO) — hydrological archive station 8500
- **Period:** 1 January 1940 – 31 December 2024 (85 years, 31,047 daily observations)
- **Variables used:** Discharge (m³/s), water level (cm), water temperature (°C)
- **Completeness:** Discharge 94%, water level 90%, temperature 82%

## Methods

### Feature engineering

- Log-transformed discharge to handle right-skewness
- Cyclical day-of-year encoding (sin/cos) to capture seasonality without a January/December discontinuity
- Lagged discharge at 1, 2, 3, 7, 14, and 30 days
- Rolling 7- and 30-day means of log-discharge
- Antecedent wetness proxies: 1- and 3-day discharge rate-of-change, 7-day discharge range
- Seasonal climatology imputation for missing water level and temperature

### Time-series split

| Split | Period | Years |
|-------|--------|-------|
| Train | 1940–2009 | 70 |
| Validation | 2010–2017 | 8 |
| Test (hold-out) | 2018–2024 | 7 |

Normalisation statistics computed on the training data only.

### Model

- 2-layer LSTM with 128 hidden units and 0.2 dropout
- Multi-output regression head producing 7 daily forecasts in a single forward pass
- 30-day input sequence length
- Peak-weighted MSE loss to address the imbalance between abundant low-flow days and rare flood events
- Adam optimiser with learning-rate-on-plateau scheduling, gradient clipping, and early stopping

### Baselines

- **Persistence:** Q(t+h) = Q(t) for all horizons h
- **Seasonal naïve:** Q(t+h) = Q(same day, previous year)

## Repository Structure
