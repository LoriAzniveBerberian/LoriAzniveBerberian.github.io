# Forecasting California kelp canopy with simple machine learning

_AOS C111 / C204 – Final Project_  
_Lori Aznive Berberian_

---

## 1. Motivation

Giant kelp (_Macrocystis pyrifera_), bull kelp (_Nereocystis_), and other canopy-forming macroalgae create structurally complex, productive habitats along the California coast. Their surface canopy area changes over time with waves, storms, marine heatwaves, and land–ocean connections such as freshwater and sediment runoff.

For my PhD research, I am interested in how changes in water clarity and light availability affect shallow benthic habitats and canopy-forming kelps, especially during events like high river discharge, post-fire runoff, and marine heatwaves. To say whether these events have an “impact,” I first need a baseline expectation for how kelp canopy would change in their absence.

Most kelp models in the literature include many environmental drivers (waves, temperature, nutrients, upwelling, etc.). Here, I intentionally use a much simpler setup that fits the scope of this ML class and ask:

> **How much of kelp canopy variability along California can we predict using only kelp’s own recent history?**

If a simple, regularized linear model using only past canopy explains a large fraction of variance, that gives a strong null model. Later, I can interpret deviations from this baseline in terms of changes in light, turbidity, or other drivers.

---

## 2. Data

This project uses a statewide Landsat-derived kelp canopy product from the KelpWatch / SBC LTER team:

- KelpWatch portal: <https://kelpwatch.org/>  
- SBC LTER Landsat kelp canopy dataset:  
  <https://sbclter.msi.ucsb.edu/data/catalog/package/?package=knb-lter-sbc.74>

The dataset is gridded at approximately 1 km resolution. Each coastal pixel is treated as a separate **station** and followed over many years at regular quarterly time steps. The main variable is kelp canopy area per pixel per quarter (units: m²).

In code, the data are stored in an `xarray.Dataset` with dimensions:

- `station` – Unique coastal pixel ID  
- `time` – Quarterly time steps  

and coordinates `latitude(station)` and `longitude(station)`. I subset to a California lat/lon box and keep only stations that have non-zero canopy at least once, so I focus on locations where kelp actually occurs.

Before fitting models, I:

1. Work with the cleaned quarterly canopy time series for each station.  
2. Treat missing values as “no observation” rather than true zero canopy.  
3. Drop stations with too few valid observations to support a history window.  
4. Optionally apply `log(1 + area)` to reduce the influence of very large canopy values.

Helper functions in `kelp_ml_utils` handle the ML-ready formatting: `make_supervised` converts each 1D station time series into supervised learning pairs (lagged inputs → next-step target), and `train_test_split_time` splits the data into an early training period and a later test period while preserving time order. All stations are then concatenated into a single `(X, y)` dataset for training and testing.

Below are two simple exploratory figures:

![Figure 1. California quarterly mean kelp canopy area.](figures/CAkelp_timeSeries.png)

**Figure 1.** California’s quarterly average kelp canopy area from 1984–2024 (mean m² of canopy per pixel over all California stations).

![Figure 2. Kelp habitat along the California coast.](figures/CAkelp_Habitat.png)

**Figure 2.** Kelp habitat along the California coast: Stations where canopy area is non-zero at least once in the time series.

---

## 3. Methods

I frame the problem as a **one-step-ahead forecasting** task. For each station `i` and time `t`, the input is a vector of past canopy values over a fixed number of quarters (a history window), and the target is the canopy at the next quarter at the same station:

- Input:  
  `x[i,t] = [y[i,t-1], y[i,t-2], ..., y[i,t-k]]`  

- Target:  
  `y[i,t] = kelp canopy at time t`  

Here `k` is the length of the history window (in most runs, 4 quarters ≈ 1 year).

As a simple baseline, I define a naive **persistence** model that just copies the last observation:

\[
\hat{y}_{\text{naive}}[i,t] = y[i,t-1].
\]

The main ML model is **ridge regression**, a linear model with L2 regularization on the coefficients. In scikit-learn form:

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=alpha)
model.fit(X_train, y_train)
y_pred_ridge = model.predict(X_test)
