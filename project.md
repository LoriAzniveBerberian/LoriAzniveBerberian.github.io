# Forecasting California kelp canopy with simple machine learning

_AOS C111 / C204 – Final Project_  
_Lori Aznive Berberian_

---

## 1. Motivation

Giant kelp (_Macrocystis pyrifera_) and other canopy-forming macroalgae create structurally complex, productive habitats along the California coast. Their surface canopy area fluctuates due to waves, storms, marine heatwaves, and land–ocean connections such as freshwater and sediment runoff. For my PhD research, I am interested in how wildfires and post-fire runoff affect kelp forests, but to measure “impact” I first need a **baseline expectation** of how kelp would behave in the absence of those events.

Most existing kelp models in the literature incorporate a rich set of covariates: waves, temperature, nutrients, upwelling indices, and sometimes local site effects. In this project, I take a deliberately simpler approach that fits within the scope of an ML class: I ask

> How much of kelp canopy variability can we predict using only kelp’s own recent history?

If a simple, regularized linear model using only past canopy already explains a large fraction of variance, then that provides a strong null model. Future wildfire or climate impacts can then be framed as deviations from this baseline, rather than from raw observations alone.

---

## 2. Data

### 2.1 Kelp canopy data set

For this project I used a gridded kelp canopy product derived from Landsat imagery along the California coast. The key features of the data set are:

- Spatial grid: approximately 1 km coastal pixels, each treated as a separate “station”.
- Time axis: repeated observations over many years, aggregated to regular time steps (e.g., quarters).
- Variable of interest: kelp canopy area per pixel and time step (units: m²). I work with a transformed version of this variable (such as log or standardized area) in the machine learning pipeline.

In code, the data are stored in an `xarray.Dataset` with dimensions:

- `station` – unique ID for each coastal pixel
- `time` – regular time steps (e.g., quarterly)
- associated coordinates for latitude and longitude per station

I subsetted to coastal pixels with at least some kelp presence over the record to avoid pixels that are always zero.

### 2.2 Preprocessing

The main preprocessing steps were:

1. **Time aggregation**  
   Starting from the original time series, I used a cleaned, regular grid (e.g., quarterly maxima). This step had already been done before entering the ML workflow.

2. **History window construction**  
   For each station and time step, I wanted a block of past canopy values as input and the next time step as the target. I used a helper function `make_supervised` (in `kelp_ml_utils`) to convert the `(station, time)` data into supervised learning arrays:
   - Features: `[y(t-1), y(t-2), ..., y(t-k)]`
   - Target: `y(t)`  
   where `k` is the length of the history window (in time steps).

3. **Train–test split by time**  
   Rather than randomly shuffling all samples, I split by time using `train_test_split_time`, so the model is trained on earlier years and evaluated on later years. This better reflects the forecasting use-case and avoids leakage from the “future” into the training set.

4. **Optional transformations**  
   I experimented with:
   - scaling/standardizing features
   - log-transforming canopy area to reduce the influence of very large canopies

The final setup is a clean `(X, y)` pair for training and testing that includes all stations concatenated together, with time order preserved.

---

## 3. Methods

### 3.1 Supervised learning setup

The supervised learning problem is:

- **Input**: a vector of past kelp canopy values at a given pixel, over a fixed number of time steps (e.g., 8 quarters or 1–2 years of history).
- **Output**: kelp canopy at the next time step at the same pixel.

Mathematically, for each station \( i \) and time \( t \), I want to learn a mapping

\[
\mathbf{x}_{i,t} = [y_{i,t-1}, y_{i,t-2}, \dots, y_{i,t-k}] \rightarrow y_{i,t}
\]

across all stations and times in the training set.

I also defined a **naive baseline** model:

- Naive prediction: \( \hat{y}_{i,t}^{\text{naive}} = y_{i,t-1} \)

This checks whether my machine learning model is doing better than simply copying the last observation.

### 3.2 Models

I focused on two models:

1. **Ridge regression (main model)**  
   Ridge regression is a linear model with L2 regularization on the coefficients. In scikit-learn notation:

   ```python
   from sklearn.linear_model import Ridge

   model = Ridge(alpha=alpha)
   model.fit(X_train, y_train)
