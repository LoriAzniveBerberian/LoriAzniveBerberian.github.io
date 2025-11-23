# Forecasting California kelp canopy with simple machine learning

_AOS C111 / C204 – Final Project_  
_Lori Aznive Berberian_

---

## 1. Motivation

Giant kelp (_Macrocystis pyrifera_), Bull kelp (_Nereocystis_), and other canopy-forming macroalgae create structurally complex, productive habitats along the California coast. Their surface canopy area fluctuates due to waves, storms, marine heatwaves, and land–ocean connections such as freshwater and sediment runoff.

For my PhD research, I am broadly interested in how changes in **water clarity and light availability** shape shallow benthic habitats and canopy-forming kelps, and how those conditions shift during events like high river discharge, post-fire runoff, and marine heatwaves. To measure “impact” from any of these disturbances, I first need a **baseline expectation** of how kelp would behave in the absence of those events.

Most existing kelp models in the literature incorporate a rich set of covariates: waves, temperature, nutrients, upwelling indices, and sometimes local site effects. In this project, I take a deliberately simpler approach that fits within the scope of an ML class: I ask

> How much of kelp canopy variability can we predict using only kelp’s own recent history?

If a simple, regularized linear model using only past canopy already explains a large fraction of variance, then that provides a strong null model. Future changes in light and habitat conditions (from fires, storms, or climate extremes) can then be framed as deviations from this baseline, rather than from raw observations alone.

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
   - scaling / standardizing features  
   - log-transforming canopy area to reduce the influence of very large canopies

The final setup is a clean `(X, y)` pair for training and testing that includes all stations concatenated together, with time order preserved.

---

## 3. Methods

### 3.1 Supervised learning setup

The supervised learning problem is:

- **Input**: a vector of past kelp canopy values at a given pixel, over a fixed number of time steps (for example, 8 quarters or about 2 years of history).
- **Output**: kelp canopy at the next time step at the same pixel.

For each station `i` and time `t`, I build an input vector and target:

- Input vector:  
  `x[i,t] = [y[i,t-1], y[i,t-2], ..., y[i,t-k]]`
- Target:  
  `y[i,t]`

I also defined a **naive baseline** model:

- Naive prediction:  
  `y_hat_naive[i,t] = y[i,t-1]`

This checks whether my machine learning model is doing better than simply copying the last observation.

### 3.2 Models

I focused on two models.

#### 1. Ridge regression (main model)

Ridge regression is a linear model with L2 regularization on the coefficients. In scikit-learn notation:

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=alpha)
model.fit(X_train, y_train)
y_pred_ridge = model.predict(X_test)
```

### 3.3 Evaluation

I evaluated the models on a held-out test set in time using:

- Root Mean Square Error (RMSE)  
- Coefficient of determination (R²)  
- Comparison to the naive persistence baseline

I also computed a REC (Regression Error Characteristic) curve to compare the distribution of absolute errors between models. For a range of tolerances `epsilon`, the REC curve shows the fraction of test samples where the prediction error is less than `epsilon`.

---

## 4. Results (summary)

Here I summarize qualitative results rather than exact numbers; the notebook contains full tables and plots.

### Ridge vs naive persistence

- The ridge regression model consistently outperformed the naive baseline in terms of RMSE and R² on the held-out years.  
- The improvement was particularly noticeable at sites with more variable canopy, where extra history (beyond just the last time step) helped.

### Neural network vs ridge

- The MLP occasionally achieved slightly lower RMSE than ridge but was not dramatically better overall.  
- Given the added complexity and tuning effort, ridge provided a strong, stable baseline model.

### REC curves

REC curves showed that, for most reasonable error tolerances, ridge (and sometimes the MLP) had a larger fraction of “good” predictions than the naive model, indicating more accurate forecasts across much of the coast.

### Spatial patterns (qualitative)

When aggregating errors by region (e.g., southern, central, northern California), performance varied, with some regions showing higher predictability than others. This spatial structure is important for interpreting future changes in light and habitat conditions, since some regions may naturally have more predictable kelp dynamics.

---

## 5. Discussion

This project demonstrates that simple, regularized linear models using only kelp’s own history can provide a useful predictive baseline for canopy dynamics along the California coast. In many pixels, the ridge model explains a substantial fraction of variance beyond a naive persistence model.

From a scientific point of view, this suggests that there is strong temporal autocorrelation and inertia in kelp canopy area: recent states carry a lot of information about the near future. From a methodological point of view, having a clean, interpretable baseline is valuable for impact studies:

- If a disturbance that changes water clarity and light (e.g., wildfire runoff, extreme storms, or marine heatwaves) leads to kelp losses that are much larger than predicted by the baseline model, that is evidence of an impact.  
- If observed changes are within the baseline’s expected variability, then attributing them to a specific event becomes less convincing.

This “history-only” model is deliberately minimal. It is not meant to replace biophysical or multi-driver models, but rather to provide a null expectation for use in a broader BACI framework.

---

## 6. Limitations and future work

Some key limitations:

- **No explicit environmental drivers**  
  The model ignores waves, temperature, nutrients, and runoff. This is by design for the class project, but clearly limits interpretability of what drives changes.

- **Single-step forecasting**  
  Here I focus on predicting one time step ahead. Multi-step forecasts (for example, 2–4 quarters) would be useful for planning and for understanding how quickly errors grow in time.

- **Statewide pooling**  
  By concatenating all stations together, I implicitly assume that the same mapping applies everywhere. In reality, parameters likely differ by region, depth, and exposure.

Future extensions I am interested in:

1. Adding simple covariates (e.g., wave height, sea surface temperature, runoff indices) to see how much they improve on the history-only baseline.  
2. Fitting hierarchical or region-specific ridge models where coefficients can vary by coastal region.  
3. Integrating this baseline explicitly into my wildfire–kelp and light-availability BACI workflow as the expected kelp trajectory in the “no-disturbance” world.

---

## 7. Code and reproducibility

All of the analysis for this project is implemented in a Python notebook using:

- `xarray` for handling the `(station, time)` grid of kelp canopy  
- `numpy` and `pandas` for general data manipulation  
- `scikit-learn` for ridge regression, MLP, and evaluation metrics  
- Custom helper functions in `kelp_ml_utils` for:
  - creating supervised learning windows (`make_supervised`)  
  - splitting train and test sets in time (`train_test_split_time`)  
  - building ridge models with consistent settings (`make_ridge_model`)

Once the notebook is uploaded and shared, the link will appear on the home page of this site.

---

## 8. Take-home message

Even a very simple machine learning model, built only on kelp’s own history, can capture much of the short-term variability in canopy area at the California coast. This kind of baseline model is a useful, lightweight tool for environmental data science: it is easy to explain, fast to run, and provides a clear yardstick for evaluating the added value of more complex models and for quantifying the effects of disturbances that alter water clarity and light availability.

