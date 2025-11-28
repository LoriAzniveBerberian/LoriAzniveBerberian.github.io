# Forecasting California kelp canopy with simple machine learning

_AOS C111 / C204 – Final Project_  
_Lori Aznive Berberian_

## 1. Motivation

giant kelp (_macrocystis pyrifera_), bull kelp (_nereocystis_), and other canopy-forming macroalgae create structurally complex, productive habitats along the california coast. their surface canopy area changes over time with waves, storms, marine heatwaves, and land–ocean connections such as freshwater and sediment runoff.

for my phd research, i am interested in how changes in water clarity and light availability affect shallow benthic habitats and canopy-forming kelps, especially during events like high river discharge, post-fire runoff, and marine heatwaves. to say whether these events have an “impact,” i first need a baseline expectation for how kelp canopy would change in their absence.

most kelp models in the literature include many environmental drivers (waves, temperature, nutrients, upwelling, etc.). here, i intentionally use a much simpler setup that fits the scope of this ml class and ask:

> **how much of kelp canopy variability along california can we predict using only kelp’s own recent history?**

if a simple, regularized linear model using only past canopy explains a large fraction of variance, that gives a strong null model. later, i can interpret deviations from this baseline in terms of changes in light, turbidity, or other drivers.

---

## 2. Data

for this project i use a gridded kelp canopy product derived from landsat imagery along the california coast. the data set consists of ~1 km coastal pixels, each treated as a separate “station,” followed over many years at regular quarterly time steps. the main variable is kelp canopy area per pixel per quarter (units: m²).

in code, the data are stored in an `xarray.Dataset` with dimensions `station` (unique coastal pixel id) and `time` (quarterly time steps), along with latitude and longitude for each station. i subset to a california lat/lon box and keep only stations that have non-zero canopy at least once, so i focus on locations where kelp actually occurs.

before fitting models, i use a cleaned quarterly canopy time series for each station, treat missing values as “no observation” rather than true zero canopy, and drop stations that have too few valid observations to support a history window. i also experiment with simple transformations like `log(1 + area)` to reduce the influence of very large canopy values, but the core idea is to keep the preprocessing minimal and transparent.

helper functions in `kelp_ml_utils` handle the ml-ready formatting: `make_supervised` converts each 1d station time series into supervised learning pairs (lagged inputs → next-step target), and `train_test_split_time` splits the data into an early training period and a later test period while preserving time order. all stations are then concatenated into a single `(X, y)` data set for training and testing.

---

## 3. Methods

i frame the problem as a one-step-ahead forecasting task. for each station `i` and time `t`, the input is a vector of past canopy values over a fixed number of quarters (a history window), and the target is the canopy at the next quarter at the same station.

in notation, i write the input vector as  
`x[i,t] = [y[i,t-1], y[i,t-2], ..., y[i,t-k]]`  

and the target as  
`y[i,t] = kelp canopy at time t`  

where `k` is the length of the history window used in the current script (on the order of a year of quarterly history).

as a simple baseline, i define a naive “persistence” model that just copies the last observation:  
`\hat{y}_naive[i,t] = y[i,t-1]`.

the main ml model is ridge regression, a linear model with l2 regularization on the coefficients. in scikit-learn form:

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=alpha)
model.fit(X_train, y_train)
y_pred_ridge = model.predict(X_test)

the regularization strength `alpha` controls how strongly the coefficients are shrunk toward zero, which helps prevent overfitting when the history window is long relative to the number of training samples. in practice i build the model using a helper function `make_ridge_model` from `kelp_ml_utils`, which sets up the ridge estimator (and any scaling) in a consistent way.

to respect time ordering, i split the data by time, not randomly: early years are used for training and later years are held out for testing. this mimics “train on the past, forecast the future” and avoids leakage from the future into the training set.

on the held-out test set, i evaluate performance using root mean squared error (`rmse`) and coefficient of determination (`r²`), and i compare the ridge model directly to the naive persistence baseline. i also compute a regression error characteristic (`rec`) curve: for a range of error tolerances, the rec curve shows the fraction of test points whose absolute error is smaller than that tolerance. this gives a more detailed look at how often predictions are “good enough” at different error levels.

## 4. Results

here i summarize qualitative patterns rather than exact values; the notebook contains full tables and plots.

overall, the history-based ridge model performs better than the naive persistence model on the held-out years. `rmse` is lower and `r²` is higher for most stations. the improvement is especially clear at sites where canopy is more variable, because the ridge model can use the full history window to anticipate trends, not just repeat the last quarter.

rec curves show that for most reasonable error thresholds, a larger fraction of ridge predictions fall within that error band compared to the naive baseline. in other words, across much of the coast, the ridge model produces more accurate forecasts more often than persistence.

when i map performance metrics back into space (for example, by looking at average `r²` per region), some parts of the coast appear more predictable than others. persistent kelp beds tend to show higher predictability, while areas with frequent crashes and recoveries are harder to predict from history alone. this spatial structure is important for later interpretation: some regions may naturally have smoother dynamics, while others are more strongly impacted by external forcing such as waves, warm events, or changes in turbidity and light.

## 5. Discussion

this project shows that a simple, regularized linear model using only kelp’s own recent history can provide a useful predictive baseline for kelp canopy dynamics along the california coast. in many pixels, the ridge model explains a substantial fraction of the variance beyond what a naive persistence model can capture.

scientifically, the results support the idea that kelp canopy has strong temporal autocorrelation and inertia: recent states carry a lot of information about the near future, particularly in persistent beds. methodologically, the ridge model functions as a “history-only” null model for impact studies. instead of comparing kelp only before and after an event, i can compare the observed trajectory to what the history-based model would have predicted in a no-disturbance world.

if a disturbance that changes water clarity and light (for example, wildfire-driven runoff that increases turbidity) leads to kelp losses much larger than the ridge baseline would expect, that is evidence of a real impact. if observed changes fall within the baseline’s expected variability, then attributing them to a specific event becomes less convincing. this connects directly to the baci-style questions i am interested in for my phd.

## 6. Limitations and future work

there are several clear limitations to this simple setup.

first, there are no explicit environmental drivers in the model. waves, sea surface temperature, nutrients, and runoff are not included, so the model can say what is predictable from history alone but not why canopy changes. second, the focus is on single-step forecasts (one quarter ahead). multi-step forecasts over two to four quarters would be useful to understand how quickly errors grow and how far into the future a history-only model remains useful. third, by pooling all stations into a single model, i implicitly assume that the same mapping from history to future applies everywhere, even though processes clearly differ by region, exposure, and depth.

future work could relax these simplifications. simple covariates such as sst anomalies, wave height, and runoff indices could be added on top of the history features to see how much they improve performance over the history-only baseline. region-specific or hierarchical ridge models could allow coefficients to vary along the coast while still sharing information. ultimately, this baseline will be integrated directly into my wildfire–kelp and light-availability analyses by comparing “expected kelp” (from the model) and “observed kelp” around disturbance events.

## 7. Code and reproducibility

all of the analysis for this project is implemented in a python notebook. the main tools are:

- `xarray` to handle the (station, time) grid of kelp canopy  
- `numpy` and `pandas` for general data manipulation  
- `scikit-learn` for ridge regression and evaluation metrics  

i also rely on a small set of custom helper functions collected in `kelp_ml_utils`, which:

- turn kelp time series into supervised learning windows (`make_supervised`)  
- split data into training and testing periods along the time axis (`train_test_split_time`)  
- build ridge models with consistent settings (`make_ridge_model`)  

once the notebook is uploaded and shared, the link will appear on the home page of this site so the results can be reproduced and extended.

## 8. Conclusion

even a very simple machine learning model that only uses kelp’s own recent history can capture much of the short-term variability in kelp canopy along the california coast. the history-only ridge model is easy to explain, fast to run, and provides a clear yardstick for two things: evaluating the added value of more complex, multi-driver models, and quantifying how much observed changes around disturbances (such as wildfire-driven turbidity events) deviate from a no-disturbance expectation.
