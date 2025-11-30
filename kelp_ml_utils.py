# lori aznive berberian 
# last upadted on november 29, 2025 

from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm  # for loading bar in global loop


def load_kelp_series(nc_path, var_name="area", station_index=0):
    # load one station as a pandas series
    nc_path = Path(nc_path)

    with xr.open_dataset(nc_path) as ds:
        if var_name not in ds.data_vars:
            raise KeyError(f"variable {var_name!r} not found")
        da = ds[var_name]
        ts = da.isel(station=station_index)
        series = ts.to_series().sort_index()

    series = series.dropna()
    if len(series) == 0:
        raise ValueError("time series is empty after dropna")
    return series


def make_supervised(series, n_input=4, n_output=1):
    # turn 1d time series into x, y
    values = series.values.astype(float)
    times = series.index

    n = len(values)
    if n < n_input + n_output:
        raise ValueError("time series too short for given n_input and n_output")

    x_list = []
    y_list = []
    t_list = []

    for i in range(n_input, n - n_output + 1):
        x = values[i - n_input:i]
        y = values[i:i + n_output]

        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            continue

        x_list.append(x)
        y_list.append(y)
        t_list.append(times[i])

    if len(x_list) == 0:
        raise ValueError("no samples created; too many nans?")

    x = np.vstack(x_list)
    y = np.vstack(y_list)
    t0 = np.array(t_list)

    return x, y, t0


def train_test_split_time(x, y, train_frac=0.8):
    # simple time-ordered split
    n_samples = x.shape[0]
    n_train = int(train_frac * n_samples)

    x_train = x[:n_train]
    x_test = x[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]

    return x_train, x_test, y_train, y_test


def make_ridge_model(alpha=1.0):
    # ridge regression with standardization
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])
    return model


def sweep_history_for_series(series, n_input_list, n_output=1, train_frac=0.8, alpha=1.0):
    # test several history lengths for one station
    results = []

    for n_input in n_input_list:
        x, y, t0 = make_supervised(series, n_input=n_input, n_output=n_output)
        if n_output == 1:
            y = y.ravel()

        x_train, x_test, y_train, y_test = train_test_split_time(
            x, y, train_frac=train_frac
        )

        model = make_ridge_model(alpha=alpha)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append((n_input, rmse, r2))

    return results


def fit_ridge_for_series(series, n_input=4, n_output=1, train_frac=0.8, alpha=1.0):
    # fit ridge on one station and return data for plotting
    series = series.dropna()
    x, y, t0 = make_supervised(series, n_input=n_input, n_output=n_output)

    if n_output == 1:
        y = y.ravel()

    x_train, x_test, y_train, y_test = train_test_split_time(
        x, y, train_frac=train_frac
    )

    model = make_ridge_model(alpha=alpha)
    model.fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    n_train = x_train.shape[0]
    t_train = t0[:n_train]
    t_test = t0[n_train:]

    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)

    out = {
        "model": model,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "t_train": t_train,
        "t_test": t_test,
        "rmse_test": rmse_test,
        "r2_test": r2_test,
    }

    return out


def rec_curve(y_true, y_pred, max_tol=None, n_points=51, percentile=95):
    # regression error characteristic curve
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    abs_errors = np.abs(y_true - y_pred)
    mask = np.isfinite(abs_errors)
    abs_errors = abs_errors[mask]

    if abs_errors.size == 0:
        return np.array([]), np.array([])

    if max_tol is None:
        max_tol = np.percentile(abs_errors, percentile)

    if max_tol <= 0:
        return np.array([0.0]), np.array([1.0])

    tols = np.linspace(0, max_tol, n_points)
    frac = np.array([(abs_errors <= t).mean() for t in tols])

    return tols, frac


def run_ridge_all_stations(
    area,
    time_index,
    n_input=4,
    n_output=1,
    train_frac=0.7,
    alpha=1.0,
    return_coefs=False,
):
    # fit ridge model to all stations
    if n_output != 1:
        raise NotImplementedError("n_output must be 1 for now")

    ntime = len(time_index)
    nstation = area.sizes["station"]

    time_to_idx = {t: i for i, t in enumerate(time_index)}

    sum_true = np.zeros(ntime, dtype=float)
    sum_pred = np.zeros(ntime, dtype=float)
    sum_naive = np.zeros(ntime, dtype=float)
    count = np.zeros(ntime, dtype=float)

    sse_ridge = 0.0
    sse_naive = 0.0
    n_global = 0

    rmse_ridge_station = np.full(nstation, np.nan)
    rmse_naive_station = np.full(nstation, np.nan)
    r2_station = np.full(nstation, np.nan)
    ntest_station = np.zeros(nstation, dtype=int)

    coefs = None
    if return_coefs:
        coefs = np.full((nstation, n_input), np.nan, dtype=float)

    for st in tqdm(range(nstation), desc="fitting stations"):
        series_st = area.isel(station=st).to_series().sort_index().dropna()

        if len(series_st) == 0 or series_st.nunique() <= 1:
            continue

        try:
            x, y, t0 = make_supervised(series_st, n_input=n_input, n_output=n_output)
        except ValueError:
            continue

        y = y.ravel()

        x_train, x_test, y_train, y_test = train_test_split_time(
            x, y, train_frac=train_frac
        )

        if x_test.shape[0] == 0:
            continue

        model = make_ridge_model(alpha=alpha)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_naive = x_test[:, -1]  # last lag

        sse_ridge += np.sum((y_pred - y_test) ** 2)
        sse_naive += np.sum((y_naive - y_test) ** 2)
        n_global += y_test.size

        rmse_ridge_station[st] = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_naive_station[st] = np.sqrt(mean_squared_error(y_test, y_naive))
        r2_station[st] = r2_score(y_test, y_pred)
        ntest_station[st] = y_test.size

        if return_coefs:
            ridge_est = model.named_steps["ridge"]
            coefs[st, :] = ridge_est.coef_.ravel()

        n_train = x_train.shape[0]
        test_time = t0[n_train:]
        idx = np.array([time_to_idx[t] for t in test_time], dtype=int)

        sum_true[idx] += y_test
        sum_pred[idx] += y_pred
        sum_naive[idx] += y_naive
        count[idx] += 1

    if n_global == 0:
        raise RuntimeError("no test samples over all stations")

    rmse_ridge_global = np.sqrt(sse_ridge / n_global)
    rmse_naive_global = np.sqrt(sse_naive / n_global)

    mean_true = np.full(ntime, np.nan)
    mean_pred = np.full(ntime, np.nan)
    mean_naive = np.full(ntime, np.nan)

    mask = count > 0
    mean_true[mask] = sum_true[mask] / count[mask]
    mean_pred[mask] = sum_pred[mask] / count[mask]
    mean_naive[mask] = sum_naive[mask] / count[mask]

    mean_true_ts = pd.Series(mean_true, index=time_index, name="true")
    mean_pred_ts = pd.Series(mean_pred, index=time_index, name="ridge")
    mean_naive_ts = pd.Series(mean_naive, index=time_index, name="naive")

    station_metrics = pd.DataFrame({
        "station": np.arange(nstation),
        "rmse_ridge": rmse_ridge_station,
        "rmse_naive": rmse_naive_station,
        "r2_ridge": r2_station,
        "n_test": ntest_station,
    })

    if return_coefs:
        return (
            rmse_ridge_global,
            rmse_naive_global,
            mean_true_ts,
            mean_pred_ts,
            mean_naive_ts,
            station_metrics,
            coefs,
        )
    else:
        return (
            rmse_ridge_global,
            rmse_naive_global,
            mean_true_ts,
            mean_pred_ts,
            mean_naive_ts,
            station_metrics,
        )
