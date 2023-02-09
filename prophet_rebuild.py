import numpy as np
import pymc as pm
import pandas as pd
from pytensor import tensor as pt


def prep_raw_data(df):
    """
    Prep the raw data for prophet. Prophet scales it similarly. Prophet does not allow negative values.
    """
    df["ds"] = pd.to_datetime(df["ds"])
    # Scale the data
    df["y_scaled"] = df["y"] / df["y"].max()
    df["t"] = (df["ds"] - df["ds"].min()) / (df["ds"].max() - df["ds"].min())
    return df


def trend_model(
    m,
    t,
    n_changepoints=25,
    changepoints_prior_scale=0.05,
    changepoint_range=0.8,
):
    """
    The piecewise linear trend with changepoint implementation in pymc
    """
    s = np.linspace(0, changepoint_range * np.max(t), n_changepoints + 1)[1:]

    # * 1 casts the boolean to integers
    A = (t[:, None] > s) * 1

    with m:
        # initial growth
        k = pm.Normal("k", 0, 5)
        # initial offset
        m = pm.Normal("m", 0, 5)
        # rate of change
        delta = pm.Laplace("delta", 0, changepoints_prior_scale, shape=n_changepoints)

        growth = k + pt.dot(A, delta)
        offset = m + pt.dot(A, -s * delta)
        trend = growth * t + offset

    return trend, A, s


def seasonal_model(
    m,
    time_column=None,
    time_column_scaled=None,
    period="yearly",
    seasonality_prior_scale=10,
):
    """
    Seasonality model for implementation in pymc
    """
    if period == "yearly":
        # Periodicity rescaled, as t is also scaled
        p = 365.25 / (time_column.max() - time_column.min()).days
        # Order
        n = 10  # order

    elif period == "weekly":
        # Periodicity rescaled, as t is also scaled
        p = 7 / (time_column.max() - time_column.min()).days
        # Order of fourier series
        n = 3

    with m:
        beta = pm.Normal(f"beta_{period}", 0, seasonality_prior_scale, shape=n * 2)
        X = fourier_series(time_column_scaled, p=p, n=n)

    return X, beta


def fourier_series(t, p=365.25, n=10):
    """
    Helper function to generate fourier series.
    """
    # 2 pi n / p
    x = 2 * np.pi * np.arange(1, n + 1) / p
    # 2 pi n / p * t
    x = x * t[:, None]
    x = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    return x


def det_trend(k, m, delta, t, s, A):
    """
    Determine g, based on the aproxed parameters
    """
    return ((k + A @ delta)) * t + (m + A @ (-s * delta))


def det_seasonality(beta, x):
    """
    Determine s based on aproxed parameters
    """
    return x @ beta
