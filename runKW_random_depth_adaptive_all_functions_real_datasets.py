#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified time-series forecasting experiment with fixed-depth and random-depth
fractal activation neural networks.

Methodology preserved from the uploaded OU scripts:
    - chronological train/validation/test split
    - sliding-window one-step training
    - StandardScaler for X and y in neural networks
    - CatBoost baseline
    - one static ReLU neural-network baseline
    - same dense neural-network architecture: Dense(width) x hidden_layers -> Dense(1)
    - either teacher-forced one-step test evaluation or recursive autoregressive rollout

Additions:
    - multiple real time-series datasets
    - fixed-depth fractal activations with depth n_terms=30
    - random-depth fractal activations with min_terms=10, max_terms=50
    - trainable-depth fractal activations with learned integer depth in [1, 30]
    - trainable coefficient-depth fractal activations with one coefficient per term
    - ExtraTreesRegressor with Bayesian hyperparameter optimization
    - one boolean switch at the top to choose autoregressive vs one-step evaluation
    - separate output folders for autoregressive and one-step runs

Expected local file:
    fractal_activation_functions.py

Optional dependency for Bayesian optimization:
    pip install scikit-optimize

If scikit-optimize is not installed, the script falls back to RandomizedSearchCV
while keeping the ExtraTrees optimization stage available.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
#import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers

from catboost import CatBoostRegressor

import fractal_activation_functions as fractal


# ============================================================
# MAIN SWITCH
# ============================================================

# False = normal one-step / teacher-forced test prediction.
# True  = recursive autoregressive rollout over the whole test interval.
AUTOREGRESSIVE_EVALUATION = True


# ============================================================
# CONFIG
# ============================================================

BASE_OUTPUT_DIR = Path("standard_timeseries_fractal_experiments_all_activation_variants_08062026")
OUTPUT_DIR = BASE_OUTPUT_DIR / (
    "autoregressive_results" if AUTOREGRESSIVE_EVALUATION else "onestep_results"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(102, 200))

# ============================================================
# CHRONOLOGICAL DATA SPLIT
# ============================================================
#
# The original time series is divided chronologically:
#
#     |---------- TRAIN ----------|---- VALIDATION ----|---- TEST ----|
#
# No shuffling is performed.
# Future observations are never used during training.
#
# Example:
#   TRAIN_FRACTION = 0.60
#   VAL_FRACTION   = 0.20
#
# gives:
#
#   60% train
#   20% validation
#   20% test
#
# The remaining fraction is always assigned to the test set:
#
#   TEST_FRACTION = 1.0 - TRAIN_FRACTION - VAL_FRACTION
#
# IMPORTANT:
# TRAIN_FRACTION + VAL_FRACTION must be < 1.0
# otherwise no test set remains.
#
# Recommended values:
#
# Small datasets (AirPassengers):
#     TRAIN_FRACTION = 0.60
#     VAL_FRACTION   = 0.20
#     TEST_FRACTION  = 0.20
#
# Medium datasets:
#     TRAIN_FRACTION = 0.70
#     VAL_FRACTION   = 0.15
#     TEST_FRACTION  = 0.15
#
# ============================================================

TRAIN_FRACTION = 0.60
VAL_FRACTION = 0.20

TEST_FRACTION = 1.0 - TRAIN_FRACTION - VAL_FRACTION

assert TEST_FRACTION > 0.0, (
    "TRAIN_FRACTION + VAL_FRACTION must be < 1.0 "
    f"(currently TEST_FRACTION={TEST_FRACTION:.4f})"
)

# Default. Individual datasets can override this below.
WINDOW_SIZE = 13
FORECAST_HORIZON = 1


RANDOM_DEPTH_MIN = 10
RANDOM_DEPTH_MAX = 50
FIXED_DEPTH = 30

# Trainable-depth fractal activations use a differentiable straight-through
# gate over integer depths in [1, 30]. Forward evaluation uses an integer
# rounded depth, while gradients flow through a smooth sigmoid gate.
TRAINABLE_DEPTH_MIN = 1
TRAINABLE_DEPTH_MAX = 30
TRAINABLE_DEPTH_INIT = 30.0
TRAINABLE_DEPTH_GATE_SHARPNESS = 10.0
INCLUDE_TRAINABLE_DEPTH_ACTIVATIONS = True

# Trainable coefficient-depth fractal activations use all terms up to depth 30.
# Each term receives one trainable scalar coefficient initialized to 1.0.
# This lets the network learn term-wise importance while preserving the same
# activation families used in the mathematical diagnostic script.
TRAINABLE_COEFFICIENT_DEPTH_MAX = 30
TRAINABLE_COEFFICIENT_INIT = 1.0
INCLUDE_TRAINABLE_COEFFICIENT_ACTIVATIONS = True

EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

NN_WIDTH = 64
NN_DEPTH = 2

PLOT_LAST_N = 500
SHOW_PLOTS = False
SAVE_PLOTS = True

# Bayesian optimization stage for ExtraTrees.
EXTRATREES_BAYES_ITER = 30
EXTRATREES_CV_SPLITS = 3


# ============================================================
# REAL DATA FROM THE UPLOADED DATASET SCRIPT
# ============================================================

AIR_PASSENGERS = [
    112,118,132,129,121,135,148,148,136,119,104,118,
    115,126,141,135,125,149,170,170,158,133,114,140,
    145,150,178,163,172,178,199,199,184,162,146,166,
    171,180,193,181,183,218,230,242,209,191,172,194,
    196,196,236,235,229,243,264,272,237,211,180,201,
    204,188,235,227,234,264,302,293,259,229,203,229,
    242,233,267,269,270,315,364,347,312,274,237,278,
    284,277,317,313,318,374,413,405,355,306,271,306,
    315,301,356,348,355,422,465,467,404,347,305,336,
    340,318,362,348,363,435,491,505,404,359,310,337,
    360,342,406,396,420,472,548,559,463,407,362,405,
    417,391,419,461,472,535,622,606,508,461,390,432,
]


def load_airpassengers() -> Tuple[pd.DatetimeIndex, np.ndarray]:
    values = np.asarray(AIR_PASSENGERS, dtype=np.float32)
    dates = pd.date_range("1949-01-01", periods=len(values), freq="MS")
    return dates, values


def load_monthly_mean_temperature_nottingham() -> Tuple[pd.DatetimeIndex, np.ndarray]:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-mean-temp.csv"
    df = pd.read_csv(url)
    dates = pd.to_datetime(df.iloc[:, 0])
    values = df.iloc[:, 1].astype(np.float32).to_numpy()
    return pd.DatetimeIndex(dates), values


def load_perrin_freres_champagne_sales() -> Tuple[pd.DatetimeIndex, np.ndarray]:
    url = (
        "https://raw.githubusercontent.com/krishnaik06/ARIMA-And-Seasonal-ARIMA/"
        "master/perrin-freres-monthly-champagne-.csv"
    )
    df = pd.read_csv(url).dropna()
    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    values = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    keep = dates.notna() & values.notna()
    dates = dates[keep]
    values = values[keep].astype(np.float32).to_numpy()
    return pd.DatetimeIndex(dates), values


def load_monthly_car_sales_quebec() -> Tuple[pd.DatetimeIndex, np.ndarray]:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv"
    df = pd.read_csv(url)
    dates = pd.to_datetime(df.iloc[:, 0])
    values = df.iloc[:, 1].astype(np.float32).to_numpy()
    return pd.DatetimeIndex(dates), values


def load_yearly_sunspots() -> Tuple[pd.DatetimeIndex, np.ndarray]:
    url = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/sunspot.year.csv"
    df = pd.read_csv(url)
    if "time" in df.columns and "value" in df.columns:
        years = df["time"].astype(int).to_numpy()
        values = df["value"].astype(np.float32).to_numpy()
    else:
        years = df.iloc[:, 1].astype(int).to_numpy()
        values = df.iloc[:, 2].astype(np.float32).to_numpy()
    dates = pd.to_datetime([f"{year}-01-01" for year in years])
    return pd.DatetimeIndex(dates), values



def load_daily_min_temperatures_melbourne() -> Tuple[pd.DatetimeIndex, np.ndarray]:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    df = pd.read_csv(url)
    dates = pd.to_datetime(df.iloc[:, 0])
    values = df.iloc[:, 1].astype(np.float32).to_numpy()
    return pd.DatetimeIndex(dates), values


def load_monthly_shampoo_sales() -> Tuple[pd.DatetimeIndex, np.ndarray]:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv"
    df = pd.read_csv(url)
    values = pd.to_numeric(df.iloc[:, 1], errors="coerce").astype(np.float32).to_numpy()
    dates = pd.date_range("1901-01-01", periods=len(values), freq="MS")
    return dates, values


def load_daily_total_female_births_california() -> Tuple[pd.DatetimeIndex, np.ndarray]:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv"
    df = pd.read_csv(url)
    dates = pd.to_datetime(df.iloc[:, 0])
    values = df.iloc[:, 1].astype(np.float32).to_numpy()
    return pd.DatetimeIndex(dates), values


def load_monthly_robberies_boston() -> Tuple[pd.DatetimeIndex, np.ndarray]:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-robberies.csv"
    df = pd.read_csv(url)
    dates = pd.to_datetime(df.iloc[:, 0])
    values = df.iloc[:, 1].astype(np.float32).to_numpy()
    return pd.DatetimeIndex(dates), values


def load_lynx_trappings_canada() -> Tuple[pd.DatetimeIndex, np.ndarray]:
    url = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/lynx.csv"
    df = pd.read_csv(url)
    if "time" in df.columns and "value" in df.columns:
        years = df["time"].astype(int).to_numpy()
        values = df["value"].astype(np.float32).to_numpy()
    else:
        years = df.iloc[:, 1].astype(int).to_numpy()
        values = df.iloc[:, 2].astype(np.float32).to_numpy()
    dates = pd.to_datetime([f"{year}-01-01" for year in years])
    return pd.DatetimeIndex(dates), values



# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# DATA GENERATION AND WINDOWING
# ============================================================

def build_window_dataset(series: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def chronological_split_windows(
    X: np.ndarray,
    y: np.ndarray,
    train_fraction: float,
    val_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    n_train = int(train_fraction * n)
    n_val = int(val_fraction * n)

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def chronological_split_series(
    series: np.ndarray,
    train_fraction: float,
    val_fraction: float,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(series)
    n_train = int(train_fraction * n)
    n_val = int(val_fraction * n)

    train_series = series[:n_train]
    val_series = series[n_train - window_size:n_train + n_val]
    test_series = series[n_train + n_val - window_size:]

    return train_series, val_series, test_series


# ============================================================
# FIXED-DEPTH FRACTAL ACTIVATIONS
# ============================================================

# These wrappers force n_terms/num_terms = 30 while keeping the same activation
# family names and neural-network architecture.

def fixed_depth_modified_weierstrass_function_tanh(x):
    return fractal.modified_weierstrass_function_tanh(x, n_terms=FIXED_DEPTH)


def fixed_depth_decaying_cosine_function_tf(x):
    return fractal.decaying_cosine_function_tf(x, n_terms=FIXED_DEPTH)


def fixed_depth_modulated_blancmange_curve(x):
    return fractal.modulated_blancmange_curve(x, n_terms=FIXED_DEPTH)


def fixed_depth_weierstrass_mandelbrot_function_xpsin(x):
    return fractal.weierstrass_mandelbrot_function_xpsin(x, num_terms=FIXED_DEPTH)


def fixed_depth_weierstrass_mandelbrot_function_tanhpsin(x):
    return fractal.weierstrass_mandelbrot_function_tanhpsin(x, num_terms=FIXED_DEPTH)


def random_depth_modified_weierstrass_function_tanh(x):
    return fractal.random_depth_modified_weierstrass_function_tanh(
        x,
        min_terms=RANDOM_DEPTH_MIN,
        max_terms=RANDOM_DEPTH_MAX,
    )


def random_depth_decaying_cosine_function_tf(x):
    return fractal.random_depth_decaying_cosine_function_tf(
        x,
        min_terms=RANDOM_DEPTH_MIN,
        max_terms=RANDOM_DEPTH_MAX,
    )


def random_depth_modulated_blancmange_curve(x):
    return fractal.random_depth_modulated_blancmange_curve(
        x,
        min_terms=RANDOM_DEPTH_MIN,
        max_terms=RANDOM_DEPTH_MAX,
    )


def random_depth_weierstrass_mandelbrot_function_xpsin(x):
    return fractal.random_depth_weierstrass_mandelbrot_function_xpsin(
        x,
        min_terms=RANDOM_DEPTH_MIN,
        max_terms=RANDOM_DEPTH_MAX,
    )


def random_depth_weierstrass_mandelbrot_function_tanhpsin(x):
    return fractal.random_depth_weierstrass_mandelbrot_function_tanhpsin(
        x,
        min_terms=RANDOM_DEPTH_MIN,
        max_terms=RANDOM_DEPTH_MAX,
    )


# ============================================================
# TRAINABLE-DEPTH FRACTAL ACTIVATION LAYERS
# ============================================================
#
# Important methodological note:
# A strictly discrete integer depth cannot be optimized directly by standard
# gradient descent, because rounding and integer selection are not
# differentiable. The layer below therefore uses a straight-through estimator:
#
#   - forward pass: uses a hard integer effective depth in [1, 30]
#   - backward pass: sends gradients through a smooth sigmoid gate
#
# This means that the depth is adapted together with the neural-network
# weights while still behaving like an integer-truncated fractal series during
# the forward computation.

class TrainableDepthFractalActivation(tf.keras.layers.Layer):
    def __init__(
        self,
        family: str,
        min_depth: int = TRAINABLE_DEPTH_MIN,
        max_depth: int = TRAINABLE_DEPTH_MAX,
        init_depth: float = TRAINABLE_DEPTH_INIT,
        gate_sharpness: float = TRAINABLE_DEPTH_GATE_SHARPNESS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.family = str(family)
        self.min_depth = int(min_depth)
        self.max_depth = int(max_depth)
        self.init_depth = float(init_depth)
        self.gate_sharpness = float(gate_sharpness)

        if self.min_depth < 1:
            raise ValueError("min_depth must be >= 1.")
        if self.max_depth <= self.min_depth:
            raise ValueError("max_depth must be larger than min_depth.")
        if not (self.min_depth <= self.init_depth <= self.max_depth):
            raise ValueError("init_depth must lie inside [min_depth, max_depth].")

    def build(self, input_shape):
        ratio = (self.init_depth - self.min_depth) / (self.max_depth - self.min_depth)
        ratio = np.clip(ratio, 1e-4, 1.0 - 1e-4)
        initial_logit = np.log(ratio / (1.0 - ratio)).astype(np.float32)

        self.depth_logit = self.add_weight(
            name="depth_logit",
            shape=(),
            initializer=tf.keras.initializers.Constant(initial_logit),
            trainable=True,
        )
        super().build(input_shape)

    def continuous_depth(self) -> tf.Tensor:
        return (
            float(self.min_depth)
            + tf.sigmoid(self.depth_logit) * float(self.max_depth - self.min_depth)
        )

    def rounded_depth(self) -> tf.Tensor:
        return tf.cast(tf.round(self.continuous_depth()), tf.int32)

    def _gate(self, term_index: int) -> tf.Tensor:
        k = tf.cast(term_index, tf.float32)
        depth_cont = self.continuous_depth()
        depth_int = tf.cast(self.rounded_depth(), tf.float32)

        soft_gate = tf.sigmoid(self.gate_sharpness * (depth_cont - k))
        hard_gate = tf.cast(k <= depth_int, tf.float32)

        # Straight-through estimator:
        #   forward  -> hard integer gate
        #   backward -> smooth sigmoid gate
        return soft_gate + tf.stop_gradient(hard_gate - soft_gate)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "family": self.family,
                "min_depth": self.min_depth,
                "max_depth": self.max_depth,
                "init_depth": self.init_depth,
                "gate_sharpness": self.gate_sharpness,
            }
        )
        return cfg

    def call(self, x):
        if self.family == "modified_weierstrass_tanh":
            return self._modified_weierstrass_tanh(x)
        if self.family == "decaying_cosine":
            return self._decaying_cosine(x)
        if self.family == "modulated_blancmange":
            return self._modulated_blancmange(x)
        if self.family == "wm_xpsin":
            return self._wm_xpsin(x)
        if self.family == "wm_tanhpsin":
            return self._wm_tanhpsin(x)
        raise ValueError(f"Unknown trainable fractal activation family: {self.family}")

    def _modified_weierstrass_tanh(self, x, a=0.5, b=1.5):
        x64 = tf.cast(x, tf.float64)
        w = tf.zeros_like(x64, dtype=tf.float64)

        for n in range(self.max_depth):
            gate = tf.cast(self._gate(n + 1), tf.float64)
            term = ((-1) ** n) * (a ** n) * tf.cos((b ** n) * np.pi * x64)
            w += gate * term

        y = w * tf.exp(-0.75 * tf.abs(x64)) + tf.tanh(x64)
        return tf.cast(y, tf.float32)

    def _decaying_cosine(self, x, c=0.5, d=2, zeta=0.2666):
        x32 = tf.cast(x, tf.float32)
        w = tf.zeros_like(x32, dtype=tf.float32)

        mirrored = tf.where(x32 < 0.0, tf.ones_like(x32), -tf.ones_like(x32))
        decay = tf.exp(-tf.abs(x32) * 0.5)

        for n in range(self.max_depth):
            gate = self._gate(n + 1)
            term = zeta * (
                0.05 * tf.tanh(np.pi * x32)
                + (c ** n) * tf.cos((d ** n) * np.pi * x32) * decay * mirrored
            )
            w += gate * term

        return tf.cast(w, tf.float32)

    def _modulated_blancmange(self, x, a=0.75):
        x32 = tf.cast(x, tf.float32)
        y = tf.zeros_like(x32, dtype=tf.float32)

        for n in range(self.max_depth):
            gate = self._gate(n + 1)
            factor = 2 ** n
            modulation = tf.tanh(a * factor * x32)
            ax = a * tf.sqrt(tf.abs(x32) + 1e-8)
            term = modulation * tf.abs(x32 * factor % 2 - 1 * ax) / factor
            y += gate * term

        return y / 2.0

    def _wm_xpsin(self, x, gamma=0.5, lambda_val=2):
        x64 = tf.cast(x, tf.float64)
        m_x = tf.zeros_like(x64, dtype=tf.float64)

        for k in range(1, self.max_depth + 1):
            gate = tf.cast(self._gate(k), tf.float64)
            term = (2 ** (-k * gamma)) * (
                x64 + tf.sin(2 * np.pi * (lambda_val ** k) * x64)
            )
            m_x += gate * term

        return tf.cast(m_x, tf.float32)

    def _wm_tanhpsin(self, x, gamma=0.5, lambda_val=2):
        x64 = tf.cast(x, tf.float64)
        m_x = tf.zeros_like(x64, dtype=tf.float64)

        for k in range(1, self.max_depth + 1):
            gate = tf.cast(self._gate(k), tf.float64)
            term = (2 ** (-k * gamma)) * (
                tf.tanh(x64) + tf.sin(2 * np.pi * (lambda_val ** k) * x64)
            )
            m_x += gate * term

        return tf.cast(m_x, tf.float32)


class TrainableDepthFractalActivationSpec:
    def __init__(self, family: str):
        self.family = str(family)

    def make_layer(self, layer_id: int) -> TrainableDepthFractalActivation:
        return TrainableDepthFractalActivation(
            family=self.family,
            min_depth=TRAINABLE_DEPTH_MIN,
            max_depth=TRAINABLE_DEPTH_MAX,
            init_depth=TRAINABLE_DEPTH_INIT,
            gate_sharpness=TRAINABLE_DEPTH_GATE_SHARPNESS,
            name=f"trainable_depth_{self.family}_hidden_{layer_id}",
        )


def extract_trainable_depth_summary(model: tf.keras.Model) -> str:
    summaries = []
    for layer in model.layers:
        if isinstance(layer, TrainableDepthFractalActivation):
            cont = float(layer.continuous_depth().numpy())
            rounded = int(layer.rounded_depth().numpy())
            summaries.append(
                f"{layer.name}:continuous={cont:.3f},integer={rounded}"
            )
    return "; ".join(summaries)


# ============================================================
# TRAINABLE COEFFICIENT-DEPTH FRACTAL ACTIVATION LAYERS
# ============================================================
#
# This variant matches the coefficient activation used in the mathematical
# diagnostic script. It does not learn one integer cutoff depth. Instead, all
# terms up to depth 30 are present, and every term gets one trainable scalar
# coefficient. With all coefficients equal to 1.0, this starts close to the
# corresponding fixed-depth internal implementation.

class TrainableCoefficientFractalActivation(tf.keras.layers.Layer):
    def __init__(
        self,
        family: str,
        max_depth: int = TRAINABLE_COEFFICIENT_DEPTH_MAX,
        coefficient_init: float = TRAINABLE_COEFFICIENT_INIT,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.family = str(family)
        self.max_depth = int(max_depth)
        self.coefficient_init = float(coefficient_init)

        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1.")

    def build(self, input_shape):
        self.term_coefficients = self.add_weight(
            name="term_coefficients",
            shape=(self.max_depth,),
            initializer=tf.keras.initializers.Constant(self.coefficient_init),
            trainable=True,
            dtype=tf.float32,
        )
        super().build(input_shape)

    def coefficient_vector(self) -> np.ndarray:
        return np.asarray(self.term_coefficients.numpy(), dtype=np.float32)

    def _coeff32(self, index: int) -> tf.Tensor:
        return tf.cast(self.term_coefficients[index], tf.float32)

    def _coeff64(self, index: int) -> tf.Tensor:
        return tf.cast(self.term_coefficients[index], tf.float64)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "family": self.family,
                "max_depth": self.max_depth,
                "coefficient_init": self.coefficient_init,
            }
        )
        return cfg

    def call(self, x):
        if self.family == "modified_weierstrass_tanh":
            return self._modified_weierstrass_tanh(x)
        if self.family == "decaying_cosine":
            return self._decaying_cosine(x)
        if self.family == "modulated_blancmange":
            return self._modulated_blancmange(x)
        if self.family == "wm_xpsin":
            return self._wm_xpsin(x)
        if self.family == "wm_tanhpsin":
            return self._wm_tanhpsin(x)
        raise ValueError(f"Unknown trainable coefficient fractal activation family: {self.family}")

    def _modified_weierstrass_tanh(self, x, a=0.5, b=1.5):
        x64 = tf.cast(x, tf.float64)
        w = tf.zeros_like(x64, dtype=tf.float64)

        for n in range(self.max_depth):
            coeff = self._coeff64(n)
            term = ((-1) ** n) * (a ** n) * tf.cos((b ** n) * np.pi * x64)
            w += coeff * term

        y = w * tf.exp(-0.75 * tf.abs(x64)) + tf.tanh(x64)
        return tf.cast(y, tf.float32)

    def _decaying_cosine(self, x, c=0.5, d=2, zeta=0.2666):
        x32 = tf.cast(x, tf.float32)
        w = tf.zeros_like(x32, dtype=tf.float32)

        mirrored = tf.where(x32 < 0.0, tf.ones_like(x32), -tf.ones_like(x32))
        decay = tf.exp(-tf.abs(x32) * 0.5)

        for n in range(self.max_depth):
            coeff = self._coeff32(n)
            term = zeta * (
                0.05 * tf.tanh(np.pi * x32)
                + (c ** n) * tf.cos((d ** n) * np.pi * x32) * decay * mirrored
            )
            w += coeff * term

        return tf.cast(w, tf.float32)

    def _modulated_blancmange(self, x, a=0.75):
        x32 = tf.cast(x, tf.float32)
        y = tf.zeros_like(x32, dtype=tf.float32)

        for n in range(self.max_depth):
            coeff = self._coeff32(n)
            factor = 2 ** n
            modulation = tf.tanh(a * factor * x32)
            ax = a * tf.sqrt(tf.abs(x32) + 1e-8)
            term = modulation * tf.abs(x32 * factor % 2 - 1 * ax) / factor
            y += coeff * term

        return y / 2.0

    def _wm_xpsin(self, x, gamma=0.5, lambda_val=2):
        x64 = tf.cast(x, tf.float64)
        m_x = tf.zeros_like(x64, dtype=tf.float64)

        for k in range(1, self.max_depth + 1):
            coeff = self._coeff64(k - 1)
            term = (2 ** (-k * gamma)) * (
                x64 + tf.sin(2 * np.pi * (lambda_val ** k) * x64)
            )
            m_x += coeff * term

        return tf.cast(m_x, tf.float32)

    def _wm_tanhpsin(self, x, gamma=0.5, lambda_val=2):
        x64 = tf.cast(x, tf.float64)
        m_x = tf.zeros_like(x64, dtype=tf.float64)

        for k in range(1, self.max_depth + 1):
            coeff = self._coeff64(k - 1)
            term = (2 ** (-k * gamma)) * (
                tf.tanh(x64) + tf.sin(2 * np.pi * (lambda_val ** k) * x64)
            )
            m_x += coeff * term

        return tf.cast(m_x, tf.float32)


class TrainableCoefficientFractalActivationSpec:
    def __init__(self, family: str):
        self.family = str(family)

    def make_layer(self, layer_id: int) -> TrainableCoefficientFractalActivation:
        return TrainableCoefficientFractalActivation(
            family=self.family,
            max_depth=TRAINABLE_COEFFICIENT_DEPTH_MAX,
            coefficient_init=TRAINABLE_COEFFICIENT_INIT,
            name=f"trainable_coefficient_{self.family}_hidden_{layer_id}",
        )


def extract_trainable_coefficient_summary(model: tf.keras.Model) -> str:
    summaries = []
    for layer in model.layers:
        if isinstance(layer, TrainableCoefficientFractalActivation):
            coeffs = layer.coefficient_vector()
            coeff_items = ",".join(
                f"d{i + 1}={coeffs[i]:.5f}" for i in range(len(coeffs))
            )
            summaries.append(f"{layer.name}:{coeff_items}")
    return "; ".join(summaries)


def extract_trainable_parameter_summary(model: tf.keras.Model) -> str:
    parts = []
    depth_summary = extract_trainable_depth_summary(model)
    coefficient_summary = extract_trainable_coefficient_summary(model)
    if depth_summary:
        parts.append(depth_summary)
    if coefficient_summary:
        parts.append(coefficient_summary)
    return " | ".join(parts)


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_nn_model(
    input_dim: int,
    activation: Callable | str,
    width: int = 64,
    hidden_layers: int = 2,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    if isinstance(activation, (TrainableDepthFractalActivationSpec, TrainableCoefficientFractalActivationSpec)):
        for layer_id in range(hidden_layers):
            model.add(layers.Dense(width))
            model.add(activation.make_layer(layer_id + 1))
    else:
        for _ in range(hidden_layers):
            model.add(layers.Dense(width, activation=activation))

    model.add(layers.Dense(1))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model


def train_nn(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tf.keras.Model:
    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[es],
    )

    return model


def train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> CatBoostRegressor:
    model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.03,
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    return model


def train_extratrees_bayes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> ExtraTreesRegressor:
    cv = TimeSeriesSplit(n_splits=EXTRATREES_CV_SPLITS)

    base_model = ExtraTreesRegressor(
        random_state=seed,
        n_jobs=-1,
    )

    try:
        from skopt import BayesSearchCV
        from skopt.space import Categorical, Integer, Real

        search_spaces = {
            "n_estimators": Integer(100, 900),
            "max_depth": Integer(2, 30),
            "min_samples_split": Integer(2, 20),
            "min_samples_leaf": Integer(1, 12),
            "max_features": Categorical(["sqrt", "log2", 0.5, 0.75, 1.0]),
            "bootstrap": Categorical([False, True]),
        }

        search = BayesSearchCV(
            estimator=base_model,
            search_spaces=search_spaces,
            n_iter=EXTRATREES_BAYES_ITER,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            random_state=seed,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
    except Exception as exc:
        print(
            "scikit-optimize unavailable or failed to import. "
            f"Falling back to RandomizedSearchCV: {type(exc).__name__}: {exc}"
        )
        param_distributions = {
            "n_estimators": [100, 200, 300, 500, 700, 900],
            "max_depth": [2, 4, 6, 8, 12, 16, 20, 30, None],
            "min_samples_split": [2, 4, 6, 8, 12, 16, 20],
            "min_samples_leaf": [1, 2, 3, 4, 6, 8, 12],
            "max_features": ["sqrt", "log2", 0.5, 0.75, 1.0],
            "bootstrap": [False, True],
        }

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=EXTRATREES_BAYES_ITER,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            random_state=seed,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )

    search.fit(X_train, y_train)
    print(f"ExtraTrees optimized best params: {search.best_params_}")
    return search.best_estimator_


# ============================================================
# PREDICTION
# ============================================================

def predict_onestep_catboost(model, X_test_s: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X_test_s), dtype=np.float32).reshape(-1)


def predict_onestep_extratrees(model, X_test_s: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X_test_s), dtype=np.float32).reshape(-1)


def predict_onestep_nn(
    model: tf.keras.Model,
    X_test_s: np.ndarray,
    scaler_y: StandardScaler,
) -> np.ndarray:
    pred_s = model.predict(X_test_s, verbose=0).reshape(-1)
    return scaler_y.inverse_transform(pred_s.reshape(-1, 1)).reshape(-1).astype(np.float32)


def autoregressive_predict_raw_output_model(
    model,
    initial_window_raw: np.ndarray,
    steps: int,
    scaler_X: StandardScaler,
) -> np.ndarray:
    window = initial_window_raw.astype(np.float32).copy()
    preds = []

    for _ in range(steps):
        window_scaled = scaler_X.transform(window.reshape(1, -1))
        pred = float(model.predict(window_scaled)[0])
        preds.append(pred)
        window = np.roll(window, -1)
        window[-1] = pred

    return np.asarray(preds, dtype=np.float32)


def autoregressive_predict_nn(
    model: tf.keras.Model,
    initial_window_raw: np.ndarray,
    steps: int,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
) -> np.ndarray:
    window = initial_window_raw.astype(np.float32).copy()
    preds = []

    for _ in range(steps):
        window_scaled = scaler_X.transform(window.reshape(1, -1))
        pred_scaled = model.predict(window_scaled, verbose=0).reshape(-1)[0]
        pred = scaler_y.inverse_transform([[pred_scaled]])[0, 0]
        preds.append(float(pred))
        window = np.roll(window, -1)
        window[-1] = float(pred)

    return np.asarray(preds, dtype=np.float32)


# ============================================================
# METRICS, PLOTTING, SAVING
# ============================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_percent": float(
            np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100.0
        ),
    }


def safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


def plot_seed_results(
    dataset_name: str,
    seed: int,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    metrics: Dict[str, Dict[str, float]],
    out_path: Path,
) -> None:
    n = min(PLOT_LAST_N, len(y_true))
    idx = np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ax.plot(idx, y_true[:n], label="true target", linewidth=2.2)

    for name, pred in predictions.items():
        pred = np.asarray(pred).reshape(-1)
        mode_label = "AR-RMSE" if AUTOREGRESSIVE_EVALUATION else "RMSE"
        label = f"{name} | {mode_label}={metrics[name]['rmse']:.4f}"
        ax.plot(idx, pred[:n], label=label, linewidth=1.4)

    mode_title = "autoregressive rollout" if AUTOREGRESSIVE_EVALUATION else "one-step forecasting"
    ax.set_title(f"{dataset_name} | {mode_title} | seed={seed}")
    ax.set_xlabel("test step")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    if SAVE_PLOTS:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show(block=True)
    plt.close(fig)


def save_forecast_table(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    df = pd.DataFrame({"test_index": np.arange(len(y_true)), "actual": y_true})
    for name, pred in predictions.items():
        df[name] = np.asarray(pred).reshape(-1)
    df.to_csv(out_path, index=False)


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def get_activation_sets() -> Dict[str, Callable]:
    fixed_depth_activations = {
        f"FixedDepth{FIXED_DEPTH}_ModWeiTanh": fixed_depth_modified_weierstrass_function_tanh,
        f"FixedDepth{FIXED_DEPTH}_DecCos": fixed_depth_decaying_cosine_function_tf,
        f"FixedDepth{FIXED_DEPTH}_Blancmange": fixed_depth_modulated_blancmange_curve,
        f"FixedDepth{FIXED_DEPTH}_WM_xpsin": fixed_depth_weierstrass_mandelbrot_function_xpsin,
        f"FixedDepth{FIXED_DEPTH}_WM_tanhpsin": fixed_depth_weierstrass_mandelbrot_function_tanhpsin,
    }

    random_depth_activations = {
        f"RandDepth{RANDOM_DEPTH_MIN}to{RANDOM_DEPTH_MAX}_ModWeiTanh": random_depth_modified_weierstrass_function_tanh,
        f"RandDepth{RANDOM_DEPTH_MIN}to{RANDOM_DEPTH_MAX}_DecCos": random_depth_decaying_cosine_function_tf,
        f"RandDepth{RANDOM_DEPTH_MIN}to{RANDOM_DEPTH_MAX}_Blancmange": random_depth_modulated_blancmange_curve,
        f"RandDepth{RANDOM_DEPTH_MIN}to{RANDOM_DEPTH_MAX}_WM_xpsin": random_depth_weierstrass_mandelbrot_function_xpsin,
        f"RandDepth{RANDOM_DEPTH_MIN}to{RANDOM_DEPTH_MAX}_WM_tanhpsin": random_depth_weierstrass_mandelbrot_function_tanhpsin,
    }

    trainable_depth_activations = {
        f"TrainDepth{TRAINABLE_DEPTH_MIN}to{TRAINABLE_DEPTH_MAX}_ModWeiTanh": TrainableDepthFractalActivationSpec("modified_weierstrass_tanh"),
        f"TrainDepth{TRAINABLE_DEPTH_MIN}to{TRAINABLE_DEPTH_MAX}_DecCos": TrainableDepthFractalActivationSpec("decaying_cosine"),
        f"TrainDepth{TRAINABLE_DEPTH_MIN}to{TRAINABLE_DEPTH_MAX}_Blancmange": TrainableDepthFractalActivationSpec("modulated_blancmange"),
        f"TrainDepth{TRAINABLE_DEPTH_MIN}to{TRAINABLE_DEPTH_MAX}_WM_xpsin": TrainableDepthFractalActivationSpec("wm_xpsin"),
        f"TrainDepth{TRAINABLE_DEPTH_MIN}to{TRAINABLE_DEPTH_MAX}_WM_tanhpsin": TrainableDepthFractalActivationSpec("wm_tanhpsin"),
    }

    trainable_coefficient_activations = {
        f"TrainCoeffDepth{TRAINABLE_COEFFICIENT_DEPTH_MAX}_ModWeiTanh": TrainableCoefficientFractalActivationSpec("modified_weierstrass_tanh"),
        f"TrainCoeffDepth{TRAINABLE_COEFFICIENT_DEPTH_MAX}_DecCos": TrainableCoefficientFractalActivationSpec("decaying_cosine"),
        f"TrainCoeffDepth{TRAINABLE_COEFFICIENT_DEPTH_MAX}_Blancmange": TrainableCoefficientFractalActivationSpec("modulated_blancmange"),
        f"TrainCoeffDepth{TRAINABLE_COEFFICIENT_DEPTH_MAX}_WM_xpsin": TrainableCoefficientFractalActivationSpec("wm_xpsin"),
        f"TrainCoeffDepth{TRAINABLE_COEFFICIENT_DEPTH_MAX}_WM_tanhpsin": TrainableCoefficientFractalActivationSpec("wm_tanhpsin"),
    }

    activation_sets = {
        **fixed_depth_activations,
        **random_depth_activations,
    }

    if INCLUDE_TRAINABLE_DEPTH_ACTIVATIONS:
        activation_sets.update(trainable_depth_activations)

    if INCLUDE_TRAINABLE_COEFFICIENT_ACTIVATIONS:
        activation_sets.update(trainable_coefficient_activations)

    return activation_sets


def prepare_dataset_for_seed(dataset_spec: Dict, seed: int) -> Tuple[pd.Index, np.ndarray]:
    dates, values = dataset_spec["loader"]()
    return dates, values.astype(np.float32)

def run_one_dataset_seed(dataset_spec: Dict, seed: int) -> pd.DataFrame:
    global BATCH_SIZE, EPOCHS
    dataset_name = dataset_spec["name"]
    window_size = int(dataset_spec.get("window_size", WINDOW_SIZE))
    batch_size = int(dataset_spec.get("batch_size", BATCH_SIZE))
    epochs = int(dataset_spec.get("epochs", EPOCHS))

    old_batch_size = BATCH_SIZE
    old_epochs = EPOCHS
    BATCH_SIZE = batch_size
    EPOCHS = epochs

    try:
        print(f"\n[{dataset_name} | seed {seed}] loading/generating data")
        set_seed(seed)

        _, series = prepare_dataset_for_seed(dataset_spec, seed)
        series = np.asarray(series, dtype=np.float32).reshape(-1)

        dataset_dir = OUTPUT_DIR / safe_name(dataset_name)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame({"index": np.arange(len(series)), "value": series}).to_csv(
            dataset_dir / f"{safe_name(dataset_name)}_raw_series_seed_{seed}.csv",
            index=False,
        )

        predictions: Dict[str, np.ndarray] = {}
        metrics: Dict[str, Dict[str, float]] = {}
        trainable_parameter_summaries: Dict[str, str] = {}

        if AUTOREGRESSIVE_EVALUATION:
            train_series, val_series, test_series = chronological_split_series(
                series,
                train_fraction=TRAIN_FRACTION,
                val_fraction=VAL_FRACTION,
                window_size=window_size,
            )

            X_train, y_train = build_window_dataset(train_series, window_size)
            X_val, y_val = build_window_dataset(val_series, window_size)

            initial_test_window = test_series[:window_size]
            y_test = test_series[window_size:]
            X_test_s = None
        else:
            X, y = build_window_dataset(series, window_size)
            X_train, y_train, X_val, y_val, X_test, y_test = chronological_split_windows(
                X,
                y,
                train_fraction=TRAIN_FRACTION,
                val_fraction=VAL_FRACTION,
            )
            initial_test_window = None

        if len(X_train) < 5 or len(X_val) < 2 or len(y_test) < 1:
            raise ValueError(
                f"Too few windows for {dataset_name}. Reduce window_size={window_size}."
            )

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_s = scaler_X.fit_transform(X_train)
        X_val_s = scaler_X.transform(X_val)
        if not AUTOREGRESSIVE_EVALUATION:
            X_test_s = scaler_X.transform(X_test)

        y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
        y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).reshape(-1)

        # --------------------------------------------------------
        # CatBoost baseline
        # --------------------------------------------------------
        print(f"[{dataset_name} | seed {seed}] training CatBoostRegressor")
        cat_model = train_catboost(X_train_s, y_train, X_val_s, y_val, seed)

        if AUTOREGRESSIVE_EVALUATION:
            cat_pred = autoregressive_predict_raw_output_model(
                cat_model,
                initial_test_window,
                len(y_test),
                scaler_X,
            )
            model_name = "CatBoost_AR"
        else:
            cat_pred = predict_onestep_catboost(cat_model, X_test_s)
            model_name = "CatBoost"

        predictions[model_name] = cat_pred
        metrics[model_name] = regression_metrics(y_test, cat_pred)

        # --------------------------------------------------------
        # ExtraTrees with Bayesian optimization
        # --------------------------------------------------------
        print(f"[{dataset_name} | seed {seed}] training ExtraTrees_BayesOpt")
        et_model = train_extratrees_bayes(X_train_s, y_train, seed)

        if AUTOREGRESSIVE_EVALUATION:
            et_pred = autoregressive_predict_raw_output_model(
                et_model,
                initial_test_window,
                len(y_test),
                scaler_X,
            )
            model_name = "ExtraTrees_BayesOpt_AR"
        else:
            et_pred = predict_onestep_extratrees(et_model, X_test_s)
            model_name = "ExtraTrees_BayesOpt"

        predictions[model_name] = et_pred
        metrics[model_name] = regression_metrics(y_test, et_pred)

        # --------------------------------------------------------
        # ReLU neural-network baseline
        # --------------------------------------------------------
        print(f"[{dataset_name} | seed {seed}] training NN_ReLU")
        relu_model = build_nn_model(
            input_dim=window_size,
            activation="relu",
            width=NN_WIDTH,
            hidden_layers=NN_DEPTH,
            learning_rate=LEARNING_RATE,
        )
        relu_model = train_nn(relu_model, X_train_s, y_train_s, X_val_s, y_val_s)

        if AUTOREGRESSIVE_EVALUATION:
            relu_pred = autoregressive_predict_nn(
                relu_model,
                initial_test_window,
                len(y_test),
                scaler_X,
                scaler_y,
            )
            model_name = "NN_ReLU_AR"
        else:
            relu_pred = predict_onestep_nn(relu_model, X_test_s, scaler_y)
            model_name = "NN_ReLU"

        predictions[model_name] = relu_pred
        metrics[model_name] = regression_metrics(y_test, relu_pred)

        # --------------------------------------------------------
        # Fixed-depth and random-depth fractal neural networks
        # --------------------------------------------------------
        for activation_name, activation_fn in get_activation_sets().items():
            suffix = "_AR" if AUTOREGRESSIVE_EVALUATION else ""
            full_name = f"{activation_name}{suffix}"
            print(f"[{dataset_name} | seed {seed}] training {full_name}")

            model = build_nn_model(
                input_dim=window_size,
                activation=activation_fn,
                width=NN_WIDTH,
                hidden_layers=NN_DEPTH,
                learning_rate=LEARNING_RATE,
            )
            model = train_nn(model, X_train_s, y_train_s, X_val_s, y_val_s)

            if AUTOREGRESSIVE_EVALUATION:
                pred = autoregressive_predict_nn(
                    model,
                    initial_test_window,
                    len(y_test),
                    scaler_X,
                    scaler_y,
                )
            else:
                pred = predict_onestep_nn(model, X_test_s, scaler_y)

            predictions[full_name] = pred
            metrics[full_name] = regression_metrics(y_test, pred)

            parameter_summary = extract_trainable_parameter_summary(model)
            if parameter_summary:
                trainable_parameter_summaries[full_name] = parameter_summary
                print(f"[{dataset_name} | seed {seed}] learned activation parameters for {full_name}: {parameter_summary}")

        # --------------------------------------------------------
        # Save outputs
        # --------------------------------------------------------
        print(f"\n[{dataset_name} | seed {seed}] RMSE ranking")
        rows = []
        for name, vals in sorted(metrics.items(), key=lambda kv: kv[1]["rmse"]):
            print(
                f"{name:48s}  RMSE={vals['rmse']:.6f}  "
                f"MAE={vals['mae']:.6f}  MAPE={vals['mape_percent']:.2f}%"
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "mode": "autoregressive" if AUTOREGRESSIVE_EVALUATION else "onestep",
                    "window_size": window_size,
                    "model": name,
                    "mae": vals["mae"],
                    "rmse": vals["rmse"],
                    "mape_percent": vals["mape_percent"],
                    "learned_activation_parameters": trainable_parameter_summaries.get(name, ""),
                }
            )

        metrics_df = pd.DataFrame(rows)
        metrics_path = dataset_dir / f"metrics_seed_{seed}.csv"
        metrics_df.to_csv(metrics_path, index=False)

        forecast_path = dataset_dir / f"forecasts_seed_{seed}.csv"
        save_forecast_table(y_test, predictions, forecast_path)

        top_names = [
            name for name, _ in sorted(metrics.items(), key=lambda kv: kv[1]["rmse"])[:8]
        ]
        plot_predictions = {name: predictions[name] for name in top_names}
        plot_metrics = {name: metrics[name] for name in top_names}
        plot_path = dataset_dir / f"forecast_seed_{seed}.png"

        plot_seed_results(
            dataset_name=dataset_name,
            seed=seed,
            y_true=y_test,
            predictions=plot_predictions,
            metrics=plot_metrics,
            out_path=plot_path,
        )

        print(f"[{dataset_name} | seed {seed}] saved metrics to: {metrics_path}")
        print(f"[{dataset_name} | seed {seed}] saved forecasts to: {forecast_path}")
        print(f"[{dataset_name} | seed {seed}] saved plot to: {plot_path}")

        return metrics_df

    finally:
        BATCH_SIZE = old_batch_size
        EPOCHS = old_epochs


def aggregate_results(all_results: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    aggregate = (
        all_results
        .groupby(["dataset", "mode", "model"], as_index=False)
        .agg(
            runs=("seed", "count"),
            window_size=("window_size", "first"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mape_mean=("mape_percent", "mean"),
            mape_std=("mape_percent", "std"),
        )
        .sort_values(["dataset", "rmse_mean"], ascending=[True, True])
    )

    aggregate.to_csv(out_dir / "aggregate_metrics.csv", index=False)
    return aggregate


def main() -> None:
    print("Starting standard time-series fractal activation forecasting experiment.")
    print(f"Evaluation mode: {'autoregressive' if AUTOREGRESSIVE_EVALUATION else 'onestep'}")
    print(f"Results folder: {OUTPUT_DIR.resolve()}")

    dataset_specs = [
        {
            "name": "AirPassengers",
            "loader": load_airpassengers,
            "window_size": 13,
            "batch_size": 8,
            "epochs": 120,
        },
        {
            "name": "MonthlyMeanTemperatureNottingham",
            "loader": load_monthly_mean_temperature_nottingham,
            "window_size": 13,
            "batch_size": 8,
            "epochs": 120,
        },
        {
            "name": "PerrinFreresChampagneSales",
            "loader": load_perrin_freres_champagne_sales,
            "window_size": 13,
            "batch_size": 8,
            "epochs": 120,
        },
        {
            "name": "MonthlyCarSalesQuebec",
            "loader": load_monthly_car_sales_quebec,
            "window_size": 13,
            "batch_size": 8,
            "epochs": 120,
        },
        {
            "name": "YearlySunspots",
            "loader": load_yearly_sunspots,
            "window_size": 13,
            "batch_size": 16,
            "epochs": 120,
        },
        {
            "name": "DailyMinTemperaturesMelbourne",
            "loader": load_daily_min_temperatures_melbourne,
            "window_size": 30,
            "batch_size": 32,
            "epochs": 100,
        },
        {
            "name": "MonthlyShampooSales",
            "loader": load_monthly_shampoo_sales,
            "window_size": 6,
            "batch_size": 4,
            "epochs": 160,
        },
        {
            "name": "DailyTotalFemaleBirthsCalifornia",
            "loader": load_daily_total_female_births_california,
            "window_size": 14,
            "batch_size": 16,
            "epochs": 120,
        },
        {
            "name": "MonthlyRobberiesBoston",
            "loader": load_monthly_robberies_boston,
            "window_size": 13,
            "batch_size": 8,
            "epochs": 120,
        },
        {
            "name": "LynxTrappingsCanada",
            "loader": load_lynx_trappings_canada,
            "window_size": 10,
            "batch_size": 8,
            "epochs": 140,
        },
    ]
    all_results: List[pd.DataFrame] = []

    for seed in SEEDS:
        for dataset_spec in dataset_specs:
            try:
                all_results.append(run_one_dataset_seed(dataset_spec, seed))
            except Exception as exc:
                print(
                    f"FAILED | dataset={dataset_spec['name']} | seed={seed} | "
                    f"{type(exc).__name__}: {exc}"
                )

    if not all_results:
        raise RuntimeError("No experiments completed successfully.")

    final = pd.concat(all_results, ignore_index=True)
    final_path = OUTPUT_DIR / "all_run_metrics.csv"
    final.to_csv(final_path, index=False)

    aggregate = aggregate_results(final, OUTPUT_DIR)

    print("\nAll run metrics")
    print(final.to_string(index=False))

    print("\nAggregate metrics")
    print(aggregate.to_string(index=False))

    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()