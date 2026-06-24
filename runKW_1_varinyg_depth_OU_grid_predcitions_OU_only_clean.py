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
    - mean-reverting OU-only benchmark suite over an alpha/theta grid
    - fixed-depth fractal activations with depth n_terms=30
    - random-depth fractal activations with max depth controlled by MAX_FRACTAL_DEPTH
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
import json
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers

from catboost import CatBoostRegressor

import fractal_activation_functions as fractal

try:
    from scipy.stats import levy_stable
    HAS_LEVY_STABLE = True
except ImportError:
    HAS_LEVY_STABLE = False
    levy_stable = None
    print("[WARNING] scipy.stats.levy_stable unavailable. Alpha-stable OU simulations require scipy.")


# ============================================================
# MAIN SWITCH
# ============================================================

# False = normal one-step / teacher-forced test prediction.
# True  = recursive autoregressive rollout over the whole test interval.
AUTOREGRESSIVE_EVALUATION = True


# ============================================================
# CONFIG
# ============================================================

BASE_OUTPUT_DIR = Path("mean_reverting_ou_alpha_theta_fractal_prediction_experiments_08062026")
OUTPUT_DIR = BASE_OUTPUT_DIR / (
    "autoregressive_results" if AUTOREGRESSIVE_EVALUATION else "onestep_results"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(100, 200))

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
# Large synthetic datasets (OU):
#     TRAIN_FRACTION = 0.75
#     VAL_FRACTION   = 0.10
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

# ============================================================
# OU-ONLY MEAN-REVERSION EXPERIMENT GRID
# ============================================================
#
# The synthetic data are dimensionless Ornstein--Uhlenbeck paths
# with symmetric alpha-stable innovations, matching the exact
# transition used in the uploaded mean-reversion/heavy-tail paper:
#
#   Y[n+1] = exp(-theta * dt) * Y[n]
#            + (1 - exp(-alpha * theta * dt))**(1/alpha) * S[n]
#
# where S[n] ~ S(alpha, 0, 1, 0).
#
# At alpha = 2 the Gaussian boundary is used directly:
#
#   Y[n+1] = exp(-theta * dt) * Y[n]
#            + sqrt(1 - exp(-2 * theta * dt)) * Z[n],
#
# where Z[n] ~ N(0, 1).
#
# This keeps the stationary marginal scale comparable across alpha
# values and uses alpha/theta notation throughout the experiment.

OU_SERIES_LENGTH = 2000
OU_DT = 0.01
OU_DISCRETIZATION = "exact"
OU_X0 = 0.0

OU_ALPHA_VALUES = [0.05, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
OU_THETA_VALUES = [1e-6, 0.50, 1.00, 2.00, 4.00, 8.00, 12.00, 16.00, 24.00, 32.00]

# Single global maximum depth. Changing this value propagates through
# fixed-depth, random-depth upper bounds, trainable-depth, coefficient-depth,
# and model naming conventions.
MAX_FRACTAL_DEPTH = 50

RANDOM_DEPTH_MIN = 1
RANDOM_DEPTH_MAX = MAX_FRACTAL_DEPTH
FIXED_DEPTH = MAX_FRACTAL_DEPTH

# Trainable-depth fractal activations use a differentiable straight-through
# gate over integer depths in [1, 30]. Forward evaluation uses an integer
# rounded depth, while gradients flow through a smooth sigmoid gate.
TRAINABLE_DEPTH_MIN = 1
TRAINABLE_DEPTH_MAX = 50
TRAINABLE_DEPTH_INIT = 25.0
TRAINABLE_DEPTH_GATE_SHARPNESS = 10.0
INCLUDE_TRAINABLE_DEPTH_ACTIVATIONS = True

# Trainable coefficient-depth fractal activations use all terms up to depth 30.
# Each term receives one trainable scalar coefficient initialized to 1.0.
# This lets the network learn term-wise importance while preserving the same
# activation families used in the mathematical diagnostic script.
TRAINABLE_COEFFICIENT_DEPTH_MAX = MAX_FRACTAL_DEPTH
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
# DATASET SCOPE
# ============================================================
#
# This script is intentionally OU-only. All external real-world dataset
# loaders and all Dow Jones/yfinance code have been removed. The only
# datasets used by the experiment are generated by build_ou_dataset_specs()
# and prepare_dataset_for_seed() through generate_ou_process().


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

def generate_ou_process(
    length: int,
    theta: float,
    alpha: float = 2.0,
    dt: float = OU_DT,
    seed: int = 0,
    x0: float = OU_X0,
    discretization: str = OU_DISCRETIZATION,
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Generate a dimensionless Ornstein--Uhlenbeck path using the alpha/theta
    notation and exact transition from the uploaded OU-heavy-tail framework.

    The default is the exact transition:

        alpha < 2:
            Y[n+1] = exp(-theta*dt) * Y[n]
                     + (1 - exp(-alpha*theta*dt))**(1/alpha) * S[n]
            S[n] ~ S(alpha, 0, 1, 0)

        alpha = 2:
            Y[n+1] = exp(-theta*dt) * Y[n]
                     + sqrt(1 - exp(-2*theta*dt)) * Z[n]
            Z[n] ~ N(0, 1)

    The Euler--Maruyama option is retained for diagnostic compatibility but the
    experiment grid uses OU_DISCRETIZATION="exact".
    """
    if not (0.0 < float(alpha) <= 2.0):
        raise ValueError(f"alpha must satisfy 0 < alpha <= 2. Got alpha={alpha}.")
    if float(theta) <= 0.0:
        raise ValueError(f"theta must be > 0. Got theta={theta}.")
    if float(dt) <= 0.0:
        raise ValueError(f"dt must be > 0. Got dt={dt}.")

    rng = np.random.default_rng(seed)
    x = np.zeros(int(length), dtype=np.float64)
    x[0] = float(x0)

    alpha = float(alpha)
    theta = float(theta)
    dt = float(dt)
    alpha2_tol = 1e-12
    use_exact = str(discretization).lower() == "exact"

    if alpha < 2.0 - alpha2_tol and not HAS_LEVY_STABLE:
        raise ImportError("scipy.stats.levy_stable is required for alpha-stable OU paths with alpha < 2.")

    exp_neg = np.exp(-theta * dt)

    for t in range(1, int(length)):
        if use_exact:
            if abs(alpha - 2.0) <= alpha2_tol:
                z = rng.normal(0.0, 1.0)
                var_inc = -np.expm1(-2.0 * theta * dt)
                x[t] = exp_neg * x[t - 1] + np.sqrt(var_inc) * z
            else:
                s = levy_stable.rvs(alpha, 0.0, loc=0.0, scale=1.0, random_state=rng)
                noise_scale = (-np.expm1(-alpha * theta * dt)) ** (1.0 / alpha)
                x[t] = exp_neg * x[t - 1] + noise_scale * s
        else:
            if abs(alpha - 2.0) <= alpha2_tol:
                dw = rng.normal(0.0, np.sqrt(dt))
                x[t] = x[t - 1] - theta * x[t - 1] * dt + np.sqrt(2.0 * theta) * dw
            else:
                s = levy_stable.rvs(alpha, 0.0, loc=0.0, scale=1.0, random_state=rng)
                x[t] = (
                    x[t - 1]
                    - theta * x[t - 1] * dt
                    + (alpha * theta) ** (1.0 / alpha) * (dt ** (1.0 / alpha)) * s
                )

    dates = pd.RangeIndex(start=0, stop=int(length), step=1, name="t")
    return dates, x.astype(np.float32)


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
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    residual = y_true - y_pred
    abs_residual = np.abs(residual)
    sq_residual = residual ** 2

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    max_abs_error = float(np.max(abs_residual)) if len(abs_residual) else np.nan
    bias = float(np.mean(y_pred - y_true)) if len(y_true) else np.nan
    residual_std = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0
    y_range = float(np.max(y_true) - np.min(y_true)) if len(y_true) else np.nan
    y_std = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else np.nan
    sae = float(np.sum(abs_residual))
    sse = float(np.sum(sq_residual))

    denominator = np.maximum(np.abs(y_true), 1e-8)
    mape = float(np.mean(abs_residual / denominator) * 100.0)
    smape = float(np.mean(2.0 * abs_residual / np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8)) * 100.0)
    wape = float(np.sum(abs_residual) / max(np.sum(np.abs(y_true)), 1e-8) * 100.0)
    nrmse_range = float(rmse / max(y_range, 1e-8)) if np.isfinite(y_range) else np.nan
    nrmse_std = float(rmse / max(y_std, 1e-8)) if np.isfinite(y_std) else np.nan

    if len(y_true) > 1 and np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        pearson_corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_corr = np.nan

    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = np.nan

    if len(y_true) > 1:
        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        directional_accuracy = float(np.mean(np.sign(true_diff) == np.sign(pred_diff)) * 100.0)
        naive_scale = float(np.mean(np.abs(true_diff)))
        mase = float(mae / max(naive_scale, 1e-8))
    else:
        directional_accuracy = np.nan
        mase = np.nan

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "medae": medae,
        "max_abs_error": max_abs_error,
        "bias": bias,
        "residual_mean": float(np.mean(residual)) if len(residual) else np.nan,
        "residual_std": residual_std,
        "sse": sse,
        "sae": sae,
        "mape_percent": mape,
        "smape_percent": smape,
        "wape_percent": wape,
        "nrmse_range": nrmse_range,
        "nrmse_std": nrmse_std,
        "r2": r2,
        "pearson_corr": pearson_corr,
        "directional_accuracy_percent": directional_accuracy,
        "mase": mase,
    }


def save_json(obj: dict, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


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


def save_predictions_long_table(
    dataset_name: str,
    seed: int,
    mode: str,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    metrics: Dict[str, Dict[str, float]],
    dataset_spec: Dict,
    out_path: Path,
) -> None:
    rows = []
    y_true = np.asarray(y_true).reshape(-1)
    for model_name, pred in predictions.items():
        pred = np.asarray(pred).reshape(-1)
        residual = y_true - pred
        for i in range(len(y_true)):
            rows.append({
                "dataset": dataset_name,
                "seed": seed,
                "mode": mode,
                "model": model_name,
                "test_index": i,
                "y_true": float(y_true[i]),
                "y_pred": float(pred[i]),
                "residual": float(residual[i]),
                "abs_error": float(abs(residual[i])),
                "sq_error": float(residual[i] ** 2),
                "theta": dataset_spec.get("theta", np.nan),
                "alpha": dataset_spec.get("alpha", np.nan),
                "dt": dataset_spec.get("dt", np.nan),
                "discretization": dataset_spec.get("discretization", ""),
                "window_size": dataset_spec.get("window_size", WINDOW_SIZE),
                "model_rmse": metrics[model_name].get("rmse", np.nan),
                "model_mae": metrics[model_name].get("mae", np.nan),
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def extract_trainable_parameter_details(model: tf.keras.Model) -> Dict[str, dict]:
    details = {}
    for layer in model.layers:
        if isinstance(layer, TrainableDepthFractalActivation):
            details[layer.name] = {
                "type": "trainable_depth",
                "family": layer.family,
                "min_depth": layer.min_depth,
                "max_depth": layer.max_depth,
                "continuous_depth": float(layer.continuous_depth().numpy()),
                "rounded_depth": int(layer.rounded_depth().numpy()),
            }
        elif isinstance(layer, TrainableCoefficientFractalActivation):
            coeffs = layer.coefficient_vector()
            details[layer.name] = {
                "type": "trainable_coefficients",
                "family": layer.family,
                "max_depth": layer.max_depth,
                "coefficients": [float(v) for v in coeffs],
                "coefficient_abs_sum": float(np.sum(np.abs(coeffs))),
                "coefficient_l2_norm": float(np.sqrt(np.sum(coeffs ** 2))),
            }
    return details


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
    if dataset_spec.get("kind") == "ou" or dataset_spec["name"].startswith("OU_alpha"):
        return generate_ou_process(
            length=int(dataset_spec.get("length", OU_SERIES_LENGTH)),
            theta=float(dataset_spec.get("theta")),
            alpha=float(dataset_spec.get("alpha", 2.0)),
            dt=float(dataset_spec.get("dt", OU_DT)),
            seed=seed,
            x0=float(dataset_spec.get("x0", OU_X0)),
            discretization=str(dataset_spec.get("discretization", OU_DISCRETIZATION)),
        )

    raise ValueError(
        "This script is OU-only. Dataset specifications must use kind='ou' "
        "or a name starting with 'OU_alpha'."
    )


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

        run_metadata = {
            "dataset": dataset_name,
            "seed": seed,
            "mode": "autoregressive" if AUTOREGRESSIVE_EVALUATION else "onestep",
            "series_length": int(len(series)),
            "train_fraction": TRAIN_FRACTION,
            "val_fraction": VAL_FRACTION,
            "test_fraction": TEST_FRACTION,
            "window_size": int(dataset_spec.get("window_size", WINDOW_SIZE)),
            "ou_alpha": dataset_spec.get("alpha", None),
            "ou_theta": dataset_spec.get("theta", None),
            "ou_dt": dataset_spec.get("dt", None),
            "ou_discretization": dataset_spec.get("discretization", None),
            "max_fractal_depth": MAX_FRACTAL_DEPTH,
            "fixed_depth": FIXED_DEPTH,
            "random_depth_min": RANDOM_DEPTH_MIN,
            "random_depth_max": RANDOM_DEPTH_MAX,
            "trainable_depth_min": TRAINABLE_DEPTH_MIN,
            "trainable_depth_max": TRAINABLE_DEPTH_MAX,
            "trainable_coefficient_depth_max": TRAINABLE_COEFFICIENT_DEPTH_MAX,
        }
        save_json(run_metadata, dataset_dir / f"metadata_seed_{seed}.json")

        predictions: Dict[str, np.ndarray] = {}
        metrics: Dict[str, Dict[str, float]] = {}
        trainable_parameter_summaries: Dict[str, str] = {}
        trainable_parameter_details: Dict[str, dict] = {}

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
            parameter_details = extract_trainable_parameter_details(model)
            if parameter_summary:
                trainable_parameter_summaries[full_name] = parameter_summary
                trainable_parameter_details[full_name] = parameter_details
                print(f"[{dataset_name} | seed {seed}] learned activation parameters for {full_name}: {parameter_summary}")

        # --------------------------------------------------------
        # Save outputs
        # --------------------------------------------------------
        print(f"\n[{dataset_name} | seed {seed}] RMSE ranking")
        rows = []
        for name, vals in sorted(metrics.items(), key=lambda kv: kv[1]["rmse"]):
            print(
                f"{name:48s}  RMSE={vals['rmse']:.6f}  "
                f"MAE={vals['mae']:.6f}  MSE={vals['mse']:.6f}  "
                f"MAPE={vals['mape_percent']:.2f}%  R2={vals['r2']:.4f}"
            )
            row = {
                "dataset": dataset_name,
                "seed": seed,
                "mode": "autoregressive" if AUTOREGRESSIVE_EVALUATION else "onestep",
                "window_size": window_size,
                "theta": dataset_spec.get("theta", np.nan),
                "alpha": dataset_spec.get("alpha", np.nan),
                "dt": dataset_spec.get("dt", np.nan),
                "discretization": dataset_spec.get("discretization", ""),
                "max_fractal_depth": MAX_FRACTAL_DEPTH,
                "model": name,
                "learned_activation_parameters": trainable_parameter_summaries.get(name, ""),
            }
            row.update(vals)
            rows.append(row)

        metrics_df = pd.DataFrame(rows)
        metrics_path = dataset_dir / f"metrics_seed_{seed}.csv"
        metrics_df.to_csv(metrics_path, index=False)

        forecast_path = dataset_dir / f"forecasts_seed_{seed}.csv"
        save_forecast_table(y_test, predictions, forecast_path)

        predictions_long_path = dataset_dir / f"predictions_long_seed_{seed}.csv"
        save_predictions_long_table(
            dataset_name=dataset_name,
            seed=seed,
            mode="autoregressive" if AUTOREGRESSIVE_EVALUATION else "onestep",
            y_true=y_test,
            predictions=predictions,
            metrics=metrics,
            dataset_spec=dataset_spec,
            out_path=predictions_long_path,
        )

        activation_details_path = dataset_dir / f"trainable_activation_parameters_seed_{seed}.json"
        save_json(trainable_parameter_details, activation_details_path)

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
        print(f"[{dataset_name} | seed {seed}] saved long predictions to: {predictions_long_path}")
        print(f"[{dataset_name} | seed {seed}] saved activation parameters to: {activation_details_path}")
        print(f"[{dataset_name} | seed {seed}] saved plot to: {plot_path}")

        return metrics_df

    finally:
        BATCH_SIZE = old_batch_size
        EPOCHS = old_epochs


def aggregate_results(all_results: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    metric_cols = [
        "mae", "mse", "rmse", "medae", "max_abs_error", "bias",
        "residual_mean", "residual_std", "sse", "sae", "mape_percent",
        "smape_percent", "wape_percent", "nrmse_range", "nrmse_std",
        "r2", "pearson_corr", "directional_accuracy_percent", "mase",
    ]
    agg_spec = {
        "runs": ("seed", "count"),
        "window_size": ("window_size", "first"),
        "theta": ("theta", "first"),
        "alpha": ("alpha", "first"),
        "dt": ("dt", "first"),
        "max_fractal_depth": ("max_fractal_depth", "first"),
    }
    for col in metric_cols:
        if col in all_results.columns:
            agg_spec[f"{col}_mean"] = (col, "mean")
            agg_spec[f"{col}_std"] = (col, "std")

    aggregate = (
        all_results
        .groupby(["dataset", "mode", "model"], as_index=False)
        .agg(**agg_spec)
        .sort_values(["dataset", "rmse_mean"], ascending=[True, True])
    )

    aggregate.to_csv(out_dir / "aggregate_metrics.csv", index=False)

    leaderboard = (
        aggregate
        .sort_values(["dataset", "rmse_mean"], ascending=[True, True])
        .copy()
    )
    leaderboard["rank_within_dataset"] = leaderboard.groupby("dataset")["rmse_mean"].rank(method="dense", ascending=True)
    leaderboard.to_csv(out_dir / "leaderboard_by_ou_parameter_setting.csv", index=False)

    cross_model = (
        leaderboard
        .groupby(["mode", "model"], as_index=False)
        .agg(
            mean_rank=("rank_within_dataset", "mean"),
            median_rank=("rank_within_dataset", "median"),
            mean_rmse=("rmse_mean", "mean"),
            std_rmse=("rmse_mean", "std"),
            datasets=("dataset", "count"),
        )
        .sort_values(["mean_rank", "mean_rmse"], ascending=[True, True])
    )
    cross_model.to_csv(out_dir / "cross_ou_grid_model_leaderboard.csv", index=False)

    return aggregate


def build_ou_dataset_specs() -> List[Dict]:
    specs: List[Dict] = []
    for alpha in OU_ALPHA_VALUES:
        for theta in OU_THETA_VALUES:
            alpha_tag = str(alpha).replace(".", "p").replace("-", "m")
            theta_tag = ("1em6" if abs(theta - 1e-6) < 1e-12 else str(theta).replace(".", "p"))
            specs.append({
                "name": f"OU_alpha{alpha_tag}_theta{theta_tag}_exact_dt{str(OU_DT).replace('.', 'p')}",
                "kind": "ou",
                "alpha": float(alpha),
                "theta": float(theta),
                "dt": float(OU_DT),
                "discretization": OU_DISCRETIZATION,
                "length": OU_SERIES_LENGTH,
                "x0": OU_X0,
                "window_size": 13,
                "batch_size": 64,
                "epochs": 80,
            })
    return specs


def write_experiment_config(dataset_specs: List[Dict]) -> None:
    config = {
        "experiment_scope": "OU-only mean-reverting alpha/theta parameter-grid forecasting",
        "evaluation_mode": "autoregressive" if AUTOREGRESSIVE_EVALUATION else "onestep",
        "output_dir": str(OUTPUT_DIR),
        "seeds": SEEDS,
        "train_fraction": TRAIN_FRACTION,
        "val_fraction": VAL_FRACTION,
        "test_fraction": TEST_FRACTION,
        "ou_series_length": OU_SERIES_LENGTH,
        "ou_dt": OU_DT,
        "ou_discretization": OU_DISCRETIZATION,
        "ou_alpha_values": OU_ALPHA_VALUES,
        "ou_theta_values": OU_THETA_VALUES,
        "max_fractal_depth": MAX_FRACTAL_DEPTH,
        "random_depth_min": RANDOM_DEPTH_MIN,
        "random_depth_max": RANDOM_DEPTH_MAX,
        "fixed_depth": FIXED_DEPTH,
        "trainable_depth_min": TRAINABLE_DEPTH_MIN,
        "trainable_depth_max": TRAINABLE_DEPTH_MAX,
        "trainable_coefficient_depth_max": TRAINABLE_COEFFICIENT_DEPTH_MAX,
        "num_dataset_specs": len(dataset_specs),
        "dataset_specs": dataset_specs,
    }
    save_json(config, OUTPUT_DIR / "experiment_config.json")
    pd.DataFrame(dataset_specs).to_csv(OUTPUT_DIR / "ou_parameter_grid.csv", index=False)


def main() -> None:
    print("Starting OU-only mean-reverting alpha/theta fractal forecasting experiment.")
    print(f"Evaluation mode: {'autoregressive' if AUTOREGRESSIVE_EVALUATION else 'onestep'}")
    print(f"Results folder: {OUTPUT_DIR.resolve()}")

    dataset_specs = build_ou_dataset_specs()
    write_experiment_config(dataset_specs)

    print(f"OU-only experiment with {len(dataset_specs)} alpha/theta parameter settings.")
    print(f"Alpha grid: {OU_ALPHA_VALUES}")
    print(f"Theta grid: {OU_THETA_VALUES}")
    print(f"OU exact-transition dt: {OU_DT}")
    print(f"Maximum fractal depth used in naming/config: {MAX_FRACTAL_DEPTH}")

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
