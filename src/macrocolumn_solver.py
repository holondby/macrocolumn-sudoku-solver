#!/usr/bin/env python
# coding: utf-8

# ──────────────────────────────────────────────────────────────────────────────
# Macrocolumn RL Sudoku Solver
# ──────────────────────────────────────────────────────────────────────────────
#
# Learns while solving: a cortical-inspired macrocolumn predicts Q(s,·) and
# guides depth-first search (DFS) in Sudoku.
#
# Architecture:
#   CNN encoder → competing LSTM/Dense minicolumns → divisive normalization
#   → soft winner-take-all → Q(s,·)
#
# Search and learning:
#   - propagate forced moves first
#   - select non-forced cells by divisive soft-WTA competition
#   - order digits greedily by Q, sampling only the first tried digit from
#     a fixed-ε behavior policy
#   - update Q-values with potential-shaped TD(λ) minibatches over
#     decision-to-decision macro-transitions
#   - learn from terminal forced outcomes, but not from nonterminal forced moves

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
# Model geometry (network capacity / representation)
CONV_FILTERS    = 36      # Number of filters in each CNN encoder layer.
KERNEL_SIZE     = 3       # Spatial size of convolution kernels.
LSTM_UNITS      = 10      # Units in each minicolumn LSTM layer.
DENSE_UNITS     = 30      # Units in each dense layer of a minicolumn MLP stack.
N_LAYERS        = 3       # Number of dense layers inside each minicolumn.
N_COLUMNS       = 7       # Number of competing minicolumns in the macrocolumn.

# Divisive normalization across minicolumns
GAIN            = 1.15    # Strength of inhibitory divisive normalization.
INH_SIGMA       = 1.5     # Spatial spread of inhibition between columns.

# Soft winner-take-all across minicolumn Q heads, reused for cell scoring
WTA_BETA        = 2.0     # Inverse temperature for Soft-WTA weighting.

# Reinforcement-learning dynamics
GAMMA           = 0.99    # Discount factor for future rewards.
PHI_GAIN        = 0.10    # Weight of the potential-based shaping signal Φ(s).
TD_LAMBDA       = 0.95    # TD(λ) mixing factor for decision-node backups.

# Sparse terminal rewards
R_SOLVED        = 1.0     # Reward when the puzzle is solved.
R_CONTRA        = -1.0    # Reward when a contradiction or dead-end occurs.

# Fixed ε-greedy behavior, used only at non-forced choice points
EPSILON0        = 0.10    # Exploration probability.

# Queued TD target optimization
TD_FLUSH_EVERY  = 32      # Flush queued samples once this many are accumulated.
TD_BATCH_SIZE   = 32      # Minibatch size used during gradient updates.
TD_BUFFER_MAX   = 256     # Maximum number of queued training samples.

# DFS inference cache
CACHE_MAX       = 20000   # Maximum number of cached states per solve.

# Reproducibility
SEED            = 121252  # Random seed for NumPy, Python, and TensorFlow.

# ──────────────────────────────────────────────────────────────────────────────
# Runtime display and checkpoint settings
# ──────────────────────────────────────────────────────────────────────────────
SHOW_MOVES      = False   # Print the full move sequence when a puzzle is solved.
SHOW_PLOT       = False   # Display graphical Sudoku boards before/after solving.
CKPT_PREFIX     = "macrocolumn_model"  # Prefix used for saved model checkpoints.

# ──────────────────────────────────────────────────────────────────────────────
# Internal constants (rarely tuned)
# ──────────────────────────────────────────────────────────────────────────────
EPS              = 1e-12  # Small epsilon.
OFFSET_DENOM     = 0.01   # Denominator offset.
RING_TOPOLOGY    = True   # Use circular distance between columns.
NEG_Q_CLAMP      = -1e9   # Replacement value for bad/non-finite Q values.
L1_REG           = 1e-6   # L1 penalty.

# ──────────────────────────────────────────────────────────────────────────────
# Imports + validation + seeding
# ──────────────────────────────────────────────────────────────────────────────
import os
import re
import sys
import random
import numbers
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.setrecursionlimit(10000)

import numpy as np


def validate_hyperparameters() -> None:
    """Validate hyperparameter ranges and types."""
    positive_ints = {
        "CONV_FILTERS": CONV_FILTERS,
        "KERNEL_SIZE": KERNEL_SIZE,
        "LSTM_UNITS": LSTM_UNITS,
        "DENSE_UNITS": DENSE_UNITS,
        "N_LAYERS": N_LAYERS,
        "N_COLUMNS": N_COLUMNS,
        "TD_FLUSH_EVERY": TD_FLUSH_EVERY,
        "TD_BATCH_SIZE": TD_BATCH_SIZE,
        "TD_BUFFER_MAX": TD_BUFFER_MAX,
        "CACHE_MAX": CACHE_MAX,
    }
    for name, value in positive_ints.items():
        if not isinstance(value, numbers.Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(SEED, numbers.Integral) or not (0 <= int(SEED) < 2**32):
        raise ValueError("SEED must be an integer in [0, 2**32).")
    bounded = {
        "EPSILON0": (EPSILON0, 0.0, 1.0),
        "GAMMA": (GAMMA, 0.0, 1.0),
        "TD_LAMBDA": (TD_LAMBDA, 0.0, 1.0),
    }
    for name, (value, lo, hi) in bounded.items():
        if not (lo <= float(value) <= hi):
            raise ValueError(f"{name} must be in [{lo}, {hi}].")
    positive = {
        "EPS": EPS,
        "INH_SIGMA": INH_SIGMA,
        "OFFSET_DENOM": OFFSET_DENOM,
    }
    for name, value in positive.items():
        if float(value) <= 0.0:
            raise ValueError(f"{name} must be > 0.")
    finite = {
        "PHI_GAIN": PHI_GAIN,
        "NEG_Q_CLAMP": NEG_Q_CLAMP,
        "R_SOLVED": R_SOLVED,
        "R_CONTRA": R_CONTRA,
    }
    for name, value in finite.items():
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")
    if not np.isfinite(float(L1_REG)) or float(L1_REG) < 0.0:
        raise ValueError("L1_REG must be finite and >= 0.")
    if not np.isfinite(float(WTA_BETA)) or float(WTA_BETA) < 0.0:
        raise ValueError("WTA_BETA must be finite and >= 0.")
    if not np.isfinite(float(GAIN)) or float(GAIN) < 0.0:
        raise ValueError("GAIN must be finite and >= 0.")
    if float(R_SOLVED) <= float(R_CONTRA):
        raise ValueError("Require R_SOLVED > R_CONTRA.")
    if not isinstance(CKPT_PREFIX, str) or not CKPT_PREFIX.strip():
        raise ValueError("CKPT_PREFIX must be a non-empty string.")


np.random.seed(SEED)
random.seed(SEED)

validate_hyperparameters()

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    try:
        tf.random.set_seed(SEED)
    except Exception:
        pass

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Dense,
    LSTM,
    Layer,
    LayerNormalization,
    Reshape,
    Softmax,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1

# ──────────────────────────────────────────────────────────────────────────────
# Aliases
# ──────────────────────────────────────────────────────────────────────────────
Decision = Any
Action = Any
MoveRecord = Tuple[int, int, int]
OptionMap = Dict[Decision, np.ndarray]

# ──────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ──────────────────────────────────────────────────────────────────────────────
def ensure_f32_contig(x: np.ndarray) -> np.ndarray:
    """Return a contiguous float32 array."""
    if isinstance(x, np.ndarray) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]:
        return x
    return np.ascontiguousarray(x, dtype=np.float32)


def sanitize_q_values(q: np.ndarray) -> np.ndarray:
    """Replace non-finite Q values."""
    q = np.asarray(q, dtype=np.float64)
    bad = ~np.isfinite(q)
    if np.any(bad):
        q = q.copy()
        q[bad] = float(NEG_Q_CLAMP)
    return q


def stable_softmax(values: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Return a stable softmax(beta * values)."""
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return np.empty((0,), dtype=np.float64)
    logits = float(beta) * vals
    logits = logits - np.max(logits)
    np.clip(logits, -60.0, 0.0, out=logits)
    exps = np.exp(logits)
    denom = float(np.sum(exps)) + float(EPS)
    return exps / denom


def soft_wta_reduce(values: np.ndarray, beta: float = WTA_BETA) -> float:
    """Reduce competing finite values to one soft-WTA score."""
    vals = np.asarray(values, dtype=np.float64)
    n = int(vals.size)
    if n <= 0:
        return float(NEG_Q_CLAMP)
    if n == 1:
        return float(vals[0])
    probs = stable_softmax(vals, beta=beta)
    score = float(np.dot(probs, vals))
    return score if np.isfinite(score) else float(NEG_Q_CLAMP)


def divisively_normalize_scores(
    activity: np.ndarray,
    weights: np.ndarray,
    gain: float = GAIN,
    offset: float = OFFSET_DENOM,
) -> np.ndarray:
    """Apply divisive normalization to activities."""
    a = np.asarray(activity, dtype=np.float64)
    n = int(a.size)
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    if not (isinstance(weights, np.ndarray) and weights.shape == (n, n)):
        return a.copy()
    inh = weights @ a
    denom = np.maximum(float(offset) + float(gain) * inh, float(EPS))
    out = a / denom
    out[~np.isfinite(out)] = 0.0
    return out


def mu_sample_first_action(greedy_actions: np.ndarray, eps: float) -> int:
    """Sample the first action from μ."""
    greedy = np.asarray(greedy_actions, dtype=np.int32)
    k = int(greedy.size)
    if k <= 0:
        return 0
    if k == 1:
        return int(greedy[0])
    eps = float(np.clip(float(eps), 0.0, 1.0))
    if eps <= 0.0:
        return int(greedy[0])
    if float(np.random.random()) < eps:
        return int(greedy[int(np.random.randint(k))])
    return int(greedy[0])


def phi_potential(progress: float, terminal: bool) -> float:
    """Return the shaping potential Φ(s)."""
    return 0.0 if terminal else float(PHI_GAIN) * float(np.clip(progress, 0.0, 1.0))


def convolution_dim(size: int, kernel: int, stride: int = 1) -> int:
    """Return the valid-convolution output size."""
    return ((int(size) - int(kernel)) // int(stride)) + 1

# ──────────────────────────────────────────────────────────────────────────────
# Generic task types
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TaskSpec:
    """State/action dimensions for the Q-network."""
    state_shape: Tuple[int, ...]
    action_dim: int


@dataclass
class ScanResult:
    """Current legal actions and progress."""
    ok: bool
    option_map: OptionMap
    legal_mask: np.ndarray
    progress: float
    forced_decisions: List[Decision]


@dataclass(frozen=True)
class Step:
    """One decision/action pair."""
    decision: Decision
    action: Action


@dataclass
class DecisionCtx:
    """Decision-node context."""
    scan: ScanResult
    decision: Decision
    actions: np.ndarray
    q: np.ndarray
    phi_val: float


class BranchingTaskState(ABC):
    """Abstract branchable task state."""
    spec: TaskSpec
    state: np.ndarray

    @abstractmethod
    def reset(self, task_input: Any) -> None:
        """Reset the task state."""

    @abstractmethod
    def is_solved(self) -> bool:
        """Return whether the task is solved."""

    @abstractmethod
    def cache_key(self) -> bytes:
        """Return a cache key for the state."""

    @abstractmethod
    def scan(self) -> ScanResult:
        """Return legal actions and progress."""

    @abstractmethod
    def apply(self, decision: Decision, action: Action) -> None:
        """Apply an action."""

    @abstractmethod
    def undo(self, decision: Decision, action: Action) -> None:
        """Undo an action."""

    @abstractmethod
    def action_index(self, decision: Decision, action: Action) -> int:
        """Return the flattened action index."""

    @abstractmethod
    def action_values(self, decision: Decision, actions: np.ndarray, q_vals: np.ndarray) -> np.ndarray:
        """Return Q values for the legal actions."""

    @abstractmethod
    def move_record(self, decision: Decision, action: Action) -> MoveRecord:
        """Return a user-visible move record."""


class BranchPolicy(ABC):
    """Decision-branching policy."""

    @abstractmethod
    def choose_decision(self, scan: ScanResult, q_vals: np.ndarray, task: BranchingTaskState) -> Optional[Decision]:
        """Pick a non-forced decision."""

    @abstractmethod
    def choose_forced_decision(self, forced_decisions: Sequence[Decision], task: BranchingTaskState) -> Optional[Decision]:
        """Pick a forced decision."""

    @abstractmethod
    def greedy_action_order(
        self,
        decision: Decision,
        actions: np.ndarray,
        q_vals: np.ndarray,
        task: BranchingTaskState,
    ) -> np.ndarray:
        """Return greedy-sorted actions."""


class BehaviorPolicy(ABC):
    """Behavior policy over greedy actions."""

    @abstractmethod
    def epsilon(self, learn: bool, explore: bool, is_choice: bool) -> float:
        """Return ε for the current node."""

    @abstractmethod
    def policy_value(self, q_greedy_ordered: np.ndarray, eps_node: float) -> float:
        """Return Vμ(s) from greedy-ranked Q values."""

    @abstractmethod
    def action_order(self, greedy_actions: np.ndarray, eps_node: float) -> np.ndarray:
        """Return the DFS action order."""


class BackupPolicy(ABC):
    """Backup and shaping policy for DFS."""

    @abstractmethod
    def phi(self, progress: float, terminal: bool) -> float:
        """Return the shaping potential Φ."""

    @abstractmethod
    def solved_reward(self) -> float:
        """Return the solved reward."""

    @abstractmethod
    def contradiction_reward(self) -> float:
        """Return the contradiction reward."""

    @abstractmethod
    def forced_terminal_target(self, phi_s: float, reward: float) -> float:
        """Return the target for terminal forced moves."""

    @abstractmethod
    def branch_target(
        self,
        phi_s: float,
        child_terminal: bool,
        child_term_reward: float,
        phi_child: float,
        g_child: float,
        v_mu_child: float,
    ) -> float:
        """Return the TD(λ) target for a branching move."""

# ──────────────────────────────────────────────────────────────────────────────
# Custom layers
# ──────────────────────────────────────────────────────────────────────────────
class SliceColumn(Layer):
    """Extract x[:, k, :]."""

    def __init__(self, index: int, **kwargs):
        super().__init__(**kwargs)
        self.index = int(index)

    def call(self, x):
        return x[:, self.index, :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"index": self.index})
        return cfg


class QScoreLayer(Layer):
    """Compute each column's best legal Q."""

    def call(self, inputs):
        q_per_col, legal_mask = inputs
        neg_inf = tf.cast(NEG_Q_CLAMP, q_per_col.dtype)
        q_safe = tf.where(tf.math.is_finite(q_per_col), q_per_col, neg_inf)
        mask = tf.cast(legal_mask, q_per_col.dtype)
        mask = tf.expand_dims(mask, axis=1)
        q_masked = tf.where(mask > 0.0, q_safe, neg_inf)
        scores = tf.reduce_max(q_masked, axis=2)
        return tf.where(tf.math.is_finite(scores), scores, tf.zeros_like(scores))


class SoftWTALayer(Layer):
    """Return Σ softmax(beta * scores) * q_per_col."""

    def __init__(self, beta: float = WTA_BETA, **kwargs):
        super().__init__(**kwargs)
        self.beta = float(beta)
        self.softmax = Softmax(axis=1)

    def call(self, inputs):
        scores, q_per_col = inputs
        q_neg = tf.cast(NEG_Q_CLAMP, q_per_col.dtype)
        score_safe = tf.where(tf.math.is_finite(scores), scores, tf.zeros_like(scores))
        q_safe = tf.where(tf.math.is_finite(q_per_col), q_per_col, q_neg)
        logits = tf.cast(self.beta, tf.float32) * tf.cast(score_safe, tf.float32)
        logits = logits - tf.reduce_max(logits, axis=1, keepdims=True)
        w = tf.expand_dims(self.softmax(logits), axis=-1)
        return tf.reduce_sum(w * q_safe, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"beta": self.beta})
        return cfg


class NormalizationPoolLayer(Layer):
    """Neighbor-pooled divisive inhibition."""

    def __init__(
        self,
        n_columns: int = N_COLUMNS,
        gain: float = GAIN,
        sigma: float = INH_SIGMA,
        threshold: Optional[float] = None,  # Retained only for older checkpoint configs.
        offset_denom: float = OFFSET_DENOM,
        ring_topology: bool = RING_TOPOLOGY,
        eps: float = EPS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_columns = int(n_columns)
        self.gain = float(gain)
        self.sigma = float(sigma)
        self.offset_denom = float(offset_denom)
        self.ring_topology = bool(ring_topology)
        self.eps = float(eps)
        self.W = None

    def build(self, input_shape):
        c = int(self.n_columns)
        idx = np.arange(c, dtype=np.float32)
        dist = np.abs(idx[:, None] - idx[None, :])
        if self.ring_topology and c > 1:
            dist = np.minimum(dist, c - dist)
        sigma = float(max(self.sigma, 1e-6))
        w = np.exp(-(dist * dist) / (2.0 * sigma * sigma)).astype(np.float32)
        np.fill_diagonal(w, 0.0)
        row_sums = w.sum(axis=1, keepdims=True).astype(np.float32)
        w_norm = np.zeros_like(w, dtype=np.float32)
        np.divide(w, row_sums + self.eps, out=w_norm, where=(row_sums > 0))
        self.W = tf.constant(w_norm, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs, training=None):
        feats = tf.stack(inputs, axis=1)
        activity = tf.reduce_mean(tf.nn.relu(feats), axis=2)
        inh_drive = tf.matmul(activity, self.W, transpose_b=True)
        denom = tf.maximum(
            tf.cast(self.offset_denom, tf.float32) + tf.cast(self.gain, tf.float32) * tf.cast(inh_drive, tf.float32),
            tf.cast(self.eps, tf.float32),
        )
        return feats / tf.expand_dims(denom, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "n_columns": self.n_columns,
                "gain": self.gain,
                "sigma": self.sigma,
                "offset_denom": self.offset_denom,
                "ring_topology": self.ring_topology,
                "eps": self.eps,
            }
        )
        return cfg

# ──────────────────────────────────────────────────────────────────────────────
# Network
# ──────────────────────────────────────────────────────────────────────────────
class MacrocolumnFactory:
    """Build macrocolumn Q-networks."""

    @staticmethod
    def sequence_shape(state_shape: Tuple[int, ...]) -> Tuple[int, int]:
        """Return the encoder sequence shape."""
        if len(state_shape) != 3:
            raise ValueError("Macrocolumn expects a 3D image-like state shape (H,W,C).")
        h, w, _ = [int(v) for v in state_shape]
        h2 = convolution_dim(h, kernel=KERNEL_SIZE, stride=2)
        w2 = convolution_dim(w, kernel=KERNEL_SIZE, stride=2)
        if h2 <= 0 or w2 <= 0:
            raise ValueError("State shape is too small for the stride-2 encoder.")
        return h2 * w2, CONV_FILTERS

    @staticmethod
    def make_minicolumn(seq_len: int, feat_dim: int) -> Sequential:
        """Build one minicolumn."""
        col = Sequential([Input((int(seq_len), int(feat_dim))), LSTM(LSTM_UNITS)])
        for _ in range(int(N_LAYERS)):
            col.add(Dense(DENSE_UNITS, activation="relu", kernel_regularizer=l1(L1_REG)))
        return col

    @staticmethod
    def make_macrocolumn(task_spec: TaskSpec) -> tf.keras.Model:
        """Build the full macrocolumn Q-network."""
        state_shape = tuple(int(v) for v in task_spec.state_shape)
        action_dim = int(task_spec.action_dim)
        seq_len, feat_dim = MacrocolumnFactory.sequence_shape(state_shape)
        state_in = Input(state_shape, name="state")
        mask_in = Input((action_dim,), name="legal_mask")
        x = Conv2D(CONV_FILTERS, KERNEL_SIZE, activation="relu", padding="same")(state_in)
        x = Conv2D(CONV_FILTERS, KERNEL_SIZE, strides=2, activation="relu")(x)
        x = LayerNormalization()(Reshape((seq_len, feat_dim))(x))
        cols = [MacrocolumnFactory.make_minicolumn(seq_len, feat_dim)(x) for _ in range(int(N_COLUMNS))]
        inhibited = NormalizationPoolLayer(
            n_columns=N_COLUMNS,
            gain=GAIN,
            sigma=INH_SIGMA,
            offset_denom=OFFSET_DENOM,
            ring_topology=RING_TOPOLOGY,
            eps=EPS,
            name="norm_pool",
        )(cols)
        q_blocks = []
        for k in range(int(N_COLUMNS)):
            hk = SliceColumn(index=k, name=f"col_feat_{k}")(inhibited)
            qk = Dense(action_dim, name=f"q_head_c{k}")(hk)
            q_blocks.append(Reshape((1, action_dim), name=f"q_expand_c{k}")(qk))
        q_per_col = Concatenate(axis=1, name="q_per_col")(q_blocks)
        scores = QScoreLayer(name="q_scores")([q_per_col, mask_in])
        out = SoftWTALayer(beta=WTA_BETA, name="q_soft_wta")([scores, q_per_col])
        return Model([state_in, mask_in], out, name="macrocolumn")

    @staticmethod
    def custom_objects() -> Dict[str, Any]:
        """Return custom layer mappings."""
        return {
            "SliceColumn": SliceColumn,
            "NormalizationPoolLayer": NormalizationPoolLayer,
            "QScoreLayer": QScoreLayer,
            "SoftWTALayer": SoftWTALayer,
        }


CKPT_PATTERN = re.compile(rf"^{re.escape(CKPT_PREFIX)} ?\((\d+)\)\.keras$")


def latest_checkpoint_index() -> Optional[int]:
    """Return the latest checkpoint index."""
    best = None
    try:
        for entry in os.scandir("."):
            if not entry.is_file():
                continue
            m = CKPT_PATTERN.match(entry.name)
            if m:
                idx = int(m.group(1))
                best = idx if best is None else max(best, idx)
    except Exception:
        pass
    return best


def compile_if_needed(model: tf.keras.Model, lr: float, for_training: bool) -> None:
    """Compile the model or update its learning rate."""
    if not for_training:
        return
    if getattr(model, "optimizer", None) is None:
        model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0), loss=tf.keras.losses.Huber())
        return
    try:
        opt = model.optimizer
        lr_attr = getattr(opt, "learning_rate", getattr(opt, "lr", None))
        if lr_attr is not None and hasattr(lr_attr, "assign"):
            lr_attr.assign(float(lr))
        else:
            opt.learning_rate = float(lr)
    except Exception:
        pass
    try:
        opt_vars = getattr(model.optimizer, "variables", None)
        if callable(opt_vars):
            opt_vars = opt_vars()
        if (opt_vars is None) or (len(opt_vars) == 0):
            if hasattr(model.optimizer, "build"):
                model.optimizer.build(model.trainable_variables)
    except Exception:
        pass


class QNetwork:
    """Thin wrapper around the Q-network."""

    def __init__(self, model: tf.keras.Model, task_spec: TaskSpec):
        self.model = model
        self.task_spec = task_spec
        self.predict_fn = self.build_predict_fn()

    def build_predict_fn(self):
        """Build the cached prediction function."""
        state_sig = (None,) + tuple(int(v) for v in self.task_spec.state_shape)
        action_dim = int(self.task_spec.action_dim)

        @tf.function(
            input_signature=[
                tf.TensorSpec(state_sig, tf.float32),
                tf.TensorSpec((None, action_dim), tf.float32),
            ],
            reduce_retracing=True,
        )
        def predict_fn(state, legal_mask):
            return self.model([state, legal_mask], training=False)

        return predict_fn

    def predict(self, state: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        """Predict Q(s,·)."""
        out = self.predict_fn(ensure_f32_contig(state), ensure_f32_contig(legal_mask))[0]
        return np.asarray(out, dtype=np.float32)

    def save(self, idx: int) -> str:
        """Save a checkpoint."""
        idx = int(idx)
        tmp = f"{CKPT_PREFIX} ({idx}).tmp.keras"
        fn = f"{CKPT_PREFIX} ({idx}).keras"
        try:
            try:
                self.model.save(tmp, include_optimizer=True)
            except Exception:
                self.model.save(tmp)
            os.replace(tmp, fn)
        finally:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
        return fn

    @classmethod
    def load_or_create(cls, task_spec: TaskSpec, lr: float = 3e-4, for_training: bool = False):
        """Load the latest checkpoint or build a new model."""
        latest_n = latest_checkpoint_index()
        if latest_n is not None:
            fn = f"{CKPT_PREFIX} ({latest_n}).keras"
            print(f"\nLoading '{fn}'")
            custom = MacrocolumnFactory.custom_objects()
            model = None
            if for_training:
                try:
                    model = load_model(fn, custom_objects=custom, compile=True, safe_mode=True)
                except TypeError:
                    try:
                        model = load_model(fn, custom_objects=custom, compile=True)
                    except Exception:
                        model = None
                except Exception:
                    model = None
            if model is None:
                try:
                    model = load_model(fn, custom_objects=custom, compile=False, safe_mode=True)
                except TypeError:
                    model = load_model(fn, custom_objects=custom, compile=False)
        else:
            print("\nBuilding a new macrocolumn.")
            model = MacrocolumnFactory.make_macrocolumn(task_spec)
            latest_n = 0
        compile_if_needed(model, lr, for_training)
        return cls(model, task_spec), int(latest_n)

# ──────────────────────────────────────────────────────────────────────────────
# Generic RL components
# ──────────────────────────────────────────────────────────────────────────────
class EpsilonGreedyFirstActionPolicy(BehaviorPolicy):
    """Fixed-ε policy for the first tried action."""

    def epsilon(self, learn: bool, explore: bool, is_choice: bool) -> float:
        """Return ε for the current node."""
        if (not is_choice) or (not learn) or (not explore):
            return 0.0
        return float(np.clip(float(EPSILON0), 0.0, 1.0))

    def policy_value(self, q_greedy_ordered: np.ndarray, eps_node: float) -> float:
        """Return Vμ for greedy-ranked Q values."""
        qd = np.asarray(q_greedy_ordered, dtype=np.float64)
        k = int(qd.size)
        if k <= 0:
            return 0.0
        if k == 1:
            return float(qd[0]) if np.isfinite(qd[0]) else float(NEG_Q_CLAMP)
        eps = float(np.clip(float(eps_node), 0.0, 1.0))
        probs = np.full((k,), eps / float(k), dtype=np.float64)
        probs[0] += 1.0 - eps
        v = float(np.dot(probs, qd))
        return float(v) if np.isfinite(v) else float(NEG_Q_CLAMP)

    def action_order(self, greedy_actions: np.ndarray, eps_node: float) -> np.ndarray:
        """Return the DFS action order."""
        greedy = np.asarray(greedy_actions, dtype=np.int32)
        if greedy.size <= 1:
            return greedy.astype(np.int32, copy=False)
        eps = float(np.clip(float(eps_node), 0.0, 1.0))
        if eps <= 0.0:
            return greedy
        first = int(mu_sample_first_action(greedy, eps))
        return np.concatenate(([first], greedy[greedy != first])).astype(np.int32, copy=False)


class TDLambdaExpectedBackup(BackupPolicy):
    """Potential-shaped TD(λ) backup over decision-node macro-transitions."""

    def __init__(
        self,
        gamma: float = GAMMA,
        td_lambda: float = TD_LAMBDA,
        r_solved: float = R_SOLVED,
        r_contra: float = R_CONTRA,
    ):
        self._gamma = float(np.clip(float(gamma), 0.0, 1.0))
        self._lambda = float(np.clip(float(td_lambda), 0.0, 1.0))
        self._r_solved = float(r_solved)
        self._r_contra = float(r_contra)

    def phi(self, progress: float, terminal: bool) -> float:
        """Return the shaping potential Φ."""
        return float(phi_potential(progress, terminal))

    def solved_reward(self) -> float:
        """Return the solved reward."""
        return self._r_solved

    def contradiction_reward(self) -> float:
        """Return the contradiction reward."""
        return self._r_contra

    def forced_terminal_target(self, phi_s: float, reward: float) -> float:
        """Return the terminal forced-move target."""
        return float(reward) - float(phi_s)

    def branch_target(
        self,
        phi_s: float,
        child_terminal: bool,
        child_term_reward: float,
        phi_child: float,
        g_child: float,
        v_mu_child: float,
    ) -> float:
        """Return the TD(λ) target for a branching move."""
        r_term = float(child_term_reward) if bool(child_terminal) else 0.0
        r_shape = self._gamma * float(phi_child) - float(phi_s)
        if bool(child_terminal):
            mix = 0.0
        else:
            mix = (1.0 - self._lambda) * float(v_mu_child) + self._lambda * float(g_child)
        return r_term + r_shape + self._gamma * mix


HUBER = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)


class Learner:
    """Queued minibatch learner."""

    def __init__(self, model: tf.keras.Model, task_spec: TaskSpec, enabled: bool, on_update=None):
        self.model = model
        self.task_spec = task_spec
        self.enabled = bool(enabled)
        self.on_update = on_update
        self.state_shape = tuple(int(v) for v in self.task_spec.state_shape)
        self.action_dim = int(self.task_spec.action_dim)
        self._buf_states: List[np.ndarray] = []
        self._buf_masks: List[np.ndarray] = []
        self._buf_actions: List[int] = []
        self._buf_targets: List[float] = []

        @tf.function(
            input_signature=[
                tf.TensorSpec((None,) + self.state_shape, tf.float32),
                tf.TensorSpec((None, self.action_dim), tf.float32),
                tf.TensorSpec((None,), tf.int32),
                tf.TensorSpec((None,), tf.float32),
            ],
            reduce_retracing=True,
        )
        def train_step(states, masks, action_idxs, targets):
            with tf.GradientTape() as tape:
                q_all = self.model([states, masks], training=True)
                qa = tf.gather(q_all, action_idxs, batch_dims=1)
                per = HUBER(tf.stop_gradient(targets), qa)
                data_loss = tf.reduce_mean(per)
                reg_loss = tf.add_n(self.model.losses) if self.model.losses else tf.constant(0.0, dtype=data_loss.dtype)
                loss = data_loss + reg_loss
            grads = tape.gradient(loss, self.model.trainable_variables)
            grads_and_vars = [(g, v) for (g, v) in zip(grads, self.model.trainable_variables) if g is not None]
            if grads_and_vars:
                self.model.optimizer.apply_gradients(grads_and_vars)
            return loss

        self.train_step = train_step

    def clear(self) -> None:
        """Discard any queued samples."""
        self._buf_states.clear()
        self._buf_masks.clear()
        self._buf_actions.clear()
        self._buf_targets.clear()

    def queue(self, state1: np.ndarray, mask1: np.ndarray, action_idx: int, target: float) -> None:
        """Queue one TD training sample."""
        if (not self.enabled) or (not np.isfinite(target)):
            return
        if not (
            isinstance(state1, np.ndarray)
            and state1.shape == (1,) + self.state_shape
            and isinstance(mask1, np.ndarray)
            and mask1.shape == (1, self.action_dim)
        ):
            return
        self._buf_states.append(ensure_f32_contig(state1[0]))
        self._buf_masks.append(ensure_f32_contig(mask1[0]))
        self._buf_actions.append(int(action_idx))
        self._buf_targets.append(float(target))
        n = len(self._buf_targets)
        if n > int(TD_BUFFER_MAX):
            drop = n - int(TD_BUFFER_MAX)
            del self._buf_states[:drop]
            del self._buf_masks[:drop]
            del self._buf_actions[:drop]
            del self._buf_targets[:drop]
        if len(self._buf_targets) >= int(TD_FLUSH_EVERY):
            self.flush()

    def flush(self) -> None:
        """Train on all queued samples."""
        if not self.enabled:
            self.clear()
            return
        n = len(self._buf_targets)
        if n <= 0:
            return
        idx = np.random.permutation(n)
        states_all = np.stack(self._buf_states, axis=0).astype(np.float32, copy=False)
        masks_all = np.stack(self._buf_masks, axis=0).astype(np.float32, copy=False)
        acts_all = np.asarray(self._buf_actions, dtype=np.int32)
        targs_all = np.asarray(self._buf_targets, dtype=np.float32)
        bs = int(TD_BATCH_SIZE)
        for start in range(0, n, bs):
            sel = idx[start:start + bs]
            self.train_step(states_all[sel], masks_all[sel], acts_all[sel], targs_all[sel])
        if self.on_update is not None:
            try:
                self.on_update()
            except Exception:
                pass
        self.clear()


class DFSEngine:
    """Generic DFS engine with decision-node TD(λ) backups."""

    def __init__(
        self,
        qnet: QNetwork,
        learn: bool,
        explore: bool,
        task: BranchingTaskState,
        branch_policy: BranchPolicy,
        behavior_policy: BehaviorPolicy,
        backup_policy: BackupPolicy,
        cache_max: int = CACHE_MAX,
    ):
        self.qnet = qnet
        self.model = qnet.model
        self.learn = bool(learn)
        self.explore = bool(explore)
        self.task = task
        self.branch_policy = branch_policy
        self.behavior_policy = behavior_policy
        self.backup_policy = backup_policy
        self.cache_max = int(cache_max)
        self.backs = 0
        self.contradictions = 0
        self.move_attempts = 0
        self.max_domain_size = 0
        self.q_cache: "OrderedDict[bytes, np.ndarray]" = OrderedDict()
        self.learner = Learner(
            self.model,
            task_spec=self.task.spec,
            enabled=self.learn,
            on_update=self.q_cache.clear,
        )

    def reset_ephemeral_state(self) -> None:
        """Clear per-run transient state after a failed solve."""
        self.q_cache.clear()
        self.learner.clear()
        self.backs = 0
        self.contradictions = 0
        self.move_attempts = 0
        self.max_domain_size = 0

    def update_msds(self, ds: int) -> None:
        """Update the max selected domain size."""
        self.max_domain_size = max(int(self.max_domain_size), int(ds))

    def solved_terminal_values(self) -> Tuple[float, float]:
        """Return solved reward and Φ."""
        return float(self.backup_policy.solved_reward()), float(self.backup_policy.phi(0.0, terminal=True))

    def contradiction_terminal_values(self) -> Tuple[float, float]:
        """Return contradiction reward and Φ."""
        return float(self.backup_policy.contradiction_reward()), float(self.backup_policy.phi(0.0, terminal=True))

    def cache_get_q(self, key: bytes) -> Optional[np.ndarray]:
        """Return cached Q values for a key."""
        q = self.q_cache.get(key)
        if q is None:
            return None
        self.q_cache.move_to_end(key, last=True)
        return q

    def cache_put_q(self, key: bytes, q: np.ndarray) -> None:
        """Store cached Q values for a key."""
        self.q_cache[key] = np.asarray(q, dtype=np.float32)
        self.q_cache.move_to_end(key, last=True)
        if len(self.q_cache) > self.cache_max:
            self.q_cache.popitem(last=False)

    def infer_q_cached(self, legal_mask: np.ndarray) -> np.ndarray:
        """Infer Q values with state caching."""
        key = self.task.cache_key()
        q = self.cache_get_q(key)
        if q is None:
            q = self.qnet.predict(self.task.state, legal_mask)
            self.cache_put_q(key, q)
        return q

    def greedy_actions_and_value(
        self,
        decision: Decision,
        actions: np.ndarray,
        q_vals: np.ndarray,
        eps_node: float,
    ) -> Tuple[np.ndarray, float]:
        """Return greedy-ranked actions and Vμ."""
        greedy = self.branch_policy.greedy_action_order(decision, actions, q_vals, self.task).astype(np.int32, copy=False)
        qd = self.task.action_values(decision, greedy, q_vals).astype(np.float64, copy=False)
        return greedy, float(self.behavior_policy.policy_value(qd, eps_node))

    def undo_prefix(self, prefix_steps: Sequence[Step]) -> None:
        """Undo a prefix of applied steps."""
        for step in reversed(prefix_steps):
            self.task.undo(step.decision, step.action)
            self.backs += 1

    def queue_if_learning(
        self,
        parent_state_np: Optional[np.ndarray],
        parent_mask_np: Optional[np.ndarray],
        action_idx: int,
        target: float,
    ) -> None:
        """Queue a target when learning is enabled."""
        if self.learn:
            self.learner.queue(parent_state_np, parent_mask_np, action_idx, float(target))

    def apply_all_forced_moves(
        self,
    ) -> Tuple[bool, List[Step], bool, float, float, Optional[ScanResult]]:
        """Apply forced moves to a fixpoint, learning only on terminal forced outcomes."""
        prefix_steps: List[Step] = []
        if self.task.is_solved():
            self.update_msds(1)
            term_reward, phi0 = self.solved_terminal_values()
            return True, prefix_steps, True, term_reward, phi0, None
        scan_curr: Optional[ScanResult] = None
        while True:
            if scan_curr is None:
                scan_curr = self.task.scan()
            if not scan_curr.ok:
                self.contradictions += 1
                self.undo_prefix(prefix_steps)
                term_reward, phi0 = self.contradiction_terminal_values()
                return False, [], True, term_reward, phi0, None
            if not scan_curr.forced_decisions:
                return True, prefix_steps, False, 0.0, 0.0, scan_curr
            self.update_msds(1)
            decision = self.branch_policy.choose_forced_decision(scan_curr.forced_decisions, self.task)
            if decision is None:
                self.undo_prefix(prefix_steps)
                raise RuntimeError("apply_all_forced_moves(): no forced decision.")
            acts = np.asarray(scan_curr.option_map.get(decision, ()), dtype=np.int32)
            if acts.size != 1:
                self.undo_prefix(prefix_steps)
                raise RuntimeError("apply_all_forced_moves(): forced decision is not singleton.")
            action = int(acts[0])
            step = Step(decision=decision, action=action)
            action_idx = int(self.task.action_index(decision, action))
            phi_s = float(self.backup_policy.phi(float(scan_curr.progress), terminal=False))
            parent_state_np = None
            parent_mask_np = None
            if self.learn:
                parent_state_np = self.task.state.copy()
                parent_mask_np = scan_curr.legal_mask.copy()
            self.move_attempts += 1
            self.task.apply(decision, action)
            prefix_steps.append(step)
            if self.task.is_solved():
                target = self.backup_policy.forced_terminal_target(phi_s, self.backup_policy.solved_reward())
                self.queue_if_learning(parent_state_np, parent_mask_np, action_idx, target)
                term_reward, phi0 = self.solved_terminal_values()
                return True, prefix_steps, True, term_reward, phi0, None
            if not self.learn:
                scan_curr = None
                continue
            scan_next = self.task.scan()
            if not scan_next.ok:
                target = self.backup_policy.forced_terminal_target(phi_s, self.backup_policy.contradiction_reward())
                self.queue_if_learning(parent_state_np, parent_mask_np, action_idx, target)
                self.contradictions += 1
                self.undo_prefix(prefix_steps)
                term_reward, phi0 = self.contradiction_terminal_values()
                return False, [], True, term_reward, phi0, None
            scan_curr = scan_next

    def build_decision_ctx(self, scan_curr: Optional[ScanResult] = None) -> DecisionCtx:
        """Build the current non-forced context."""
        if scan_curr is None:
            scan_curr = self.task.scan()
        if not scan_curr.ok:
            raise RuntimeError("build_decision_ctx(): contradiction state.")
        if scan_curr.forced_decisions:
            raise RuntimeError("build_decision_ctx(): forced decisions remain after forced-move propagation.")
        q = self.infer_q_cached(scan_curr.legal_mask)
        decision = self.branch_policy.choose_decision(scan_curr, q, self.task)
        if decision is None:
            raise RuntimeError("build_decision_ctx(): no decision.")
        acts = np.asarray(scan_curr.option_map.get(decision, ()), dtype=np.int32)
        if acts.size == 0:
            raise RuntimeError("build_decision_ctx(): no legal actions.")
        return DecisionCtx(
            scan=scan_curr,
            decision=decision,
            actions=acts,
            q=q,
            phi_val=float(self.backup_policy.phi(float(scan_curr.progress), terminal=False)),
        )

    def solve(self, task_input: Any) -> Tuple[bool, List[MoveRecord]]:
        """Solve one task instance."""
        self.task.reset(task_input)
        self.backs = 0
        self.contradictions = 0
        self.move_attempts = 0
        self.max_domain_size = 0
        self.q_cache.clear()
        self.learner.flush()
        if self.task.is_solved():
            self.update_msds(1)
            self.learner.flush()
            return True, []
        ok, steps, *_ = self.dfs()
        self.learner.flush()
        return ok, [self.task.move_record(s.decision, s.action) for s in steps]

    def dfs(self) -> Tuple[bool, List[Step], bool, float, float, float, float]:
        """Run one DFS step with decision-node TD(λ) backups."""
        if self.task.is_solved():
            self.update_msds(1)
            term_reward, phi0 = self.solved_terminal_values()
            return True, [], True, term_reward, phi0, 0.0, 0.0
        okp, prefix_steps, terminal_now, term_r_now, phi_term_now, scan_curr = self.apply_all_forced_moves()
        if not okp:
            return False, [], True, float(term_r_now), float(phi_term_now), 0.0, 0.0
        if terminal_now:
            return True, prefix_steps, True, float(term_r_now), float(phi_term_now), 0.0, 0.0
        ctx = self.build_decision_ctx(scan_curr)
        phi_here = float(ctx.phi_val)
        decision = ctx.decision
        actions = ctx.actions
        ds = int(actions.size)
        self.update_msds(ds)
        eps_node = self.behavior_policy.epsilon(self.learn, self.explore, is_choice=(ds > 1))
        greedy_actions, v_mu_here = self.greedy_actions_and_value(decision, actions, ctx.q, eps_node)
        order = self.behavior_policy.action_order(greedy_actions, eps_node)
        parent_state_np = None
        parent_mask_np = None
        if self.learn:
            parent_state_np = self.task.state.copy()
            parent_mask_np = ctx.scan.legal_mask.copy()
        for action in order:
            action = int(action)
            action_idx = int(self.task.action_index(decision, action))
            if not bool(ctx.scan.legal_mask[0, action_idx] > 0.5):
                continue
            self.move_attempts += 1
            self.task.apply(decision, action)
            this_step = Step(decision=decision, action=action)
            if self.task.is_solved():
                phi_child = 0.0
                r_term = float(self.backup_policy.solved_reward())
                target = self.backup_policy.branch_target(
                    phi_s=phi_here,
                    child_terminal=True,
                    child_term_reward=r_term,
                    phi_child=phi_child,
                    g_child=0.0,
                    v_mu_child=0.0,
                )
                self.queue_if_learning(parent_state_np, parent_mask_np, action_idx, target)
                return (
                    True,
                    prefix_steps + [this_step],
                    True,
                    r_term,
                    phi_child,
                    float(target),
                    float(v_mu_here),
                )
            ok2, tail, child_terminal, child_term_r, phi_child, g_child, v_mu_child = self.dfs()
            target = self.backup_policy.branch_target(
                phi_s=phi_here,
                child_terminal=bool(child_terminal),
                child_term_reward=float(child_term_r),
                phi_child=float(phi_child),
                g_child=float(g_child),
                v_mu_child=float(v_mu_child),
            )
            self.queue_if_learning(parent_state_np, parent_mask_np, action_idx, target)
            if ok2:
                return True, prefix_steps + [this_step] + tail, False, 0.0, phi_here, float(target), float(v_mu_here)
            self.task.undo(decision, action)
            self.backs += 1
        self.undo_prefix(prefix_steps)
        term_reward, phi0 = self.contradiction_terminal_values()
        return False, [], True, term_reward, phi0, 0.0, 0.0

# ──────────────────────────────────────────────────────────────────────────────
# Sudoku task adapter
# ──────────────────────────────────────────────────────────────────────────────
EYE9 = np.eye(9, dtype=np.float32)
ALL9 = (1 << 9) - 1
LOG9 = float(np.log(9.0))
MASK_TO_DIGITS = [
    np.array([k + 1 for k in range(9) if (m & (1 << k))], dtype=np.int32)
    for m in range(1 << 9)
]


def box_index(i: int, j: int) -> int:
    """Return the 3×3 box index."""
    return (int(i) // 3) * 3 + (int(j) // 3)


_CELL_ROWS = np.repeat(np.arange(9, dtype=np.int16), 9)
_CELL_COLS = np.tile(np.arange(9, dtype=np.int16), 9)
_CELL_BOXS = (_CELL_ROWS // 3) * 3 + (_CELL_COLS // 3)
_CELL_BASES = np.arange(81, dtype=np.int32) * 9
_FULL_PEER_MASK = (
    (_CELL_ROWS[:, None] == _CELL_ROWS[None, :]) |
    (_CELL_COLS[:, None] == _CELL_COLS[None, :]) |
    (_CELL_BOXS[:, None] == _CELL_BOXS[None, :])
)
np.fill_diagonal(_FULL_PEER_MASK, False)


def sudoku_peer_weight_matrix(cells: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Return the row-normalized peer inhibition matrix."""
    n = len(cells)
    if n <= 1:
        return np.zeros((n, n), dtype=np.float64)
    idx = np.fromiter(
        (int(r) * 9 + int(c) for (r, c) in cells),
        dtype=np.int16,
        count=n,
    )
    w = _FULL_PEER_MASK[np.ix_(idx, idx)].astype(np.float64, copy=True)
    row_sums = w.sum(axis=1, keepdims=True)
    np.divide(w, row_sums + float(EPS), out=w, where=(row_sums > 0.0))
    return w


def digits_from_mask(mask: int) -> np.ndarray:
    """Return legal digits from a 9-bit mask."""
    return MASK_TO_DIGITS[int(mask) & ALL9]


def encode_state(board: np.ndarray) -> np.ndarray:
    """Encode a Sudoku board as one-hot."""
    state = np.zeros((1, 9, 9, 9), dtype=np.float32)
    filled = board > 0
    if np.any(filled):
        state[0, filled] = EYE9[board[filled] - 1]
    return state


def set_cell_state(state: np.ndarray, i: int, j: int, d: int) -> None:
    """Update the one-hot state for one cell."""
    state[0, int(i), int(j), :] = 0.0
    if int(d):
        state[0, int(i), int(j), int(d) - 1] = 1.0


def init_masks(board: np.ndarray):
    """Initialize row, column, and box masks."""
    row_used = np.zeros(9, dtype=np.int32)
    col_used = np.zeros(9, dtype=np.int32)
    box_used = np.zeros(9, dtype=np.int32)
    row_cnt = np.zeros((9, 9), dtype=np.int16)
    col_cnt = np.zeros((9, 9), dtype=np.int16)
    box_cnt = np.zeros((9, 9), dtype=np.int16)
    for i in range(9):
        for j in range(9):
            d = int(board[i, j])
            if d:
                k = d - 1
                b = box_index(i, j)
                if row_cnt[i, k] or col_cnt[j, k] or box_cnt[b, k]:
                    raise ValueError("Invalid puzzle: duplicate digit in row/col/box.")
                row_cnt[i, k] += 1
                col_cnt[j, k] += 1
                box_cnt[b, k] += 1
                bit = 1 << k
                row_used[i] |= bit
                col_used[j] |= bit
                box_used[b] |= bit
    return row_used, col_used, box_used, row_cnt, col_cnt, box_cnt


def add_mask_digit(masks, i: int, j: int, d: int) -> None:
    """Add a placed digit to the masks."""
    row_used, col_used, box_used, row_cnt, col_cnt, box_cnt = masks
    k = int(d) - 1
    b = box_index(i, j)
    row_cnt[i, k] += 1
    col_cnt[j, k] += 1
    box_cnt[b, k] += 1
    bit = 1 << k
    row_used[i] |= bit
    col_used[j] |= bit
    box_used[b] |= bit


def remove_mask_digit(masks, i: int, j: int, d: int) -> None:
    """Remove a digit from the masks."""
    row_used, col_used, box_used, row_cnt, col_cnt, box_cnt = masks
    k = int(d) - 1
    b = box_index(i, j)
    row_cnt[i, k] -= 1
    col_cnt[j, k] -= 1
    box_cnt[b, k] -= 1
    bit = 1 << k
    if row_cnt[i, k] <= 0:
        row_used[i] &= ~bit
    if col_cnt[j, k] <= 0:
        col_used[j] &= ~bit
    if box_cnt[b, k] <= 0:
        box_used[b] &= ~bit


class BlankTracker:
    """O(1) blank add/remove tracker."""

    def __init__(self, blanks: Sequence[Tuple[int, int]]):
        self.blanks = list(blanks)
        self.pos = {cell: k for k, cell in enumerate(self.blanks)}

    def remove(self, cell: Tuple[int, int]) -> None:
        """Remove a blank cell."""
        k = self.pos.pop(cell)
        last = self.blanks.pop()
        if k < len(self.blanks):
            self.blanks[k] = last
            self.pos[last] = k

    def add(self, cell: Tuple[int, int]) -> None:
        """Add a blank cell."""
        if cell in self.pos:
            return
        self.pos[cell] = len(self.blanks)
        self.blanks.append(cell)


class SudokuTaskState(BranchingTaskState):
    """Sudoku adapter with incremental row/column/box masks."""

    spec = TaskSpec(state_shape=(9, 9, 9), action_dim=729)

    def __init__(self):
        self.board: Optional[np.ndarray] = None
        self.state: Optional[np.ndarray] = None
        self.masks = None
        self.blank_tracker: Optional[BlankTracker] = None

    def reset(self, board: np.ndarray) -> None:
        """Reset from a Sudoku board."""
        self.board = np.ascontiguousarray(board, dtype=np.uint8)
        self.state = np.ascontiguousarray(encode_state(self.board))
        self.masks = init_masks(self.board)
        blanks = [(int(i), int(j)) for (i, j) in np.argwhere(self.board == 0)]
        self.blank_tracker = BlankTracker(blanks)

    def is_solved(self) -> bool:
        """Return whether the Sudoku is solved."""
        return (self.blank_tracker is not None) and (not self.blank_tracker.blanks)

    def cache_key(self) -> bytes:
        """Return a bytes snapshot of the board."""
        return self.board.tobytes()

    def scan(self) -> ScanResult:
        """Scan legal actions and compute progress."""
        blanks = self.blank_tracker.blanks if self.blank_tracker is not None else []
        legal_mask = np.zeros((1, self.spec.action_dim), dtype=np.float32)
        option_map: OptionMap = {}
        forced_cells: List[Decision] = []
        if not blanks:
            return ScanResult(True, option_map, legal_mask, 1.0, forced_cells)
        row_used, col_used, box_used, _, _, _ = self.masks
        total_log_domain = 0.0
        count = 0
        for (i, j) in blanks:
            cell_idx = int(i) * 9 + int(j)
            used = int(row_used[i]) | int(col_used[j]) | int(box_used[_CELL_BOXS[cell_idx]])
            legal_bits = int(ALL9 & ~used)
            if legal_bits == 0:
                return ScanResult(False, {}, legal_mask, 0.0, [])
            base = int(_CELL_BASES[cell_idx])
            digits = digits_from_mask(legal_bits)
            option_map[(int(i), int(j))] = digits
            legal_mask[0, base + (digits - 1)] = 1.0
            ds = int(digits.size)
            if ds == 1:
                forced_cells.append((int(i), int(j)))
            total_log_domain += float(np.log(float(ds)))
            count += 1
        mean_log_domain = total_log_domain / float(max(count, 1))
        progress = float(np.clip(1.0 - np.clip(mean_log_domain / LOG9, 0.0, 1.0), 0.0, 1.0))
        return ScanResult(True, option_map, legal_mask, progress, forced_cells)

    def action_index(self, decision: Decision, action: Action) -> int:
        """Return the flattened action index."""
        i, j = int(decision[0]), int(decision[1])
        return i * 81 + j * 9 + (int(action) - 1)

    def action_values(self, decision: Decision, actions: np.ndarray, q_vals: np.ndarray) -> np.ndarray:
        """Return Q values for the legal digits."""
        acts = np.asarray(actions, dtype=np.int32)
        if acts.size == 0:
            return np.empty((0,), dtype=np.float64)
        if not (isinstance(q_vals, np.ndarray) and q_vals.shape == (self.spec.action_dim,)):
            return np.full((int(acts.size),), float(NEG_Q_CLAMP), dtype=np.float64)
        i, j = int(decision[0]), int(decision[1])
        base = int(_CELL_BASES[i * 9 + j])
        return sanitize_q_values(q_vals[base + (acts - 1)])

    def apply(self, decision: Decision, action: Action) -> None:
        """Apply a Sudoku digit placement."""
        i, j, d = int(decision[0]), int(decision[1]), int(action)
        self.board[i, j] = d
        set_cell_state(self.state, i, j, d)
        add_mask_digit(self.masks, i, j, d)
        self.blank_tracker.remove((i, j))

    def undo(self, decision: Decision, action: Action) -> None:
        """Undo a Sudoku digit placement."""
        i, j, d = int(decision[0]), int(decision[1]), int(action)
        self.board[i, j] = 0
        set_cell_state(self.state, i, j, 0)
        remove_mask_digit(self.masks, i, j, d)
        self.blank_tracker.add((i, j))

    def move_record(self, decision: Decision, action: Action) -> MoveRecord:
        """Return a user-visible move record."""
        return int(decision[0]), int(decision[1]), int(action)

# ──────────────────────────────────────────────────────────────────────────────
# Sudoku branch policy
# ──────────────────────────────────────────────────────────────────────────────
def divisive_soft_wta_cell_choice(option_map: OptionMap, q_vals: np.ndarray) -> Optional[Tuple[int, int]]:
    """Pick the cell with the largest normalized soft-WTA score."""
    if not option_map:
        return None
    items = list(option_map.items())
    cells = [cell for cell, _ in items]
    if not (isinstance(q_vals, np.ndarray) and q_vals.shape == (729,)):
        return min(cells, key=lambda cell: (cell[0], cell[1]))
    q_safe = sanitize_q_values(q_vals)
    raw_drives = np.empty((len(items),), dtype=np.float64)
    for k, ((i, j), digits) in enumerate(items):
        acts = np.asarray(digits, dtype=np.int32)
        if acts.size == 0:
            raw_drives[k] = float(NEG_Q_CLAMP)
            continue
        base = int(_CELL_BASES[int(i) * 9 + int(j)])
        raw_drives[k] = soft_wta_reduce(q_safe[base + (acts - 1)], beta=WTA_BETA)
    activity = stable_softmax(raw_drives, beta=WTA_BETA)
    weights = sudoku_peer_weight_matrix(cells)
    norm_scores = divisively_normalize_scores(
        activity,
        weights,
        gain=GAIN,
        offset=OFFSET_DENOM,
    )
    best_idx = min(
        range(len(cells)),
        key=lambda idx: (-norm_scores[idx], int(cells[idx][0]), int(cells[idx][1])),
    )
    return int(cells[best_idx][0]), int(cells[best_idx][1])


class SudokuDivNormSoftWTABranchPolicy(BranchPolicy):
    """Divisive soft-WTA cell choice with greedy digits."""

    def choose_decision(self, scan: ScanResult, q_vals: np.ndarray, task: BranchingTaskState) -> Optional[Decision]:
        """Choose a non-forced decision."""
        return divisive_soft_wta_cell_choice(scan.option_map, q_vals)

    def choose_forced_decision(self, forced_decisions: Sequence[Decision], task: BranchingTaskState) -> Optional[Decision]:
        """Choose a forced decision."""
        if not forced_decisions:
            return None
        return min(forced_decisions, key=lambda c: (int(c[0]), int(c[1])))

    def greedy_action_order(
        self,
        decision: Decision,
        actions: np.ndarray,
        q_vals: np.ndarray,
        task: BranchingTaskState,
    ) -> np.ndarray:
        """Sort digits by Q, tie-breaking by smaller digit."""
        acts = np.asarray(actions, dtype=np.int32)
        if acts.size <= 1:
            return acts.astype(np.int32, copy=False)
        qd = task.action_values(decision, acts, q_vals)
        return acts[np.lexsort((acts, -qd))].astype(np.int32, copy=False)

# ──────────────────────────────────────────────────────────────────────────────
# Solver factory
# ──────────────────────────────────────────────────────────────────────────────
def make_sudoku_solver(
    qnet: QNetwork,
    train: bool,
    explore: bool,
    *,
    branch_policy: BranchPolicy,
    behavior_policy: BehaviorPolicy,
    backup_policy: BackupPolicy,
) -> DFSEngine:
    """Build a Sudoku solver instance."""
    return DFSEngine(
        qnet=qnet,
        learn=train,
        explore=explore,
        task=SudokuTaskState(),
        branch_policy=branch_policy,
        behavior_policy=behavior_policy,
        backup_policy=backup_policy,
        cache_max=CACHE_MAX,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────────────────────────────────────
def draw(board, title: str):
    """Draw a Sudoku board."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    for k in range(10):
        lw = 2 if k % 3 == 0 else 0.5
        ax.plot([k, k], [0, 9], "k", lw=lw)
        ax.plot([0, 9], [k, k], "k", lw=lw)
    for r in range(9):
        for c in range(9):
            v = int(board[r, c])
            if v:
                ax.text(c + 0.5, 8.5 - r, v, ha="center", va="center", fontsize=10)
    ax.set_title(title, fontsize=10)
    plt.show()
    plt.close(fig)


def print_move_sequence(moves: Sequence[MoveRecord]) -> None:
    """Print the move sequence."""
    print("\nMove sequence:")
    for k, (r, c, d) in enumerate(moves, 1):
        print(f"{k:3d}. place {d} at (r{r+1}, c{c+1})")


def reconstruct_solution(board: np.ndarray, moves: Sequence[MoveRecord]) -> np.ndarray:
    """Apply moves to reconstruct the solution."""
    sol = board.copy()
    for r, c, d in moves:
        sol[r, c] = d
    return sol

# ──────────────────────────────────────────────────────────────────────────────
# Puzzle loading
# ──────────────────────────────────────────────────────────────────────────────
def load_puzzle_dir(directory: str) -> List[Tuple[str, np.ndarray]]:
    """Load 9×9 text puzzles from a directory."""
    puzzles: List[Tuple[str, np.ndarray]] = []
    try:
        paths = sorted(Path(directory).iterdir(), key=lambda p: p.name.casefold())
    except Exception:
        return puzzles
    for p in paths:
        if not p.is_file():
            continue
        try:
            board = np.loadtxt(p, dtype=int, comments="#")
            board = np.atleast_2d(board)
            if board.shape == (9, 9) and board.min() >= 0 and board.max() <= 9:
                puzzles.append((p.name, board))
        except Exception:
            pass
    return puzzles

# ──────────────────────────────────────────────────────────────────────────────
# Run + CLI
# ──────────────────────────────────────────────────────────────────────────────
def run_puzzle_with_solver(
    solver: DFSEngine,
    board: np.ndarray,
) -> None:
    """Run one puzzle with an existing solver."""
    exploration_active = bool(solver.learn and solver.explore)
    if SHOW_PLOT:
        draw(board, "\nSudoku Puzzle")
    done, moves = solver.solve(board)
    if not done:
        print("\nNo solution found.\n")
    print(f"\nMove attempts ({'exploration active' if exploration_active else 'exploration inactive'}): {solver.move_attempts}")
    print(f"Maximum selected domain size: {solver.max_domain_size}")
    print(f"Contradiction count: {solver.contradictions}")
    print(f"Backtrack count: {solver.backs}")
    if not done:
        return
    if SHOW_MOVES:
        print_move_sequence(moves)
    if SHOW_PLOT:
        draw(reconstruct_solution(board, moves), "\nProposed Solution")


def ask_int(prompt, lo=0):
    """Read an integer >= lo."""
    while True:
        try:
            v = int(input(prompt))
            if v >= lo:
                return v
        except Exception:
            pass
        print(f"... Enter an integer ≥ {lo}.")


def ask_float(prompt, lo=0.0, hi=1.0):
    """Read a float in [lo, hi]."""
    while True:
        try:
            v = float(input(prompt))
            if lo <= v <= hi:
                return v
        except Exception:
            pass
        print(f"... Enter a number in [{lo}, {hi}].")


def main():
    """CLI entry point."""
    directory = input("Puzzle directory: ").strip()
    if not os.path.isdir(directory):
        print("\nDirectory not found.")
        return
    puzzles = load_puzzle_dir(directory)
    if not puzzles:
        print("\nNo valid puzzles found.")
        return

    branch_policy = SudokuDivNormSoftWTABranchPolicy()
    behavior_policy = EpsilonGreedyFirstActionPolicy()
    backup_policy = TDLambdaExpectedBackup()

    while True:
        trials = ask_int("\nEnter the number of trials (0 to test & exit): ", lo=0)

        if trials == 0:
            qnet, _ = QNetwork.load_or_create(SudokuTaskState.spec, for_training=False)
            solver = make_sudoku_solver(
                qnet=qnet,
                train=False,
                explore=False,
                branch_policy=branch_policy,
                behavior_policy=behavior_policy,
                backup_policy=backup_policy,
            )
            n = len(puzzles)
            for k, (fn, board) in enumerate(sorted(puzzles, key=lambda x: x[0].casefold()), start=1):
                print(f"\nPuzzle {k}/{n}: {fn}")
                try:
                    run_puzzle_with_solver(solver, board)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"Skipped (runtime error): {type(e).__name__}: {e}")
                    traceback.print_exc()
                    try:
                        solver.reset_ephemeral_state()
                    except Exception:
                        pass
                    solver = make_sudoku_solver(
                        qnet=qnet,
                        train=False,
                        explore=False,
                        branch_policy=branch_policy,
                        behavior_policy=behavior_policy,
                        backup_policy=backup_policy,
                    )
            print("\nDone.")
            return

        lr = ask_float("\nEnter the learning rate (0–1): ", 0.0, 1.0)
        qnet, loaded_n = QNetwork.load_or_create(SudokuTaskState.spec, lr=lr, for_training=True)
        base_ckpt = int(loaded_n)
        solver = make_sudoku_solver(
            qnet=qnet,
            train=True,
            explore=True,
            branch_policy=branch_policy,
            behavior_policy=behavior_policy,
            backup_policy=backup_policy,
        )

        for t in range(1, trials + 1):
            print(f"\nTrial number: {t}/{trials}")
            n = len(puzzles)
            for k, (fn, board) in enumerate(puzzles, start=1):
                print(f"\nPuzzle {k}/{n}: {fn}")
                try:
                    run_puzzle_with_solver(solver, board)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"Skipped (runtime error): {type(e).__name__}: {e}")
                    traceback.print_exc()
                    try:
                        solver.reset_ephemeral_state()
                    except Exception:
                        pass
                    solver = make_sudoku_solver(
                        qnet=qnet,
                        train=True,
                        explore=True,
                        branch_policy=branch_policy,
                        behavior_policy=behavior_policy,
                        backup_policy=backup_policy,
                    )

            try:
                ck = qnet.save(idx=base_ckpt + t)
                print(f"\nCheckpoint saved: '{ck}'")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"\nCheckpoint save failed: {type(e).__name__}: {e}")
                traceback.print_exc()
                return

            random.shuffle(puzzles)


if __name__ == "__main__":
    main()
