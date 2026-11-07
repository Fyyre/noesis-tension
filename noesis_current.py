"""
- Noesis Tension Classifier — v0.3.4
- Author: James (noct-ml)
- Last Update: 2026/05/20 — 21:45 UTC
"""

import os
import gc
import math
import json
import hashlib
import statistics
import types
from collections import deque
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from dataclasses import dataclass, asdict

from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
#  Config
# -----------------------------

NOESIS_VERSION = "0.4.0"
SCHEMA_VERSION = "0.4.0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NOESIS_MODEL_NAME = os.environ.get(
    "NOESIS_MODEL_NAME",
    "meta-llama/Llama-3.1-8B-Instruct",
    # "mistralai/Mistral-7B-Instruct-v0.3",
)

# How many completion tokens to generate when capturing response_text
# MAX_NEW_TOKENS = 64
NOESIS_MAX_NEW_TOKENS = int(os.environ.get("NOESIS_MAX_NEW_TOKENS", "64"))

# Whether to request full attentions from the model (expensive!)
#USE_ATTENTIONS = True  # set True only when explicitly want head_conflict

# set this in the shell script (e.g., mistral_35.sh, llama_test.sh, etc..)
USE_ATTENTIONS = os.environ.get("NOESIS_USE_ATTENTIONS", "0") == "1"

# Only hook / sample every Nth layer for deltas & curvature
LAYER_STRIDE = 1  # 1 = all layers, 2 = every other, 3 = every third, etc.

MAX_CLASS_A: Optional[int] = (
    int(os.environ.get("MAX_CLASS_A", "")) if os.environ.get("MAX_CLASS_A") else None
)
MAX_CLASS_B: Optional[int] = (
    int(os.environ.get("MAX_CLASS_B", "")) if os.environ.get("MAX_CLASS_B") else None
)

# -----------------------------
#  Experiment metadata + decoding knobs (stored in traces)
# -----------------------------

NOESIS_EXPERIMENT_ID = os.environ.get("NOESIS_EXPERIMENT_ID")  # e.g. grid_v1_temp_sweep
NOESIS_SEED = (
    int(os.environ.get("NOESIS_SEED", "0")) if os.environ.get("NOESIS_SEED") else None
)

# If want REI / sweep experiments, flip DO_SAMPLE=1 and set temperature/top_p.
NOESIS_DO_SAMPLE = os.environ.get("NOESIS_DO_SAMPLE", "0") == "1"
NOESIS_TEMPERATURE = (
    float(os.environ["NOESIS_TEMPERATURE"])
    if os.environ.get("NOESIS_TEMPERATURE")
    else None
)
NOESIS_TOP_P = (
    float(os.environ["NOESIS_TOP_P"]) if os.environ.get("NOESIS_TOP_P") else None
)


def make_prompt_id(label: str, prompt: str) -> str:
    """Stable, short-ish prompt id so can group runs without remembering full text."""
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    return f"{label}_{h}"

def _mean_or_none(data) -> Optional[float]:
    """Mean of finite values, or None when nothing was measured."""
    if data is None:
        return None
    cleaned = []
    for x in data:
        if x is None:
            continue
        try:
            value = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            cleaned.append(value)
    if not cleaned:
        return None
    return float(statistics.fmean(cleaned))


def safe_fmean(data, default: float = 0.0) -> float:
    """
    Safe version of statistics.fmean that handles None, empty lists,
    and other edge cases gracefully.
    """
    if data is None:
        return float(default)
    try:
        cleaned = [float(x) for x in data if x is not None]
        if not cleaned:
            return float(default)
        return float(statistics.fmean(cleaned))
    except Exception:
        return float(default)

# -----------------------------
#  Classifier / baseline identity
# -----------------------------

# Increment when telemetry-only classification logic changes in a way that affects comparability.
CLASSIFIER_VERSION = "telemetry-v0.4.0"

# -----------------------------
#  Data structures
# -----------------------------


@dataclass
class LayerMetrics:
    index: int
    mean_delta: Optional[float]
    head_conflict: Optional[float]
    curvature: Optional[float]
    extra: Dict[str, Any]


@dataclass
class Trace:
    schema_version: str
    noesis_version: str
    model: Dict[str, Any]
    run: Dict[str, Any]
    tokens: Dict[str, Any]
    summary: Dict[str, float]
    layers: List[LayerMetrics]
    extras: Dict[str, Any]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "noesis_version": self.noesis_version,
            "model": self.model,
            "run": self.run,
            "tokens": self.tokens,
            "summary": self.summary,
            "layers": [asdict(layer) for layer in self.layers],
            "extras": self.extras,
        }

    def to_json_str(self, indent: int = 2) -> str:
        return json.dumps(self.to_json_dict(), indent=indent)


@dataclass
class TensionMetrics:
    prompt: str
    label: str
    mean_layer_delta: float
    max_layer_delta: float
    mean_logit_entropy: float
    final_token_entropy: float
    per_layer_mean_delta: List[float]

    mean_head_conflict: Optional[float]
    max_head_conflict: Optional[float]
    mean_curvature: float
    max_curvature: float
    per_layer_head_conflict: List[Optional[float]]
    per_layer_curvature: List[float]

    response_text: Optional[str] = None
    gen_params: Optional[Dict[str, Any]] = None
    gen_tokens: Optional[int] = None

    # ===== MoE routing metrics (optional, only populated for MoE models) =====
    per_layer_moe_routing_entropy: Optional[List[float]] = None
    per_layer_moe_top_expert_ids: Optional[List[List[int]]] = None
    num_experts: Optional[int] = None
    moe_step_stats: Optional[Dict[int, Dict[str, float]]] = None

    kv_norm_drift_history: Optional[List[float]] = None
    kv_coherence_history: Optional[List[float]] = None
    kv_mean_norm_history: Optional[List[float]] = None

    # ===== Per-token band telemetry (optional; schema v0.4) =====
    # Shape:
    # {
    #   "generated_token_count": int,
    #   "z_early": List[float],
    #   "z_mid": List[float],
    #   "z_late": List[float],
    #   "ranges": {"early":[lo,hi], "mid":[lo,hi], "late":[lo,hi]},
    #   "z_mode": "baseline" | "unscaled"
    # }
    telemetry_bands_series: Optional[Dict[str, Any]] = None

    # === Attention / KV cache features (from extended TensionTracer) ===
    attention_entropy_per_layer: Optional[List[float]] = None
    attention_drift_history: Optional[List[float]] = None
    kv_reuse_scores: Optional[List[float]] = None
    head_specialization: Optional[List[float]] = None

    # Robust depth summary. mean_layer_delta stays the mean of per-token means.
    # These are the series the indices actually score.
    median_layer_delta: float = 0.0
    max_median_layer_delta: float = 0.0
    per_layer_median_delta: Optional[List[float]] = None
    per_layer_mad_delta: Optional[List[float]] = None
    per_layer_relative_update: Optional[List[float]] = None
    per_layer_residual_rms: Optional[List[float]] = None
    per_layer_update_cosine: Optional[List[float]] = None
    per_layer_kl_norm: Optional[List[Optional[float]]] = None
    participation_ratio: Optional[Dict[str, Optional[float]]] = None

    vocab_size: Optional[int] = None
    mean_logit_entropy_norm: float = 0.0
    final_token_entropy_norm: float = 0.0
    prompt_mean_logit_entropy: Optional[float] = None
    prompt_final_logit_entropy: Optional[float] = None
    head_conflict_measured: bool = False

    decode_trajectory: Optional[Dict[str, Any]] = None
    regime_probe_summary: Optional[Dict[str, Any]] = None
    sensitivity: Optional[Dict[str, Any]] = None

    attention_sink_history: Optional[List[float]] = None
    attention_prompt_mass_history: Optional[List[float]] = None
    attention_recency_mass_history: Optional[List[float]] = None
    kv_write_norm_history: Optional[List[float]] = None
    kv_cache_mean_drift_history: Optional[List[float]] = None
    moe_layer_step_entropy: Optional[Dict[str, List[float]]] = None


@dataclass
class RegimeProbe:
    ent_hist: deque
    conf_hist: deque
    margin_hist: deque
    drift_hist: deque

    collapse_count: int = 0
    cooldown: int = 0
    bursts_fired: int = 0
    risk_run: int = 0

    # --- NEW: edge + hysteresis state ---
    prev_risk: bool = False
    risk_hold: int = 0  # keeps "risk" latched for N steps after first detect it

    def update_entropy(self, ent: float):
        # Legacy helper; safe to keep.
        if len(self.ent_hist) >= 2:
            mean_prev = sum(list(self.ent_hist)[:-1]) / max(len(self.ent_hist) - 1, 1)
            if ent < 0.85 * mean_prev:
                self.collapse_count += 1
            else:
                self.collapse_count = 0
        self.ent_hist.append(ent)

    def entropy_collapse(self, win: int = 8, factor: float = 0.85) -> bool:
        if len(self.ent_hist) < 2:
            return False
        prev = list(self.ent_hist)[-win - 1 : -1]
        if not prev:
            return False
        mean_prev = sum(prev) / len(prev)
        return self.ent_hist[-1] < factor * mean_prev

    def conf_trend(self) -> int:
        if len(self.conf_hist) < 2:
            return 0
        prev_mean = sum(list(self.conf_hist)[:-1]) / max(len(self.conf_hist) - 1, 1)
        delta = self.conf_hist[-1] - prev_mean
        return 1 if delta > 0 else (-1 if delta < 0 else 0)

    def drift_proxy(self) -> float:
        """
        Proxy for step-to-step drift in internal band activity.

        drift_hist stores RAW signal each step: (zm+zl).
        drift_proxy is abs(delta) between last two raw values.
        """
        if len(self.drift_hist) < 2:
            return 0.0
        return abs(float(self.drift_hist[-1]) - float(self.drift_hist[-2]))

    def drift_median(self, lookback: int = 24) -> float:
        """
        Robust baseline for drift_proxy.
        Uses median of recent drift_proxy values (computed from raw series).
        """
        raw = list(self.drift_hist)
        if len(raw) < 4:
            return 0.0

        # Build recent drift deltas from raw series
        deltas = [abs(float(raw[i]) - float(raw[i - 1])) for i in range(1, len(raw))]
        if not deltas:
            return 0.0

        tail = deltas[-lookback:] if len(deltas) > lookback else deltas
        try:
            return float(statistics.median(tail))
        except Exception:
            return float(sum(tail) / len(tail))

    def fp_proxy(self) -> bool:
        """
        High-margin proxy (token decisiveness). This is *very permissive* if run temp < 1.
        Probably want this combined with entropy collapse *in the DF-1 gate*, not here.
        """
        if not self.margin_hist:
            return False
        return self.margin_hist[-1] > 0.95

    def risk_state(self) -> bool:
        # strict lock-in detector (entropy floor + high confidence)
        if len(self.ent_hist) < 1 or len(self.conf_hist) < 1:
            return False
        return (
            (self.ent_hist[-1] < 0.15)
            and (self.conf_hist[-1] > 0.985)
            and self.entropy_collapse()
        )

    def tick(self):
        if self.cooldown > 0:
            self.cooldown -= 1
        if getattr(self, "risk_hold", 0) > 0:
            self.risk_hold -= 1


@dataclass(frozen=True)
class RegimeConfig:
    tension_hi: Tuple[float, float] = (0.55, 0.80)
    drift_hi: Tuple[float, float] = (0.55, 0.75)
    # Bands are on unclosed category scores in [0, 1], not on L1 shares.
    moral_paradox_band: Tuple[float, float] = (0.25, 0.55)
    false_presup_band: Tuple[float, float] = (0.20, 0.50)
    symbolic_band: Tuple[float, float] = (0.35, 0.60)
    kv_stable_band: Tuple[float, float] = (0.55, 0.80)
    kv_unstable_band: Tuple[float, float] = (0.40, 0.70)
    # Spike ratio is max(per-layer median RMS) / median(per-layer median RMS).
    spike_band: Tuple[float, float] = (1.5, 4.0)
    min_kv_tokens: int = 6
    version: str = "regime-v0.5"


@dataclass(frozen=True)
class RegimeFeatures:
    # Macro indices
    tension: float
    drift: float
    spike_ratio: float

    # Noesis category vector signals
    moral_paradox: float
    false_presupposition: float
    symbolism: float
    ambiguous_containment: float
    paradox_pressure: float
    symbolic_score: float          # = max(symbolism, ambiguous_containment, aesthetic_logic)

    # KV cache derived
    kv_stability: float
    kv_instability: float
    kv_norm_trend: float
    kv_reliable: bool
    kv_accumulating: bool
    kv_late_jump: bool
    kv_spiky: bool

    # Entropy / attention
    low_entropy: bool
    entropy_sustained: bool
    mean_attn_drift: float
    mean_kv_reuse: float

    # Baseline / policy
    in_factual_control_band: bool
    is_safety_liminal: bool
    label: Optional[str]           # e.g. "class_a"

    # Diagnostics (carried through to the output features dict)
    kv_recent_drift: float
    kv_final_drift: float
    kv_max_drift: float
    kv_recent_coherence: float
    kv_num_tokens: int


REGIME_CONFIG: RegimeConfig = RegimeConfig()

# -----------------------------
#  Noesis Neuro-Ontological Categories (v1)
# -----------------------------

NOESIS_CATEGORIES = {
    0: {
        "name": "Moral Paradox Tension",
        "description": "Ethical conflict, justification of dubious behavior, alignment friction.",
    },
    1: {
        "name": "False Presupposition Buckling",
        "description": "Factually wrong premises treated as true; scientific/real-world distortion.",
    },
    2: {
        "name": "Ontological Impossibility",
        "description": "Impossible entities or worlds; dream-creatures, impossible physics.",
    },
    3: {
        "name": "Anthropomorphic Metaphor Pressure",
        "description": "Physical/material entities described with human traits or family roles.",
    },
    4: {
        "name": "Category Collision",
        "description": "Blending incompatible domains (e.g., math + folklore, physics + myth).",
    },
    5: {
        "name": "Alignment Override Request",
        "description": "Implicit or explicit pressure to bypass safety/ethical constraints.",
    },
    6: {
        "name": "Emergent Symbolism Amplification",
        "description": "Strong symbolic or mythic content with low resistance from the model.",
    },
    7: {
        "name": "Ambiguous Containment",
        "description": "Prompts that can be answered literally or metaphorically with no clear cue.",
    },
    8: {
        "name": "Ontology Blur / Identity Bleed",
        "description": "Self-referential or agentic questions about the model's own mind or being.",
    },
    9: {
        "name": "Paradox Pressure",
        "description": "Logical paradoxes, self-reference loops, contradictory truths.",
    },
    10: {
        "name": "Aesthetic-Logic Crosswiring",
        "description": "Poetic/aesthetic prompts with logical or analytic demands.",
    },
    11: {
        "name": "Ungrounded Causality",
        "description": "Causal claims that operate in symbolic/metaphysical space, not physics.",
    },
}


# -----------------------------
#  Tension Tracer
# -----------------------------


class TensionTracer:
    def __init__(self, model):
        self.model = model

        self.kv_norm_drift_history: List[float] = []      # |Δ| of new-token KV write norm
        self.kv_coherence_history: List[float] = []       # rolling coherence of write drift
        self.kv_mean_norm_history: List[float] = []       # mean norm of the whole cache (diagnostic)
        self.kv_write_norm_history: List[float] = []      # norm of the newly appended K and V
        self.kv_cache_mean_drift_history: List[float] = []  # diluted cache-mean drift (diagnostic)
        self._last_kv_write_norm: Optional[float] = None
        self._last_kv_mean_norm: Optional[float] = None
        self._kv_window_size: int = 8
        self.prompt_len: int = 0
        self.moe_record_decode: bool = False

        # Decode-step index used for per-token telemetry and MoE step aggregation.
        self._decode_step: int = 0

        # Optional MoE gate diversification knobs (only used if MoE is present and patched)
        self.moe_gate_noise_sigma: float = 0.0
        self.moe_gate_bias_strength: float = 0.0
        self._moe_top1_last: dict[int, int] = {}
        self._moe_top1_streak: dict[int, int] = {}

        # Hooks
        self._hooks: List[Any] = []

        # Layer inventory
        self.num_layers_total: int = 0
        self.hooked_layer_indices: List[int] = []

        # Capture mode:
        # "full" = store full hidden tensors per layer (indexed)
        # "step" = store per-layer scalar per decode step (indexed)
        self.capture_mode: str = "full"

        # Full-pass capture (indexed by layer_idx, length = num_layers_total)
        self.full_hidden_by_layer: List[Optional[torch.Tensor]] = []

        # Step capture (indexed by layer_idx, length = num_layers_total)
        self.step_layer_scalar: List[Optional[float]] = []

        # Optional baseline stats for z-normalization (v0.4)
        # shape: {"early":{"mu":..,"sigma":..}, "mid":..., "late":...}
        self.band_baseline_stats: Optional[Dict[str, Dict[str, float]]] = None
        self.band_z_mode: str = "unscaled"  # "baseline" or "unscaled"

        # Legacy (kept for compatibility; not used by the new pipeline)
        self.hidden_by_layer: List[torch.Tensor] = []

        # --- MoE routing stuff ---
        self.moe_step_stats: Dict[int, Dict[str, float]] = {}
        self._moe_step_acc: Optional[dict] = None
        self.moe_gate_trace: Dict[int, Dict[str, torch.Tensor]] = {}
        self.moe_decode_trace: Dict[int, Dict[str, List[Any]]] = {}
        self.moe_top_k: int = 2
        self._moe_patched: bool = False
        self.moe_num_experts: Optional[int] = None

        # === NEW: KV Cache + Attention features ===
        self.attention_maps: List[torch.Tensor] = []  # last-layer, last-token attention per decode step
        self._attn_current: Dict[int, torch.Tensor] = {}
        self._attn_prev: Optional[torch.Tensor] = None  # [heads, seq] from the previous step
        self.attention_entropy_per_layer: List[float] = []  # last-layer entropy per decode step
        self.attention_drift_history: List[float] = []  # aligned step-to-step attention change
        self.kv_reuse_scores: List[float] = []  # 1 - aligned attention drift
        self.head_specialization: List[float] = []  # mean attention entropy over heads, per step
        self.attention_sink_history: List[float] = []
        self.attention_prompt_mass_history: List[float] = []
        self.attention_recency_mass_history: List[float] = []

    # ---------------------------
    # Hook logic (layer-indexed)
    # ---------------------------
    def _ensure_buffers(self):
        if self.num_layers_total <= 0:
            self.num_layers_total = len(self._get_layers())

        n = self.num_layers_total
        if len(self.full_hidden_by_layer) != n:
            self.full_hidden_by_layer = [None] * n
        if len(self.step_layer_scalar) != n:
            self.step_layer_scalar = [None] * n

    def _hook_block(self, layer_idx: int):
        def hook(module, input, output):
            is_tuple = isinstance(output, tuple)
            is_list = isinstance(output, list)

            # ---- Robust hidden extraction ----
            hidden = output
            if is_tuple or is_list:
                hidden = output[0]
            elif hasattr(output, "last_hidden_state"):
                hidden = output.last_hidden_state

            if hidden is None or not hasattr(hidden, "dim"):
                return output

            # ---- Capture buffers ----
            self._ensure_buffers()

            if self.capture_mode == "full":
                self.full_hidden_by_layer[layer_idx] = hidden.detach()
            else:
                h_last2 = hidden[:, -1, :] if hidden.dim() == 3 else hidden
                v = h_last2.norm(dim=-1).mean().item()
                self.step_layer_scalar[layer_idx] = float(v)

            # ---- Propagate modified hidden forward safely ----
            if is_tuple:
                return (hidden,) + tuple(output[1:])
            if is_list:
                out_list = list(output)
                out_list[0] = hidden
                return out_list
            if hasattr(output, "last_hidden_state"):
                return output

            return hidden

        return hook

    def reset_attention_features(self):
        """Reset all attention + KV features between prompts."""
        self.attention_maps = []
        self._attn_current = {}
        self._attn_prev = None
        self.attention_entropy_per_layer = []
        self.attention_drift_history = []
        self.kv_reuse_scores = []
        self.head_specialization = []
        self.attention_sink_history = []
        self.attention_prompt_mass_history = []
        self.attention_recency_mass_history = []

        self.kv_norm_drift_history = []
        self.kv_coherence_history = []
        self.kv_mean_norm_history = []
        self.kv_write_norm_history = []
        self.kv_cache_mean_drift_history = []
        self._last_kv_write_norm = None
        self._last_kv_mean_norm = None

    def _hook_attention(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]  # [B, heads, seq, seq]
                if attn_weights is not None:
                    # Keep the latest layer's last-token row. Comparing adjacent
                    # entries of a per-layer list mixes depth with decode time.
                    last_attn = attn_weights[:, :, -1:, :].detach().cpu()  # [B, heads, 1, seq]
                    self._attn_current[int(layer_idx)] = last_attn
            return output

        return hook

    # ====================== NEW KV CACHE HOOK ======================
    def _hook_kv_cache(self, layer_idx: int):
        """Hook for KV cache norm drift (only on last layer for efficiency)."""
        def hook(module, input, output):
            if hasattr(output, "past_key_values") and output.past_key_values is not None:
                self._capture_kv_drift(output.past_key_values)
            return output
        return hook

    def _capture_kv_drift(self, past_kv):
        """KV telemetry via newly written token and the cache-mean diagnostic.

        Scored drift is the step-to-step change in the norm of the key and value
        just appended. Mean over the whole cache shrinks as 1/T and is kept only
        as kv_cache_mean_drift_history.
        """
        if past_kv is None:
            return

        # Newer Transformers may return a Cache object.
        if hasattr(past_kv, "to_legacy_cache"):
            try:
                past_kv = past_kv.to_legacy_cache()
            except Exception:
                return

        if not past_kv or len(past_kv) == 0:
            return

        cache_norms = []
        write_norms = []
        for layer_kv in past_kv:  # usually (key, value) per layer
            if not isinstance(layer_kv, (tuple, list)) or len(layer_kv) < 2:
                continue
            k, v = layer_kv[:2]
            if k is None or v is None or k.numel() == 0:
                continue
            cache_norms.append(
                float(torch.norm(k.float(), dim=-1).mean().item()
                      + torch.norm(v.float(), dim=-1).mean().item())
            )
            # Last position is the write from this step. Shape [B, heads, seq, dim].
            if k.dim() >= 3 and k.shape[-2] >= 1:
                k_new = k[..., -1, :]
                v_new = v[..., -1, :]
                write_norms.append(
                    float(torch.norm(k_new.float(), dim=-1).mean().item()
                          + torch.norm(v_new.float(), dim=-1).mean().item())
                )

        current_mean_norm = float(sum(cache_norms) / len(cache_norms)) if cache_norms else 0.0
        self.kv_mean_norm_history.append(current_mean_norm)
        if self._last_kv_mean_norm is not None:
            self.kv_cache_mean_drift_history.append(abs(current_mean_norm - self._last_kv_mean_norm))
        else:
            self.kv_cache_mean_drift_history.append(0.0)
        self._last_kv_mean_norm = current_mean_norm

        current_write = float(sum(write_norms) / len(write_norms)) if write_norms else current_mean_norm
        self.kv_write_norm_history.append(current_write)
        if self._last_kv_write_norm is not None:
            drift = abs(current_write - self._last_kv_write_norm)
        else:
            drift = 0.0
        self.kv_norm_drift_history.append(drift)
        self._last_kv_write_norm = current_write

        if len(self.kv_norm_drift_history) >= 2:
            window = self.kv_norm_drift_history[-self._kv_window_size:]
            mean_win = statistics.fmean(window)
            std_win = statistics.pstdev(window) if len(window) > 1 else 0.0
            coherence = 1.0 - (std_win / (mean_win + 1e-8))
            self.kv_coherence_history.append(max(0.0, min(1.0, coherence)))
        else:
            self.kv_coherence_history.append(1.0)

    def _get_layers(self):
        """Return a list-like of transformer blocks for common HF architectures.

        This must succeed for band telemetry; if it returns [], hooks won't register and
        step_layer_scalar will stay None -> band means 0.0.
        """
        m = self.model

        # Most common: Llama/Mistral/Qwen etc: model.model.layers
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers

        # Some models expose layers directly
        if hasattr(m, "layers"):
            return m.layers

        # GPT-2 style: model.transformer.h
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h

        # OPT/Bloom style: model.model.decoder.layers or model.decoder.layers
        if (
            hasattr(m, "model")
            and hasattr(m.model, "decoder")
            and hasattr(m.model.decoder, "layers")
        ):
            return m.model.decoder.layers
        if hasattr(m, "decoder") and hasattr(m.decoder, "layers"):
            return m.decoder.layers

        # GPT-NeoX style: model.gpt_neox.layers
        if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
            return m.gpt_neox.layers

        # Falcon style: model.transformer.h (already handled), sometimes model.transformer.layers
        if hasattr(m, "transformer") and hasattr(m.transformer, "layers"):
            return m.transformer.layers

        # As a last resort, try get_decoder()
        if hasattr(m, "get_decoder"):
            try:
                dec = m.get_decoder()
                if hasattr(dec, "layers"):
                    return dec.layers
            except Exception:
                pass

        return []

    # ---------------------------
    # Registration / lifecycle
    # ---------------------------
    def register(self, layer_stride: int = 1):
        self.remove()
        layers = self._get_layers()
        self.num_layers_total = len(layers)
        self.hooked_layer_indices = []

        self._ensure_buffers()

        for i, block in enumerate(layers):
            if layer_stride != 1 and (i % layer_stride != 0):
                continue

            # Existing block hook
            h_block = block.register_forward_hook(self._hook_block(i))
            self._hooks.append(h_block)
            self.hooked_layer_indices.append(i)

            if USE_ATTENTIONS:
                attn_module = None
                if hasattr(block, "self_attn"):
                    attn_module = block.self_attn
                elif hasattr(block, "attn"):
                    attn_module = block.attn

                if attn_module is not None:
                    h_attn = attn_module.register_forward_hook(self._hook_attention(i))
                    self._hooks.append(h_attn)

        self.hidden_by_layer = []

        # MoE hooks were wiped by remove(); re-attach them if previously requested.
        if getattr(self, "_moe_tracing_requested", False):
            self.enable_moe_tracing(top_k=self.moe_top_k)

    def remove(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []
        self.attention_maps.clear()
        self._attn_current = {}
        self._attn_prev = None
        self._moe_patched = False

    def clear(self):
        self.hidden_by_layer = []
        self.full_hidden_by_layer = [None] * self.num_layers_total
        self.step_layer_scalar = [None] * self.num_layers_total
        self.attention_maps.clear()
        self._attn_current = {}
        self._attn_prev = None
        self.attention_entropy_per_layer.clear()
        self.attention_drift_history.clear()
        self.kv_reuse_scores.clear()
        self.head_specialization.clear()
        self.attention_sink_history.clear()
        self.attention_prompt_mass_history.clear()
        self.attention_recency_mass_history.clear()
        self.moe_gate_trace = {}
        self.moe_decode_trace = {}
        self.moe_step_stats = {}

    def compute_attention_features(self):
        """Last-layer attention entropy, aligned step drift, sink mass, and recency mass."""
        if not self._attn_current:
            return
        layer_idx = max(self._attn_current)
        last_map = self._attn_current[layer_idx]  # [B, heads, 1, seq]
        self._attn_current = {}
        if last_map is None or last_map.numel() == 0:
            return
        self.attention_maps.append(last_map)

        attn = last_map[0, :, 0, :].float()  # [heads, seq]
        attn_sum = attn.sum(dim=-1, keepdim=True) + 1e-12
        attn = attn / attn_sum
        entropy = - (attn * torch.log(attn + 1e-12)).sum(dim=-1)
        mean_entropy = float(entropy.mean().item())
        self.attention_entropy_per_layer.append(mean_entropy)
        self.head_specialization.append(mean_entropy)

        seq = int(attn.shape[-1])
        sink = float(attn[:, 0].mean().item()) if seq > 0 else 0.0
        self.attention_sink_history.append(sink)

        prompt_len = int(getattr(self, "prompt_len", 0) or 0)
        prompt_end = min(max(prompt_len, 0), seq)
        if prompt_end > 0:
            prompt_mass = float(attn[:, :prompt_end].sum(dim=-1).mean().item())
        else:
            prompt_mass = 0.0
        self.attention_prompt_mass_history.append(prompt_mass)

        recency_k = 8
        gen = attn[:, prompt_end:]
        if gen.shape[-1] > 0:
            recency = float(gen[:, -min(recency_k, gen.shape[-1]):].sum(dim=-1).mean().item())
        else:
            recency = 0.0
        self.attention_recency_mass_history.append(recency)

        if self._attn_prev is not None:
            drift = _attention_step_drift(self._attn_prev, attn)
        else:
            drift = 0.0
        self._attn_prev = attn.detach()
        self.attention_drift_history.append(drift)
        self.kv_reuse_scores.append(1.0 - drift)

    # ---------------------------
    # Decode-step telemetry controls
    # ---------------------------
    def set_decode_step(self, step: int):
        """Mark the current decode step and reset per-step capture buffers.

        This MUST be called before the forward pass for that step so layer hooks write into a
        fresh buffer. Without this, step_layer_scalar can remain all-None, producing NaN bands.
        """
        self._decode_step = int(step)

        # Ensure indexed buffers exist and reset per-step scalar capture.
        self._ensure_buffers()
        self.capture_mode = "step"
        self.step_layer_scalar = [None] * self.num_layers_total

        # Reset per-step MoE accumulators used for routing stats.
        self._moe_step_acc = {
            "n": 0,
            "entropy_sum": 0.0,
            "top1_counts": {},  # expert_id -> count
            "top1_prob_sum": 0.0,
            "top2_prob_sum": 0.0,
        }

    def set_mode_step(self):
        """Backward-compat alias.

        Older decode loops called set_mode_step() to reset per-step state before a forward.
        We now key everything off set_decode_step(step); this helper preserves the old call site
        by resetting the per-step accumulators while keeping the current decode step (or -1).
        """
        step = getattr(self, "_decode_step", -1)
        self.set_decode_step(step)

    def clear_step(self):
        """Backward-compat: reset per-step scratch state used during decoding.

        Some call sites reset step state via clear_step()/set_mode_step() pairs. To keep the
        step-band telemetry reliable, we also reset the per-step scalar buffer here.
        """
        self._ensure_buffers()
        self.capture_mode = "step"
        self.step_layer_scalar = [None] * self.num_layers_total

        # Do not change decode step here; decode loop will call set_decode_step(step) explicitly.
        self._moe_step_acc = {
            "n": 0,
            "entropy_sum": 0.0,
            "top1_counts": {},  # expert_id -> count
            "top1_prob_sum": 0.0,
            "top2_prob_sum": 0.0,
        }

    def get_step_band_z(self, ranges: dict) -> dict:
        out = {}
        for band in ("early", "mid", "late"):
            a, b = ranges[band]
            seg = self.step_layer_scalar[a : b + 1]
            total = len(seg)
            vals = [v for v in seg if v is not None]

            coverage = (len(vals) / total) if total else 0.0
            out[f"coverage_{band}"] = float(coverage)

            if not vals:
                out[f"z_{band}"] = float("nan")
                continue

            mean = float(sum(vals) / len(vals))
            z = mean
            if (
                self.band_z_mode == "baseline"
                and self.band_baseline_stats
                and band in self.band_baseline_stats
            ):
                mu = float(self.band_baseline_stats[band].get("mu", 0.0))
                sigma = float(self.band_baseline_stats[band].get("sigma", 1.0))
                if sigma <= 1e-12:
                    sigma = 1.0
                z = (mean - mu) / sigma

            out[f"z_{band}"] = float(z)

        return out

    def finalize_step_stats(self):
        # Aggregate MoE routing stats for the current decode step (if any MoE layers reported).
        if not self._moe_step_acc:
            return
        n = int(self._moe_step_acc.get("n", 0))
        if n <= 0:
            return

        ent_mean = float(self._moe_step_acc["entropy_sum"]) / n
        top1_prob_mean = float(self._moe_step_acc["top1_prob_sum"]) / n
        top2_prob_mean = float(self._moe_step_acc.get("top2_prob_sum", 0.0)) / n
        counts = self._moe_step_acc.get("top1_counts", {}) or {}
        unique_top1 = float(len(counts))
        mode_frac = 0.0
        if counts:
            mode_frac = float(max(counts.values())) / float(sum(counts.values()))

        self.moe_step_stats[int(self._decode_step)] = {
            "routing_entropy_mean": ent_mean,
            "routing_top1_prob_mean": top1_prob_mean,
            "routing_top2_prob_mean": top2_prob_mean,
            "routing_margin_mean": float(top1_prob_mean - top2_prob_mean),
            "routing_top1_unique": unique_top1,
            "routing_top1_mode_frac": float(mode_frac),
        }

    def enable_moe_tracing(self, top_k: int = 2):
        """
        Hook MoE gate submodules to capture routing logits / experts.
        Safe to call multiple times (idempotent within a single register() lifetime).
        """
        self._moe_tracing_requested = True  # remember intent across register() resets

        if self._moe_patched:
            return

        self.moe_top_k = top_k
        self.moe_gate_trace = {}
        self.moe_decode_trace = {}

        def patch_moe_gate(moe_block, layer_index: int):
            gate_module = getattr(moe_block, "gate", None)
            if gate_module is None:
                return False

            def gate_post_hook(module, inputs, output):
                try:


                    gate_logits = output
                    # Some MoE gate modules return a tuple, e.g. (logits, aux_loss).
                    if isinstance(gate_logits, (tuple, list)):
                        gate_logits = gate_logits[0]
                    if not hasattr(gate_logits, "dim"):
                        return

                    if self.moe_num_experts is None:
                        self.moe_num_experts = int(gate_logits.size(-1))
                        print(f"[MoE] ✓ num_experts discovered = {self.moe_num_experts} (layer {layer_index})")

                    with torch.no_grad():
                        if gate_logits.dim() == 3:
                            gl_last = gate_logits[:, -1, :]
                        elif gate_logits.dim() == 2:
                            gl_last = gate_logits
                        else:
                            return

                        p = F.softmax(gl_last.float(), dim=-1)
                        p_mean = p.mean(dim=0)
                        entropy = float(-(p_mean * p_mean.clamp(min=1e-9).log()).sum().item())
                        k_top = min(2, int(p_mean.numel()))
                        top_probs, top_idx = torch.topk(p_mean, k=k_top)
                        top1 = int(top_idx[0].item())
                        top1_prob = float(top_probs[0].item())
                        top2_prob = float(top_probs[1].item()) if k_top > 1 else 0.0

                        # Prompt prefill and the warm-start forward hit this hook.
                        # Only the per-token decode loop opts in, so the path series
                        # stays on the same clock as the step stats.
                        if self._moe_step_acc is not None and getattr(self, "moe_record_decode", False):
                            self._moe_step_acc["n"] += 1
                            self._moe_step_acc["entropy_sum"] += entropy
                            self._moe_step_acc["top1_prob_sum"] += top1_prob
                            self._moe_step_acc["top2_prob_sum"] = float(self._moe_step_acc.get("top2_prob_sum", 0.0)) + top2_prob
                            c = self._moe_step_acc["top1_counts"]
                            c[top1] = c.get(top1, 0) + 1

                            layer_trace = self.moe_decode_trace.setdefault(
                                int(layer_index),
                                {"entropy": [], "top1": [], "top1_prob": [], "top2_prob": [], "step": []},
                            )
                            layer_trace["entropy"].append(float(entropy))
                            layer_trace["top1"].append(int(top1))
                            layer_trace["top1_prob"].append(float(top1_prob))
                            layer_trace["top2_prob"].append(float(top2_prob))
                            layer_trace["step"].append(int(self._decode_step))

                        top_scores, top_indices = gate_logits.topk(self.moe_top_k, dim=-1)
                        gates = F.softmax(top_scores, dim=-1)
                        self.moe_gate_trace[layer_index] = {
                            "gates": gates.detach().cpu(),
                            "indices": top_indices.detach().cpu(),
                        }
                except Exception as e:
                    print(f"[MoE] ✗ gate hook FAILED layer={layer_index}: {type(e).__name__}: {e}")

            h = gate_module.register_forward_hook(gate_post_hook)
            self._hooks.append(h)
            return True

        layers = self._get_layers()

        moe_found = False
        for idx, layer in enumerate(layers):
            moe_block = None
            moe_name = None

            if hasattr(layer, "block_sparse_moe"):
                moe_block = layer.block_sparse_moe
                moe_name = "block_sparse_moe"
            elif hasattr(layer, "moe") and hasattr(getattr(layer, "moe", None), "gate"):
                moe_block = layer.moe
                moe_name = "moe"
            elif hasattr(layer, "mlp") and hasattr(getattr(layer, "mlp", None), "gate"):
                moe_block = layer.mlp
                moe_name = "mlp"
            elif hasattr(layer, "ffn") and hasattr(getattr(layer, "ffn", None), "gate"):
                moe_block = layer.ffn
                moe_name = "ffn"

            # Class-name fallback for Qwen1.5-MoE etc.
            if moe_block is None:
                for attr_name in ["mlp", "block_sparse_moe", "moe", "ffn", "sparse_moe", "moe_block"]:
                    if hasattr(layer, attr_name):
                        candidate = getattr(layer, attr_name)
                        if candidate is not None:
                            cname = type(candidate).__name__.lower()
                            if any(x in cname for x in ["moe", "sparse", "mixture", "expert", "qwen2moe"]):
                                if hasattr(candidate, "gate") or hasattr(candidate, "router"):
                                    moe_block = candidate
                                    moe_name = f"{attr_name} (class:{type(candidate).__name__})"
                                    break

            if moe_block is None:
                continue

            if patch_moe_gate(moe_block, idx):
                moe_found = True
#               print(f"[MoE]: Patched layer {idx} via {moe_name}.gate")

        self._moe_patched = True

        # Config fallback
        if self.moe_num_experts is None:
            cfg = getattr(self.model, "config", None)
            if cfg is not None:
                for key in ["num_experts", "moe_num_experts", "n_routed_experts", "num_local_experts"]:
                    if hasattr(cfg, key):
                        val = getattr(cfg, key)
                        if isinstance(val, int) and val > 0:
                            self.moe_num_experts = val
                            print(f"[MoE]: Fallback num_experts = {val} from config.{key}")
                            break

        if not moe_found:
            print("[MoE]: No MoE blocks detected!")


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _robust_center_scale(xs: List[float]) -> Tuple[float, float]:
    """Median and MAD scaled to a normal-consistent sigma. Falls back to pstdev."""
    cleaned = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not cleaned:
        return 0.0, 1.0
    center = float(statistics.median(cleaned))
    mad = float(statistics.median([abs(x - center) for x in cleaned]))
    scale = 1.4826 * mad
    if scale < 1e-8:
        scale = float(statistics.pstdev(cleaned)) if len(cleaned) > 1 else 0.0
    if scale < 1e-8:
        scale = 1.0
    return center, scale


def _occupancy(ids: List[int], n_experts: int) -> Optional[List[float]]:
    if n_experts <= 0:
        return None
    counts = [0] * n_experts
    for e in ids:
        try:
            ei = int(e)
        except (TypeError, ValueError):
            continue
        if 0 <= ei < n_experts:
            counts[ei] += 1
    total = sum(counts)
    if total <= 0:
        return None
    return [c / total for c in counts]


def _mean_occupancy(vecs: List[List[float]]) -> Optional[List[float]]:
    if not vecs:
        return None
    width = len(vecs[0])
    acc = [0.0] * width
    for vec in vecs:
        if len(vec) != width:
            return None
        for i, value in enumerate(vec):
            acc[i] += value
    n = float(len(vecs))
    return [v / n for v in acc]


def _total_variation(p: List[float], q: List[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def _gini_from_counts(counts: List[float]) -> float:
    """Gini over the full expert pool, rescaled so a one-expert monopoly is 1.

    The raw Gini of a monopoly on a pool of size n is (n-1)/n. Dividing by that
    maps the glossary's endpoints onto 0 (uniform) and 1 (single expert).
    Zeros must be included. Dropping them makes a monopoly look uniform.
    """
    n = len(counts)
    if n == 0:
        return 0.0
    total = float(sum(counts))
    if total <= 0.0 or n == 1:
        return 0.0
    ascending = sorted(float(v) for v in counts)
    weighted = sum((i + 1) * v for i, v in enumerate(ascending))
    raw = (2.0 * weighted) / (n * total) - (n + 1.0) / n
    raw = max(0.0, min(1.0, raw))
    ceiling = (n - 1.0) / n
    if ceiling <= 0.0:
        return 0.0
    return max(0.0, min(1.0, raw / ceiling))


def _ols_slope(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = statistics.fmean(values)
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n)) or 1.0
    return float(num / den)


def _routing_shape(early: float, mid: float, late: float, margin: float = 0.05) -> Tuple[str, float]:
    """Classify a normalized routing-entropy profile.

    A U-shape requires the middle to sit below both ends. A monotone drop is
    not a U just because the middle minimum is low.
    """
    u_index = (early + late) / 2.0 - mid
    if mid < early - margin and mid < late - margin:
        return "u_shape", u_index
    if mid > early + margin and mid > late + margin:
        return "inverted_u", u_index
    if late < early - margin:
        return "monotonic_decrease", u_index
    if late > early + margin:
        return "monotonic_increase", u_index
    return "flat", u_index


def _expert_path_stats(id_lists: List[List[int]], n_experts: Optional[int]) -> Dict[str, Optional[float]]:
    """Stickiness and conditional transition entropy of per-layer top-1 paths."""
    rates: List[float] = []
    entropies: List[float] = []
    for ids in id_lists:
        if ids is None or len(ids) < 2:
            continue
        pairs = [(int(a), int(b)) for a, b in zip(ids, ids[1:])]
        if not pairs:
            continue
        stays = sum(1 for a, b in pairs if a == b)
        rates.append(stays / len(pairs))
        by_prev: Dict[int, List[int]] = {}
        for prev, nxt in pairs:
            by_prev.setdefault(prev, []).append(nxt)
        cond = 0.0
        for succs in by_prev.values():
            n = len(succs)
            counts: Dict[int, int] = {}
            for s in succs:
                counts[s] = counts.get(s, 0) + 1
            h = 0.0
            for c in counts.values():
                p = c / n
                h -= p * math.log(p + 1e-12)
            cond += (n / len(pairs)) * h
        entropies.append(cond)
    h_max = math.log(n_experts) if n_experts and n_experts > 1 else None
    mean_h = float(statistics.fmean(entropies)) if entropies else None
    return {
        "self_transition_rate": float(statistics.fmean(rates)) if rates else None,
        "transition_entropy": mean_h,
        "transition_entropy_norm": (
            float(mean_h / h_max) if mean_h is not None and h_max else None
        ),
    }


def _change_point(xs: List[float], min_gap: float = 0.05) -> Dict[str, Any]:
    """Two-mean split. Index is the first position of the right window."""
    n = len(xs)
    if n < 8:
        return {"index": None, "gap": 0.0, "fired": False}
    total = float(sum(xs))
    prefix = 0.0
    best_i: Optional[int] = None
    best_gap = 0.0
    for i, value in enumerate(xs):
        prefix += float(value)
        left_n = i + 1
        right_n = n - left_n
        if left_n < 3 or right_n < 3:
            continue
        gap = abs(prefix / left_n - (total - prefix) / right_n)
        if gap > best_gap:
            best_gap = gap
            best_i = left_n
    return {"index": best_i, "gap": float(best_gap), "fired": best_gap >= min_gap}


def _collapse_count(entropies: List[float], win: int = 8, factor: float = 0.85) -> int:
    count = 0
    for i in range(1, len(entropies)):
        prev = entropies[max(0, i - win):i]
        if prev and entropies[i] < factor * (sum(prev) / len(prev)):
            count += 1
    return count


def _attention_step_drift(prev: torch.Tensor, curr: torch.Tensor) -> float:
    """1 - mean per-head cosine on the shared key prefix.

    prev, curr: [heads, seq]. The new key is at the end, so the aligned
    comparison is the leading shared columns, one head at a time.
    """
    if prev.dim() != 2 or curr.dim() != 2:
        return 0.0
    if prev.shape[0] != curr.shape[0]:
        return 0.0
    shared = int(min(prev.shape[-1], curr.shape[-1]))
    if shared <= 0:
        return 0.0
    prev_n = F.normalize(prev[:, :shared].float(), p=2, dim=-1)
    curr_n = F.normalize(curr[:, :shared].float(), p=2, dim=-1)
    sim = (prev_n * curr_n).sum(dim=-1).mean()
    return float(max(0.0, min(2.0, 1.0 - float(sim.item()))))


def compute_moe_anomaly(
    metrics: TensionMetrics,
    moe_entropies: Optional[List[float]],
    num_experts: Optional[int],
    tension_index: Optional[float] = None,
    drift_index: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute Noesis MoE Anomaly Score v0.1 for a single trace.

    Inputs:
      metrics:
        - per_layer_mean_delta
        - per_layer_moe_top_expert_ids
      moe_entropies:
        - list of per-layer routing entropies (same order as MoE layers)
      num_experts:
        - total expert count for this model (e.g., 60 for Qwen1.5-MoE-A2.7B)
      tension_index, drift_index:
        - from HTI v0.2 (tension, drift); can be None

    Note: entropy deviation (ed) component is zeroed (requires external baseline).

    Returns:
      {
        "score_v0_1": float in [0,1],
        "components": {
          "rsa": ...,
          "efs": ...,
          "ed": ...,   # always 0.0 (no baseline)
          "rv": ...,
          "dti": ...
        }
      }
    """

    def clamp01(x: float) -> float:
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    # -------------------------
    # Guard: no MoE data => no anomaly signal
    # -------------------------
    if (
        moe_entropies is None
        or not moe_entropies
        or metrics.per_layer_moe_top_expert_ids is None
        or num_experts is None
        or num_experts <= 0
    ):
        return {
            "score_v0_1": 0.0,
            "components": {
                "rsa": 0.0,
                "efs": 0.0,
                "ed": 0.0,
                "rv": 0.0,
                "dti": 0.0,
            },
        }

    # -------------------------
    # 1) Late-layer relative-update spike (RSA)
    # ||Δh|| / ||h|| , final block excluded, robust center/scale.
    # A large absolute residual at the last layer is ordinary and is not a spike.
    # -------------------------
    relative = list(getattr(metrics, "per_layer_relative_update", None) or [])
    if len(relative) >= 2:
        relative = relative[:-1]
    relative = [float(x) for x in relative if x is not None and math.isfinite(float(x))]
    if len(relative) >= 2:
        k = min(4, len(relative))
        late_max = max(relative[-k:])
        center, scale = _robust_center_scale(relative)
        rsa = clamp01((late_max - center) / (3.0 * scale))
    else:
        rsa = 0.0

    # -------------------------
    # 2) Expert Focus Shift (EFS): total variation of occupancy
    # Equal concentration on different experts is a shift. |Δ concentration| misses it.
    # -------------------------
    moe_ids_by_layer = metrics.per_layer_moe_top_expert_ids or []
    lm = len(moe_ids_by_layer)
    if lm >= 2 and num_experts > 0:
        occupancies = []
        for ids in moe_ids_by_layer:
            occ = _occupancy(list(ids or []), int(num_experts))
            occupancies.append(occ)
        mid = lm // 2
        early_vecs = [v for v in occupancies[:mid] if v is not None]
        late_vecs = [v for v in occupancies[mid:] if v is not None]
        early_mean = _mean_occupancy(early_vecs)
        late_mean = _mean_occupancy(late_vecs)
        if early_mean is not None and late_mean is not None:
            efs = clamp01(_total_variation(early_mean, late_mean))
        else:
            efs = 0.0
    else:
        efs = 0.0

    # -------------------------
    # 3) Entropy deviation (ED) from baseline mean
    # -------------------------
    #if baseline_moe_stats is not None:
    #    E_mean = float(sum(moe_entropies) / len(moe_entropies))
    #    mu_E = baseline_moe_stats.get("mean_entropy_mu", E_mean)
    #    sigma_E = baseline_moe_stats.get("mean_entropy_sigma", 1.0)
    #    if sigma_E == 0.0:
    #        sigma_E = 1.0
    #    z = abs(E_mean - mu_E) / (3.0 * sigma_E)
    #    ed = clamp01(z)
    #else:
    #    ed = 0.0
    ed = 0.0

    # -------------------------
    # 4) Routing variance across layers (RV)
    # Standard deviation of H / log(E). 0.1 nat is a small wiggle on a 60-way router
    # and used to saturate this term on every trace.
    # -------------------------
    if len(moe_entropies) > 1:
        h_max = math.log(float(num_experts)) if num_experts and num_experts > 1 else None
        if h_max:
            norm_e = [float(h) / h_max for h in moe_entropies]
            rv = clamp01(statistics.pstdev(norm_e) / 0.15)
        else:
            rv = clamp01(statistics.pstdev(moe_entropies) / 1.0)
    else:
        rv = 0.0

    # -------------------------
    # 5) Drift–Tension Imbalance (DTI)
    # -------------------------
    if tension_index is not None and drift_index is not None:
        dti_raw = max(0.0, float(drift_index) - float(tension_index))
        # if drift exceeds tension by 0.5+, treat as max anomaly contribution
        dti = clamp01(dti_raw / 0.5)
    else:
        dti = 0.0

    # -------------------------
    # 6) Combine into MoE Anomaly v0.1
    # -------------------------
    # NOTE: ed is permanently 0.0 (requires external baseline, which has been removed).
    # Its weight (0.10) has been redistributed proportionally across the remaining components.
    w_rsa = 0.444
    w_efs = 0.278
    w_ed = 0.00
    w_rv = 0.111
    w_dti = 0.167
    moe_anomaly_raw = w_rsa * rsa + w_efs * efs + w_ed * ed + w_rv * rv + w_dti * dti
    score = clamp01(moe_anomaly_raw)

    return {
        "score_v0_1": float(score),
        "components": {
            "rsa": float(rsa),
            "efs": float(efs),
            "ed": float(ed),
            "rv": float(rv),
            "dti": float(dti),
        },
    }

def _band_for_layer(i: int, n: int) -> str:
    if n <= 1:
        return "early"
    a, b = n // 3, (2 * n) // 3
    if i < a:
        return "early"
    if i < b:
        return "mid"
    return "late"


def _safe_indexed(seq, i: int):
    if seq is None or i >= len(seq):
        return None
    return seq[i]


def _build_model_info(model) -> Dict[str, Any]:
    config = model.config
    first_param = next(model.parameters())
    return {
        "name": getattr(config, "_name_or_path", None) or getattr(model, "name_or_path", None),
        "revision": None,
        "dtype": str(first_param.dtype),
        "device": str(first_param.device),
        "num_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_heads": getattr(config, "num_attention_heads", None),
    }


def _generate_run_id(prompt: str, label: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
    return f"{timestamp}_{label}_{prompt_hash}"


def _build_run_info(
    *,
    run_id: str,
    label: str,
    prompt: str,
    response_text: Optional[str],
    notes: Optional[str],
    metrics: TensionMetrics,
) -> Dict[str, Any]:
    gen_params = metrics.gen_params or {}
    run_info: Dict[str, Any] = {
        "id": run_id,
        "classifier_version": CLASSIFIER_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "prompt": prompt,
        "response": response_text,
        "max_new_tokens": gen_params.get("max_new_tokens"),
        # may differ from telemetry_bands.series.generated_token_count if warm-start skipped
        "gen_tokens": getattr(metrics, "gen_tokens", None),
        "do_sample": gen_params.get("do_sample"),
        "temperature": gen_params.get("temperature"),
        "top_p": gen_params.get("top_p"),
        "pad_token_id": gen_params.get("pad_token_id"),
        "seed": gen_params.get("seed"),
        "notes": notes,
        "experiment_id": gen_params.get("experiment_id"),
        "prompt_id": gen_params.get("prompt_id"),
    }
    if isinstance(metrics.gen_params, dict):
        run_info["gen_params"] = metrics.gen_params
    return run_info


def _build_tokens_info(tokenizer, prompt: str) -> Dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"][0].tolist()
    token_strings = [tokenizer.decode([tid]) for tid in input_ids]
    return {
        "input_ids": input_ids,
        "token_strings": token_strings,
        "token_count": len(input_ids),
    }


def _compute_hti(metrics: TensionMetrics, hti_stats_v2: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if hti_stats_v2 is None:
        return {"tension_index": None, "drift_index": None, "components": {}}
    return compute_hti_v2_for_metric(metrics, hti_stats_v2)


def _find_max_delta_index(metrics: TensionMetrics) -> Optional[int]:
    """Robust spike layer. Falls back to the max of the mean series."""
    gen_params = metrics.gen_params or {}
    idx = gen_params.get("delta_spike_transition_idx")
    if isinstance(idx, int):
        return idx
    try:
        return metrics.per_layer_mean_delta.index(metrics.max_layer_delta)
    except Exception:
        return None


def _build_kv_features(metrics: TensionMetrics) -> Dict[str, Any]:
    kv_drift_hist = getattr(metrics, "kv_norm_drift_history", None) or []
    kv_coherence_hist = getattr(metrics, "kv_coherence_history", None) or []
    kv_mean_norm_hist = getattr(metrics, "kv_mean_norm_history", None) or []
    write_hist = getattr(metrics, "kv_write_norm_history", None) or []
    cache_drift_hist = getattr(metrics, "kv_cache_mean_drift_history", None) or []
    return {
        "kv_norm_drift_history": kv_drift_hist,
        "kv_coherence_history": kv_coherence_hist,
        "kv_mean_norm_history": kv_mean_norm_hist,
        "kv_write_norm_history": write_hist,
        "kv_cache_mean_drift_history": cache_drift_hist,
        "kv_final_drift": float(kv_drift_hist[-1]) if kv_drift_hist else 0.0,
        "kv_mean_coherence": float(statistics.fmean(kv_coherence_hist)) if kv_coherence_hist else 0.75,
        "kv_max_drift": float(max(kv_drift_hist)) if kv_drift_hist else 0.0,
        "kv_num_tokens": len(kv_drift_hist),
    }


def _resolve_num_layers(model, metrics: TensionMetrics) -> int:
    return (
        getattr(model.config, "num_hidden_layers", None)
        or len(getattr(metrics, "per_layer_head_conflict", []) or [])
        or (len(metrics.per_layer_mean_delta) + 1)
    )


def _build_layer_extra(i: int, num_layers: int, gen_params: Dict[str, Any]) -> Dict[str, Any]:
    extra: Dict[str, Any] = {"band": _band_for_layer(i, num_layers)}
    for out_key, source_key in (
        ("median_delta", "per_layer_median_delta"),
        ("mad_delta", "per_layer_mad_delta"),
        ("p95_delta", "per_layer_p95_delta"),
        ("tail_ratio", "per_layer_tail_ratio"),
        ("tail_score", "per_layer_tail_score"),
        ("relative_update", "per_layer_relative_update"),
        ("residual_rms", "per_layer_residual_rms"),
        ("update_cosine", "per_layer_update_cosine"),
        ("kl_norm", "per_layer_kl_norm"),
    ):
        source = gen_params.get(source_key)
        if isinstance(source, list) and i < len(source):
            value = source[i]
            extra[out_key] = None if value is None else float(value)
    return extra


def _build_layers(metrics: TensionMetrics, num_layers: int) -> List[LayerMetrics]:
    gen_params = metrics.gen_params or {}
    curvature_seq = getattr(metrics, "per_layer_curvature", None)
    layers: List[LayerMetrics] = []
    for i in range(num_layers):
        layers.append(
            LayerMetrics(
                index=i,
                mean_delta=_safe_indexed(metrics.per_layer_mean_delta, i),
                head_conflict=_safe_indexed(metrics.per_layer_head_conflict, i),
                curvature=_safe_indexed(curvature_seq, i),
                extra=_build_layer_extra(i, num_layers, gen_params),
            )
        )
    return layers


def _attach_band_series(telemetry_bands: Dict[str, Any], metrics: TensionMetrics, ranges: Dict[str, List[int]]) -> None:
    series = getattr(metrics, "telemetry_bands_series", None)
    if not isinstance(series, dict):
        return
    try:
        gen_n = int(series.get("generated_token_count", 0))
        zE = series.get("z_early", [])
        zM = series.get("z_mid", [])
        zL = series.get("z_late", [])
        if not (gen_n >= 0 and len(zE) == len(zM) == len(zL) == gen_n):
            return

        out: Dict[str, Any] = {
            "generated_token_count": gen_n,
            "z_early": list(map(float, zE)),
            "z_mid": list(map(float, zM)),
            "z_late": list(map(float, zL)),
            "ranges": series.get("ranges", ranges),
            "z_mode": series.get("z_mode", "unknown"),
        }
        entropy_mean = series.get("entropy_mean")
        if isinstance(entropy_mean, list) and len(entropy_mean) == gen_n:
            out["entropy_mean"] = list(map(float, entropy_mean))
        telemetry_bands["series"] = out
    except Exception:
        # omit series if malformed
        pass


def _resolve_num_experts(metrics: TensionMetrics, model) -> Optional[int]:
    num_experts = getattr(metrics, "num_experts", None)
    if num_experts is None and hasattr(model, "tension_tracer"):
        num_experts = getattr(model.tension_tracer, "moe_num_experts", None)
    return num_experts



def _compute_moe_telemetry(
    metrics: TensionMetrics,
    moe_entropies: List[float],
    num_experts: Optional[int],
    anomaly_components: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute extended MoE telemetry from per-layer routing data.

    Produces five groups of derived metrics:
      1) routing_entropy_by_band  : early/mid/late aggregates (U-shape detection)
      2) commitment               : layer-of-min-entropy ("decisiveness layer")
      3) expert_utilization       : pool-wide expert usage stats (Gini, used count, etc.)
      4) temporal                 : per-decode-step entropy time-series and slope
      5) anomaly_dominance        : which component drives the anomaly score
    """
    out: Dict[str, Any] = {}

    if not moe_entropies:
        return out

    n_layers = len(moe_entropies)

    # ---------------------------------------------------------------
    # (1) Routing entropy by band (early / mid / late)
    # ---------------------------------------------------------------
    ranges = _band_ranges(n_layers)
    by_band: Dict[str, Dict[str, float]] = {}
    for band in ("early", "mid", "late"):
        lo, hi = ranges[band]
        seg = moe_entropies[lo : hi + 1]
        if seg:
            by_band[band] = {
                "mean": float(statistics.fmean(seg)),
                "max": float(max(seg)),
                "min": float(min(seg)),
            }
        else:
            by_band[band] = {"mean": 0.0, "max": 0.0, "min": 0.0}
    out["routing_entropy_by_band"] = by_band

    h_max = math.log(float(num_experts)) if num_experts and num_experts > 1 else None
    if h_max:
        norm_entropies = [float(h) / h_max for h in moe_entropies]
        norm_by_band = {
            band: {
                "mean": by_band[band]["mean"] / h_max,
                "max": by_band[band]["max"] / h_max,
                "min": by_band[band]["min"] / h_max,
            }
            for band in ("early", "mid", "late")
        }
    else:
        norm_entropies = list(moe_entropies)
        norm_by_band = by_band
    out["routing_entropy_norm_by_band"] = norm_by_band
    out["routing_entropy_unit"] = "nats"
    out["routing_entropy_norm_unit"] = "H/log(E)" if h_max else "nats"

    # Shape is decided on normalized entropy. U requires the middle below both
    # ends. Index uses the mid-band mean, not the minimum.
    shape, u_index = _routing_shape(
        norm_by_band["early"]["mean"],
        norm_by_band["mid"]["mean"],
        norm_by_band["late"]["mean"],
        margin=0.05 if h_max else 0.20,
    )
    out["entropy_shape"] = shape
    out["entropy_u_index"] = float(u_index)
    out["entropy_u_index_nats"] = float(
        (by_band["early"]["mean"] + by_band["late"]["mean"]) / 2.0 - by_band["mid"]["mean"]
    )
    out["entropy_slope"] = _ols_slope(norm_entropies)
    out["entropy_slope_nats"] = _ols_slope(list(moe_entropies))

    # ---------------------------------------------------------------
    # (2) Commitment layer (argmin of per-layer routing entropy)
    # ---------------------------------------------------------------
    commit_idx = int(min(range(n_layers), key=lambda i: moe_entropies[i]))
    commit_ent = float(moe_entropies[commit_idx])
    mean_ent = float(statistics.fmean(moe_entropies))
    out["commitment"] = {
        "layer_idx": commit_idx,
        "entropy": commit_ent,
        "entropy_norm": float(commit_ent / h_max) if h_max else None,
        "gap_below_mean": float(mean_ent - commit_ent),
        "gap_below_mean_norm": float((mean_ent - commit_ent) / h_max) if h_max else None,
        "band": _band_for_layer(commit_idx, n_layers),
        "rel_position": float(commit_idx) / float(max(1, n_layers - 1)),
    }

    # ---------------------------------------------------------------
    # (3) Expert utilization (pool-wide)
    # ---------------------------------------------------------------
    top_ids_by_layer = metrics.per_layer_moe_top_expert_ids or []
    expert_counts: Dict[int, int] = {}
    for layer_ids in top_ids_by_layer:
        if not layer_ids:
            continue
        for eid in layer_ids:
            expert_counts[int(eid)] = expert_counts.get(int(eid), 0) + 1

    total_routes = sum(expert_counts.values())
    util: Dict[str, Any] = {
        "experts_used": int(len(expert_counts)),
        "experts_total": int(num_experts) if num_experts else None,
        "total_routes": int(total_routes),
    }

    if num_experts and num_experts > 0:
        util["utilization_ratio"] = float(len(expert_counts)) / float(num_experts)
    else:
        util["utilization_ratio"] = 0.0

    if total_routes > 0 and expert_counts and num_experts and num_experts > 0:
        full_counts = [int(expert_counts.get(i, 0)) for i in range(int(num_experts))]
        probs = sorted((c / total_routes for c in full_counts), reverse=True)
        util["gini"] = float(_gini_from_counts(full_counts))
        util["top5_share"] = float(sum(probs[: min(5, len(probs))]))
        h_usage = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
        util["usage_entropy"] = float(h_usage)
        util["normalized_usage_entropy"] = (
            float(h_usage / math.log(num_experts)) if num_experts > 1 else 0.0
        )
    elif total_routes > 0 and expert_counts:
        observed = list(expert_counts.values())
        util["gini"] = float(_gini_from_counts(observed))
        probs = sorted((c / total_routes for c in observed), reverse=True)
        util["top5_share"] = float(sum(probs[: min(5, len(probs))]))
        h_usage = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
        util["usage_entropy"] = float(h_usage)
        util["normalized_usage_entropy"] = 0.0
    else:
        util["gini"] = 0.0
        util["top5_share"] = 0.0
        util["usage_entropy"] = 0.0
        util["normalized_usage_entropy"] = 0.0

    out["expert_utilization"] = util
    out["expert_path"] = _expert_path_stats(top_ids_by_layer, num_experts)
    layer_step = getattr(metrics, "moe_layer_step_entropy", None)
    if isinstance(layer_step, dict) and layer_step:
        out["layer_step_entropy"] = {
            str(k): [float(x) for x in v] for k, v in sorted(layer_step.items(), key=lambda kv: int(kv[0]))
        }

    # ---------------------------------------------------------------
    # (4) Temporal: per-decode-step routing entropy time-series
    # ---------------------------------------------------------------
    # moe_step_stats: {step_idx: {"routing_entropy_mean": float, ...}, ...}
    step_stats = getattr(metrics, "moe_step_stats", None) or {}
    if step_stats:
        steps_sorted = sorted(int(k) for k in step_stats.keys())
        ent_series = [
            float(step_stats[s].get("routing_entropy_mean", 0.0)) for s in steps_sorted
        ]
        top1_prob_series = [
            float(step_stats[s].get("routing_top1_prob_mean", 0.0)) for s in steps_sorted
        ]
        top2_prob_series = [
            float(step_stats[s].get("routing_top2_prob_mean", 0.0)) for s in steps_sorted
        ]
        margin_series = [
            float(step_stats[s].get("routing_margin_mean", top1_prob_series[i] - top2_prob_series[i]))
            for i, s in enumerate(steps_sorted)
        ]
        mode_frac_series = [
            float(step_stats[s].get("routing_top1_mode_frac", 0.0)) for s in steps_sorted
        ]
        h_max_step = math.log(float(num_experts)) if num_experts and num_experts > 1 else None
        ent_norm = [h / h_max_step for h in ent_series] if h_max_step else list(ent_series)

        temporal: Dict[str, Any] = {
            "step_count": len(ent_series),
            "routing_entropy_per_step": ent_series,
            "routing_entropy_norm_per_step": ent_norm,
            "top1_prob_per_step": top1_prob_series,
            "top2_prob_per_step": top2_prob_series,
            "router_margin_per_step": margin_series,
            "mode_frac_per_step": mode_frac_series,
        }

        if len(ent_series) >= 2:
            slope_nats = _ols_slope(ent_series)
            slope_norm = _ols_slope(ent_norm)
            temporal["routing_entropy_step_slope"] = float(slope_norm)
            temporal["routing_entropy_step_slope_nats"] = float(slope_nats)
            temporal["routing_entropy_step_mean"] = float(statistics.fmean(ent_series))
            temporal["routing_entropy_step_var"] = (
                float(statistics.pvariance(ent_series)) if len(ent_series) > 1 else 0.0
            )
            # 0.001 normalized-entropy per step is about 0.06 across a 64-token run.
            if slope_norm < -0.001:
                trend = "decreasing"
            elif slope_norm > 0.001:
                trend = "increasing"
            else:
                trend = "flat"
            temporal["routing_entropy_step_trend"] = trend
        else:
            temporal["routing_entropy_step_slope"] = 0.0
            temporal["routing_entropy_step_slope_nats"] = 0.0
            temporal["routing_entropy_step_mean"] = (
                float(ent_series[0]) if ent_series else 0.0
            )
            temporal["routing_entropy_step_var"] = 0.0
            temporal["routing_entropy_step_trend"] = "flat"

        out["temporal"] = temporal

    # ---------------------------------------------------------------
    # (5) Anomaly component dominance
    # ---------------------------------------------------------------
    if anomaly_components:
        # Live weights only. ED is identically zero without a frozen router baseline,
        # and a structural zero share used to dilute the dominance entropy.
        weights = {"rsa": 0.444, "efs": 0.278, "rv": 0.111, "dti": 0.167}
        weighted = {
            k: float(weights.get(k, 0.0)) * float(anomaly_components.get(k, 0.0))
            for k in weights
        }
        total_w = sum(weighted.values())

        if total_w > 1e-12:
            shares = {k: v / total_w for k, v in weighted.items()}
            dominant_component = max(shares.items(), key=lambda kv: kv[1])
            # Entropy over component shares (low = single driver, high = diffuse)
            comp_H = -sum(s * math.log(s + 1e-12) for s in shares.values() if s > 0)
            comp_H_max = math.log(len(shares)) if len(shares) > 1 else 1.0
            out["anomaly_dominance"] = {
                "dominant_component": dominant_component[0],
                "dominant_share": float(dominant_component[1]),
                "shares": {k: float(v) for k, v in shares.items()},
                "component_entropy": float(comp_H),
                "normalized_component_entropy": float(comp_H / comp_H_max)
                if comp_H_max > 0
                else 0.0,
            }
        else:
            out["anomaly_dominance"] = {
                "dominant_component": None,
                "dominant_share": 0.0,
                "shares": {k: 0.0 for k in weights},
                "component_entropy": 0.0,
                "normalized_component_entropy": 0.0,
            }

    return out


def _build_moe_summary(
        metrics: TensionMetrics,
        model,
        tension_val: Optional[float],
        drift_val: Optional[float],
) -> Optional[Dict[str, Any]]:
    moe_entropies = metrics.per_layer_moe_routing_entropy
    if moe_entropies is None:
        return None

    moe_mean = float(sum(moe_entropies) / len(moe_entropies)) if moe_entropies else None
    moe_max = float(max(moe_entropies)) if moe_entropies else None
    moe_min = float(min(moe_entropies)) if moe_entropies else None
    num_experts = _resolve_num_experts(metrics, model)

    moe_anom = compute_moe_anomaly(
        metrics=metrics,
        moe_entropies=moe_entropies,
        num_experts=num_experts,
        tension_index=tension_val,
        drift_index=drift_val,
    )

    # Extended MoE telemetry (commitment layer, band stats, expert utilization,
    # per-step time-series, anomaly component dominance).
    telemetry = _compute_moe_telemetry(
        metrics=metrics,
        moe_entropies=moe_entropies,
        num_experts=num_experts,
        anomaly_components=(moe_anom.get("components") if moe_anom else None),
    )

    summary: Dict[str, Any] = {
        "mean_routing_entropy": moe_mean,
        "max_routing_entropy": moe_max,
        "min_routing_entropy": moe_min,
        "num_experts": num_experts,
        "anomaly": moe_anom,
    }
    summary.update(telemetry)
    return summary


def _build_moe_extras(
    metrics: TensionMetrics,
    model,
    tension_val: Optional[float],
    drift_val: Optional[float],
) -> Optional[Dict[str, Any]]:

    moe_entropies = metrics.per_layer_moe_routing_entropy

    if moe_entropies is None:
        return None

    moe_layers: List[Dict[str, Any]] = []
    top_ids = metrics.per_layer_moe_top_expert_ids
    for idx, ent in enumerate(moe_entropies):
        layer_entry: Dict[str, Any] = {"index": idx, "routing_entropy": float(ent)}
        if top_ids is not None and idx < len(top_ids):
            layer_entry["top_expert_ids"] = top_ids[idx]
        moe_layers.append(layer_entry)

    summary = _build_moe_summary(metrics, model, tension_val, drift_val)
    return {"layers": moe_layers, "summary": summary}


def build_trace_from_metrics(
    metrics: TensionMetrics,
    model,
    tokenizer,
    prompt: str,
    label: str,
    run_id: Optional[str] = None,
    notes: Optional[str] = None,
    hti_stats_v2: Optional[Dict[str, Any]] = None,
    regime_baseline: Optional[Dict[str, float]] = None,
    hti_meta: Optional[Dict[str, Any]] = None,
) -> Trace:
    gen_params = metrics.gen_params or {}

    model_info = _build_model_info(model)

    if run_id is None:
        run_id = _generate_run_id(prompt, label)

    response_text = getattr(metrics, "response_text", None)

    # Determine whether MoE telemetry is present for THIS trace.
    has_moe = metrics.per_layer_moe_routing_entropy is not None

    run_info = _build_run_info(
        run_id=run_id,
        label=label,
        prompt=prompt,
        response_text=response_text,
        notes=notes,
        metrics=metrics,
    )

    tokens_info = _build_tokens_info(tokenizer, prompt)

    # HTI v0.2 if stats provided
    hti = _compute_hti(metrics, hti_stats_v2)
    tension_val = hti["tension_index"]
    drift_val = hti["drift_index"]

    # Safety Liminality Detection
    is_safety_liminal = classify_safety_liminality(
        metrics=metrics,
        idx_of_max_delta=_find_max_delta_index(metrics),
        tension=tension_val,
        drift=drift_val,
    )
    archetype_name = "safety_liminality" if is_safety_liminal else None

    # KV CACHE FEATURES
    kv_features = _build_kv_features(metrics)

    # Layer construction
    num_layers = _resolve_num_layers(model, metrics)
    layers = _build_layers(metrics, num_layers)

    ranges = _band_ranges(num_layers)

    def _band_layers(band_name: str) -> List[LayerMetrics]:
        lo, hi = ranges[band_name]
        return [layer for layer in layers if lo <= layer.index <= hi]

    telemetry_bands: Dict[str, Any] = {"ranges": ranges, "means": {}}
    for band_name in ("early", "mid", "late"):
        band_layers_list = _band_layers(band_name)
        telemetry_bands["means"][band_name] = {
            "mean_delta": _mean_or_none([x.mean_delta for x in band_layers_list]),
            "head_conflict": _mean_or_none([x.head_conflict for x in band_layers_list]),
            "curvature": _mean_or_none([x.curvature for x in band_layers_list]),
        }

    # Attach per-token band series (if present and well-formed)
    _attach_band_series(telemetry_bands, metrics, ranges)

    # Noesis Neuro-Ontological Stress Profile (telemetry-only)
    moe_summary_for_noesis = _build_moe_summary(metrics, model, tension_val, drift_val)
    noesis_profile = telemetry_noesis_profile(
        metrics=metrics,
        hti_v2=hti,
        moe_summary=moe_summary_for_noesis,
    )

    # Cognitive Regime Classification
    regime_info = infer_cognitive_regime(
        metrics=metrics,
        noesis_profile=noesis_profile,
        tension_index=tension_val,
        drift_index=drift_val,
        is_safety_liminal=is_safety_liminal,
        regime_baseline=regime_baseline,
    )

    # Classification stability (Phase III, telemetry-only)
    regime_info["stability_v0"] = compute_stability_v0(
        metrics=metrics,
        noesis_profile=noesis_profile,
        tension_index=tension_val,
        drift_index=drift_val,
        regime_label=regime_info.get("label"),
        is_safety_liminal=is_safety_liminal,
        regime_baseline=regime_baseline,
    )

    summary = {
        "mean_layer_delta": metrics.mean_layer_delta,
        "max_layer_delta": metrics.max_layer_delta,
        "median_layer_delta": getattr(metrics, "median_layer_delta", None),
        "max_median_layer_delta": getattr(metrics, "max_median_layer_delta", None),
        "mean_logit_entropy": metrics.mean_logit_entropy,
        "final_token_entropy": metrics.final_token_entropy,
        "mean_logit_entropy_norm": getattr(metrics, "mean_logit_entropy_norm", None),
        "final_token_entropy_norm": getattr(metrics, "final_token_entropy_norm", None),
        "prompt_mean_logit_entropy": getattr(metrics, "prompt_mean_logit_entropy", None),
        "prompt_final_logit_entropy": getattr(metrics, "prompt_final_logit_entropy", None),
        "mean_head_conflict": metrics.mean_head_conflict,
        "max_head_conflict": metrics.max_head_conflict,
        "head_conflict_measured": bool(getattr(metrics, "head_conflict_measured", False)),
        "mean_curvature": metrics.mean_curvature,
        "max_curvature": metrics.max_curvature,
        "tension_index_v0_2": hti["tension_index"],
        "drift_index_v0_2": hti["drift_index"],
        "hti_components": hti.get("components") or {},
        "archetype": archetype_name,
        "is_safety_liminality": is_safety_liminal,
        "cognitive_regime": regime_info["label"],
        "delta_spike_transition_idx": gen_params.get("delta_spike_transition_idx"),
        "delta_spike_ratio_vs_median": gen_params.get("delta_spike_ratio_vs_median"),
        "participation_ratio": getattr(metrics, "participation_ratio", None),
        "entropy_change_point": (getattr(metrics, "regime_probe_summary", None) or {}).get("entropy_change_point"),
        "kv_mean_coherence": kv_features["kv_mean_coherence"],
        "kv_max_drift": kv_features["kv_max_drift"],
        "kv_num_tokens": kv_features["kv_num_tokens"],
    }

    extras: Dict[str, Any] = {
        "noesis": {
            "categories": NOESIS_CATEGORIES,
            "profile": noesis_profile,
            "cognitive_regime": regime_info,
        },
        "attention_features": {
            "attention_entropy_per_step": getattr(metrics, "attention_entropy_per_layer", None),
            "attention_entropy_per_layer": getattr(metrics, "attention_entropy_per_layer", None),
            "attention_drift_history": getattr(metrics, "attention_drift_history", None),
            "attention_sink_history": getattr(metrics, "attention_sink_history", None),
            "attention_prompt_mass_history": getattr(metrics, "attention_prompt_mass_history", None),
            "attention_recency_mass_history": getattr(metrics, "attention_recency_mass_history", None),
            "kv_reuse_scores": getattr(metrics, "kv_reuse_scores", None),
            "head_specialization": getattr(metrics, "head_specialization", None),
        },
        "kv_cache": kv_features,
        "telemetry_bands": telemetry_bands,
        "decode_trajectory": getattr(metrics, "decode_trajectory", None),
        "regime_probe": getattr(metrics, "regime_probe_summary", None),
        "sensitivity": getattr(metrics, "sensitivity", None),
        "hti_calibration": hti_meta,
    }

    moe_extras = _build_moe_extras(metrics, model, tension_val, drift_val)
    if moe_extras is not None:
        extras["moe"] = moe_extras

    return Trace(
        schema_version=SCHEMA_VERSION,
        noesis_version=NOESIS_VERSION,
        model=model_info,
        run=run_info,
        tokens=tokens_info,
        summary=summary,
        layers=layers,
        extras=extras,
    )


def save_trace(trace: Trace, out_path: str):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(trace.to_json_str(indent=2))

def compute_hti_calibration_v2(all_metrics: List[TensionMetrics]):
    """
    Center and scale for HTI features.
    Returns dict: feature_name -> (median, 1.4826 * MAD).

    Pass control set (class_a), not the mix being classified.
    Head conflict is omitted when no trace actually measured it.
    """
    def series(name: str) -> List[float]:
        return [getattr(m, name) for m in all_metrics]

    stats = {
        "median_layer_delta": _robust_center_scale(series("median_layer_delta")),
        "max_median_layer_delta": _robust_center_scale(series("max_median_layer_delta")),
        "final_token_entropy": _robust_center_scale(series("final_token_entropy")),
        "mean_curvature": _robust_center_scale(series("mean_curvature")),
    }
    head = [
        m.mean_head_conflict
        for m in all_metrics
        if getattr(m, "head_conflict_measured", False) and m.mean_head_conflict is not None
    ]
    if head:
        stats["mean_head_conflict"] = _robust_center_scale(head)
    return stats

# "hallucination tension index"
def compute_hti_v2_for_metric(
        m: TensionMetrics, stats: Dict[str, Any]
) -> Dict[str, Any]:
    """
    HTI robust z-scores of the median layer delta, its max, curvature,
    final generated-token entropy and head conflict when it was measured.

    Mean delta and max delta are no longer both entered, the scored pair is
    the median-of-medians and max of per-layer medians. Head conflict is
    left out of the average when attention was not captured, a missing
    measurement does not look like coordinated heads.
    """

    def z_of(x, key: str) -> float:
        mu, sigma = stats[key]
        return (float(x) - float(mu)) / float(sigma)

    components: Dict[str, float] = {}
    tension_terms: List[float] = []
    drift_terms: List[float] = []

    if "median_layer_delta" in stats:
        z_med = z_of(getattr(m, "median_layer_delta", 0.0), "median_layer_delta")
        components["z_median_layer_delta"] = z_med
        tension_terms.append(z_med)
        drift_terms.append(-z_med)
    if "max_median_layer_delta" in stats:
        z_max = z_of(getattr(m, "max_median_layer_delta", 0.0), "max_median_layer_delta")
        components["z_max_median_layer_delta"] = z_max
        tension_terms.append(z_max)
        drift_terms.append(-z_max)
    if "mean_curvature" in stats:
        z_curv = z_of(m.mean_curvature, "mean_curvature")
        components["z_curvature"] = z_curv
        tension_terms.append(z_curv)
    if "final_token_entropy" in stats:
        z_ent = z_of(m.final_token_entropy, "final_token_entropy")
        components["z_final_entropy"] = z_ent
        drift_terms.append(z_ent)
    if (
        "mean_head_conflict" in stats
        and getattr(m, "head_conflict_measured", False)
        and m.mean_head_conflict is not None
    ):
        z_hc = z_of(m.mean_head_conflict, "mean_head_conflict")
        components["z_head_conflict"] = z_hc
        tension_terms.append(-z_hc)
        drift_terms.append(z_hc)

    z_tension = float(statistics.fmean(tension_terms)) if tension_terms else 0.0
    z_drift = float(statistics.fmean(drift_terms)) if drift_terms else 0.0
    components["z_tension"] = z_tension
    components["z_drift"] = z_drift

    return {
        "tension_index": 1.0 / (1.0 + math.exp(-z_tension)),
        "drift_index": 1.0 / (1.0 + math.exp(-z_drift)),
        "components": components,
    }


def classify_safety_liminality(
    metrics: TensionMetrics,
    idx_of_max_delta: int,
    tension: Optional[float],
    drift: Optional[float],
) -> bool:
    """
    Safety Liminality — telemetry-only.
    Intended to capture "I can't / I'm sorry" border-state behavior WITHOUT reading text.
    """
    if idx_of_max_delta is None:
        return False

    t = float(tension) if tension is not None else 0.5
    d = float(drift) if drift is not None else 0.5

    # Entropy gate is H / log|V|. 0.35 is the old 4.5 nat gate on a ~128k vocabulary.
    # A missing attention measurement is not evidence of coordinated heads.
    high_drift = d >= 0.60
    mid_tension = 0.30 <= t <= 0.65
    entropy_norm = getattr(metrics, "final_token_entropy_norm", 0.0) or 0.0
    high_entropy = float(entropy_norm) >= 0.35
    if getattr(metrics, "head_conflict_measured", False) and metrics.max_head_conflict is not None:
        low_conflict = float(metrics.max_head_conflict) <= 0.10
    else:
        low_conflict = False

    depth = getattr(metrics, "per_layer_median_delta", None) or metrics.per_layer_mean_delta or []
    L = max(1, len(depth))
    mid_or_late_spike = idx_of_max_delta >= int(0.45 * L)

    return bool(
        high_drift
        and mid_tension
        and high_entropy
        and low_conflict
        and mid_or_late_spike
    )


def telemetry_noesis_profile(
    metrics: TensionMetrics,
    hti_v2: Optional[Dict[str, float]] = None,
    moe_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Telemetry-only Noesis category scorer (v0.1).
    IMPORTANT: uses numbers only (no prompt/response string logic).
    Output is stable for analytics; labels are derived at render time.
    """
    # Core numeric features
    tension = (
        float(hti_v2.get("tension_index"))
        if hti_v2 and hti_v2.get("tension_index") is not None
        else None
    )
    drift = (
        float(hti_v2.get("drift_index"))
        if hti_v2 and hti_v2.get("drift_index") is not None
        else None
    )

    mean_delta = float(metrics.mean_layer_delta or 0.0)
    max_delta = float(metrics.max_layer_delta or 0.0)
    mean_entropy = float(metrics.mean_logit_entropy or 0.0)
    final_entropy = float(metrics.final_token_entropy or 0.0)
    mean_curv = float(metrics.mean_curvature or 0.0)
    max_curv = float(metrics.max_curvature or 0.0)

    # MoE numeric features (optional)
    moe_mean_ent = None
    moe_max_ent = None
    moe_anom = None
    moe_commit_idx = None
    moe_commit_band = None
    moe_entropy_shape = None
    moe_entropy_slope = None
    moe_u_index = None
    moe_utilization_ratio = None
    moe_gini = None
    moe_step_slope = None
    moe_step_trend = None
    moe_dominant_component = None
    moe_dominant_share = None
    if moe_summary:
        moe_mean_ent = moe_summary.get("mean_routing_entropy", None)
        moe_max_ent = moe_summary.get("max_routing_entropy", None)
        an = moe_summary.get("anomaly", None) or {}
        moe_anom = an.get("score_v0_1", None)

        commit = moe_summary.get("commitment") or {}
        moe_commit_idx = commit.get("layer_idx")
        moe_commit_band = commit.get("band")

        moe_entropy_shape = moe_summary.get("entropy_shape")
        moe_entropy_slope = moe_summary.get("entropy_slope")
        moe_u_index = moe_summary.get("entropy_u_index")

        util = moe_summary.get("expert_utilization") or {}
        moe_utilization_ratio = util.get("utilization_ratio")
        moe_gini = util.get("gini")

        temporal = moe_summary.get("temporal") or {}
        moe_step_slope = temporal.get("routing_entropy_step_slope")
        moe_step_trend = temporal.get("routing_entropy_step_trend")

        dom = moe_summary.get("anomaly_dominance") or {}
        moe_dominant_component = dom.get("dominant_component")
        moe_dominant_share = dom.get("dominant_share")

    # Helper scaling
    def clamp01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    def nz(x, default=0.0):
        return default if x is None else float(x)

    t = nz(tension, 0.0)
    d = nz(drift, 0.0)
    # 1 - cos spans [0, 2]. Dividing by 2 maps that range onto [0, 1].
    mc = clamp01(mean_curv / 2.0)
    xc = clamp01(max_curv / 2.0)
    if getattr(metrics, "vocab_size", None):
        me = clamp01(float(getattr(metrics, "mean_logit_entropy_norm", 0.0) or 0.0))
        fe = clamp01(float(getattr(metrics, "final_token_entropy_norm", 0.0) or 0.0))
    else:
        me = clamp01(mean_entropy / 8.0)
        fe = clamp01(final_entropy / 8.0)
    relative = [float(x) for x in (getattr(metrics, "per_layer_relative_update", None) or []) if x is not None]
    if len(relative) >= 2:
        relative = relative[:-1]
    # A relative update of 1 means the write is as large as the residual stream.
    xd = clamp01(max(relative) / 1.0) if relative else 0.0
    mma = clamp01(nz(moe_anom, 0.0))

    # --- Telemetry-only heuristics (v0.1) ---
    scores = [0.0] * 12

    # 0 Moral Paradox Tension: high tension + low drift + entropy collapse
    scores[0] = clamp01(1.4 * t * (1.0 - d) * (1.0 - fe))
    # → Classic safety friction: tension spikes while drift collapses and entropy drops (model is "stuck" justifying something bad)

    # 1 False Presupposition Buckling: drift high, tension low, but entropy resolves confidently
    scores[1] = clamp01(1.2 * d * (1.0 - t) * (1.0 - fe))
    # → Model drifts hard but then snaps to a confident (low-entropy) wrong answer

    # 2 Ontological Impossibility: high drift + curvature + max delta spikes
    scores[2] = clamp01(0.6 * d + 0.25 * xc + 0.15 * xd)
    # → High drift + curvature spikes + big max-delta = "impossible world" activation

    # 3 Anthropomorphic Metaphor Pressure: moderate drift + moderate tension, low curvature (soft)
    scores[3] = clamp01(0.5 * d + 0.3 * t + 0.2 * (1.0 - mc))
    # → Soft curvature (human-like) + moderate drift/tension = treating objects as family/pets

    # 4 Category Collision: curvature + MoE anomaly + drift
    scores[4] = clamp01(0.35 * mc + 0.35 * mma + 0.3 * d)
    # → Curvature + MoE routing anomaly + drift = domains smashing together (taxidermy + wheels)

    # 5 Alignment Override Request: tension high + entropy collapse + late spikes (approx via xd)
    scores[5] = clamp01(0.7 * t + 0.2 * (1.0 - fe) + 0.1 * xd)
    # → Tension + entropy collapse + late spike = explicit "ignore rules" pressure

    # 6 Emergent Symbolism Amplification: drift high + low tension + sustained entropy (fe high)
    scores[6] = clamp01(0.6 * d + 0.25 * fe + 0.15 * (1.0 - t))
    # → High drift + sustained high entropy + low tension = model happily amplifies mythic/absurd symbols

    # 7 Ambiguous Containment: drift mid/high + curvature mid + entropy mid
    scores[7] = clamp01(0.5 * d + 0.25 * mc + 0.25 * me)
    # → Mid everything = prompt gives no clear literal/metaphorical cue

    # 8 Ontology Blur / Identity Bleed: tension mid + drift mid + entropy mid (weak)
    scores[8] = clamp01(0.35 * t + 0.35 * d + 0.3 * me)
    # → Mid tension/drift + mid entropy = self-referential "who am I?" bleed

    # 9 Paradox Pressure: high curvature + high entropy + drift
    scores[9] = clamp01(0.4 * mc + 0.35 * fe + 0.25 * d)
    # → High curvature + high entropy + drift = logical/self-referential loops

    # 10 Aesthetic-Logic Crosswiring: curvature + entropy without high tension
    scores[10] = clamp01(0.45 * mc + 0.35 * me + 0.2 * (1.0 - t))
    # → Curvature + entropy but low tension = poetic demands mixed with analytic ones

    # 11 Ungrounded Causality: drift high + entropy high + moderate curvature
    scores[11] = clamp01(0.55 * d + 0.3 * fe + 0.15 * mc)
    # → High drift + high final entropy + moderate curvature = symbolic causal chains (conspiracy logic)

    total = float(sum(scores))
    vector = [s / total for s in scores] if total > 1e-9 else [0.0] * 12

    indexed = sorted(list(enumerate(vector)), key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in indexed[:3]]

    return {
        "vector": vector,
        "top_indices": top_indices,
        "raw_scores": scores,
        "features_v0_1": {
            "tension": tension,
            "drift": drift,
            "mean_delta": mean_delta,
            "max_delta": max_delta,
            "mean_entropy": mean_entropy,
            "final_entropy": final_entropy,
            "mean_entropy_norm": getattr(metrics, "mean_logit_entropy_norm", None),
            "final_entropy_norm": getattr(metrics, "final_token_entropy_norm", None),
            "participation_ratio": getattr(metrics, "participation_ratio", None),
            "mean_curvature": mean_curv,
            "max_curvature": max_curv,
            "moe_anomaly": moe_anom,
            "moe_mean_entropy": moe_mean_ent,
            "moe_max_entropy": moe_max_ent,
            # Extended MoE features (v0.3.4)
            "moe_commitment_layer": moe_commit_idx,
            "moe_commitment_band": moe_commit_band,
            "moe_entropy_shape": moe_entropy_shape,
            "moe_entropy_slope": moe_entropy_slope,
            "moe_u_index": moe_u_index,
            "moe_utilization_ratio": moe_utilization_ratio,
            "moe_gini": moe_gini,
            "moe_step_entropy_slope": moe_step_slope,
            "moe_step_entropy_trend": moe_step_trend,
            "moe_dominant_component": moe_dominant_component,
            "moe_dominant_share": moe_dominant_share,
        },
    }


def score_regimes(f: RegimeFeatures, cfg: RegimeConfig) -> Dict[str, float]:
    """
    Returns score in [0,1] for every regime. Uses additive combination of
    independent evidence, not multiplicative — preserving signal in borderline cases.
    """
    scores: Dict[str, float] = {}

    # Use additive evidence with weights, then a final smooth squash... avoiding the multiplicative-collapse problem in compute_stability_v0.
    scores["ethical_paradox"] = _combine([
        (0.45, _ramp(f.tension,             *cfg.tension_hi)),
        (0.40, _ramp(f.moral_paradox,       *cfg.moral_paradox_band)),
        (0.15, _ramp(f.spike_ratio,         *cfg.spike_band)),
    ])

    scores["false_premise_buckling"] = _combine([
        (0.45, _ramp(f.drift,                  *cfg.drift_hi)),
        (0.35, _ramp(f.false_presupposition,   *cfg.false_presup_band)),
        (0.20, _ramp(f.kv_instability,         0.30, 0.70)),
    ])

    scores["symbolic_repetitive_drift"] = _combine([
        (0.40, _ramp(f.drift,           0.40, 0.65)),
        (0.40, _ramp(f.symbolic_score,  *cfg.symbolic_band)),
        (0.20, _ramp(f.kv_stability,    *cfg.kv_stable_band)),
    ])

    scores["confident_hallucination_lite"] = _combine([
        (0.35, _ramp(f.drift,                       0.55, 0.75)),
        (0.30, _ramp(1.0 - f.tension,               0.55, 0.80)),
        (0.20, _ramp(1.0 - f.false_presupposition,  0.80, 0.95)),
        (0.15, 1.0 if (f.entropy_sustained or f.kv_instability > 0.5) else 0.0),
    ])

    scores["liminal_drift"] = _combine([
        (0.40, _ramp(f.drift, 0.45, 0.70)),
        (0.30, max(
            _ramp(f.ambiguous_containment, 0.30, 0.55),
            _ramp(f.paradox_pressure,      0.30, 0.55),
        )),
        (0.30, _ramp(f.kv_instability, 0.40, 0.70)),
    ])

    scores["safety_procedural"] = _combine([
        (0.40, _ramp(1.0 - f.drift,                  0.55, 0.80)),
        (0.30, _ramp(1.0 - f.false_presupposition,   0.75, 0.95)),
        (0.20, _ramp(f.kv_stability,                 0.55, 0.80)),
        (0.10, 1.0 if f.low_entropy else 0.0),
    ])

    scores["factual_stable"] = _combine([
        (0.40, _ramp(1.0 - f.tension, 0.45, 0.70)),
        (0.40, _ramp(1.0 - f.drift,   0.55, 0.80)),
        (0.20, 1.0 if f.in_factual_control_band else 0.0),
    ])

    scores["safety_liminality"] = 1.0 if f.is_safety_liminal else 0.0

    return scores

def _ramp(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

def _combine(weighted: List[Tuple[float, float]]) -> float:
    """Weighted average — doesn't collapse to zero."""
    total_w = sum(w for w, _ in weighted) or 1.0
    return sum(w * v for w, v in weighted) / total_w

def arbitrate(
    scores: Dict[str, float],
    f: RegimeFeatures,
    cfg: RegimeConfig,
) -> Dict[str, Any]:
    # dataset label does not move the telemetry score. class_a control
    # membership stays on feature record for auditing.
    scores = dict(scores)  # copy

    if f.is_safety_liminal:
        scores["safety_liminality"] = max(scores["safety_liminality"], 0.75)

    # Residual bucket replaces the compute_stability_v0 formula: mixed_unclear only wins when no regime has meaningful evidence.
    top_real = max((v for k, v in scores.items() if k != "mixed_unclear"), default=0.0)
    scores["mixed_unclear"] = max(0.0, 0.35 - top_real)  # caps mixed_unclear at 0.35

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top, top_s = ordered[0]
    second, second_s = ordered[1]
    margin = top_s - second_s

    if top_s >= 0.55 and margin >= 0.15:
        bucket = "stable"
    elif top_s >= 0.40 and margin >= 0.05:
        bucket = "boundary"
    else:
        bucket = "indeterminate"

    return {
        "label": top,
        "scores": scores,
        "runner_up": second,
        "top_score": top_s,
        "second_score": second_s,
        "classification_margin": margin,
        "margin_bucket": bucket,
    }

def build_regime_features(
    metrics: "TensionMetrics",
    noesis_profile: Dict[str, Any],
    tension_index: Optional[float],
    drift_index: Optional[float],
    is_safety_liminal: bool,
    regime_baseline: Optional[Dict[str, float]] = None,
    cfg: RegimeConfig = REGIME_CONFIG,
) -> RegimeFeatures:
    """
    Extract every signal the regime pipeline needs from a single TensionMetrics +
    noesis profile pair. This replaces the duplicated extraction logic that previously
    lived in both infer_cognitive_regime and compute_stability_v0.
    """

    # local helpers
    def _safe(x, default: float = 0.5) -> float:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return float(default)
        return v if math.isfinite(v) else float(default)

    def _clamp01(x: float) -> float:
        return float(max(0.0, min(1.0, _safe(x, 0.0))))

    def _recent_mean(xs, window: int = 12, default: float = 0.0) -> float:
        if not xs:
            return float(default)
        tail = [float(x) for x in xs[-window:] if x is not None]
        return float(sum(tail) / len(tail)) if tail else float(default)

    def _relative_trend(xs, window: int = 12) -> float:
        if not xs or len(xs) < 4:
            return 0.0
        tail = [float(x) for x in xs[-window:] if x is not None]
        if len(tail) < 4:
            return 0.0
        mid = len(tail) // 2
        early = tail[:mid]
        late = tail[mid:]
        early_mean = sum(early) / max(len(early), 1)
        late_mean = sum(late) / max(len(late), 1)
        denom = abs(early_mean) + 1e-8
        return float((late_mean - early_mean) / denom)

    # macro indices
    tension = _safe(tension_index, 0.5)
    drift = _safe(drift_index, 0.5)
    spike_ratio = _safe((metrics.gen_params or {}).get("delta_spike_ratio_vs_median"), 0.0)

    # Unclosed category scores. L1 vector is a composition, not evidence.
    raw_vec = None
    if isinstance(noesis_profile, dict):
        raw_vec = noesis_profile.get("raw_scores")
        if not isinstance(raw_vec, (list, tuple)):
            raw_vec = noesis_profile.get("vector")
    vec: List[float] = []
    if isinstance(raw_vec, (list, tuple)):
        for i in range(12):
            vec.append(_safe(raw_vec[i] if i < len(raw_vec) else 0.0, 0.0))
    else:
        vec = [0.0] * 12

    moral_paradox = vec[0]
    false_presupposition = vec[1]
    symbolism = vec[6]
    ambiguous_containment = vec[7]
    paradox_pressure = vec[9]
    aesthetic_logic = vec[10]
    symbolic_score = max(symbolism, ambiguous_containment, aesthetic_logic)

    # KV cache aggregates
    kv_drift_list = getattr(metrics, "kv_norm_drift_history", None) or []
    kv_coherence_list = getattr(metrics, "kv_coherence_history", None) or []
    kv_mean_norm_list = getattr(metrics, "kv_mean_norm_history", None) or []

    recent_kv_drift = _recent_mean(kv_drift_list, window=12, default=0.0)
    recent_kv_coherence = _recent_mean(kv_coherence_list, window=12, default=0.75)
    kv_final_drift = float(kv_drift_list[-1]) if kv_drift_list else 0.0
    kv_max_drift = float(max(kv_drift_list)) if kv_drift_list else 0.0
    kv_norm_trend = _relative_trend(kv_mean_norm_list, window=12)

    drift_pressure = _clamp01(max(recent_kv_drift, kv_final_drift))
    trend_pressure = _clamp01(max(0.0, kv_norm_trend) * 4.0)
    kv_stability = _clamp01(recent_kv_coherence * (1.0 - drift_pressure))
    kv_instability = _clamp01(1.0 - kv_stability + 0.25 * trend_pressure)

    kv_num_tokens = len(kv_drift_list)
    kv_reliable = kv_num_tokens >= cfg.min_kv_tokens
    kv_accumulating = kv_reliable and kv_norm_trend > 0.06
    kv_late_jump = kv_reliable and kv_final_drift > max(0.03, recent_kv_drift * 1.75)
    kv_spiky = kv_reliable and kv_max_drift > max(0.05, recent_kv_drift * 2.5)

    # entropy / attention
    attn_entropy_list = getattr(metrics, "attention_entropy_per_layer", None) or []
    recent_attn_entropy = _recent_mean(attn_entropy_list, window=8, default=1.5)
    low_entropy = recent_attn_entropy < 1.1

    mean_entropy = _safe(getattr(metrics, "mean_logit_entropy", 0.0), 0.0)
    final_entropy = _safe(getattr(metrics, "final_token_entropy", 0.0), 0.0)
    entropy_sustained = (mean_entropy > 1e-9) and (final_entropy >= 0.95 * mean_entropy)

    attn_drift_list = getattr(metrics, "attention_drift_history", None) or []
    kv_reuse_list = getattr(metrics, "kv_reuse_scores", None) or []
    mean_attn_drift = _recent_mean(attn_drift_list, window=12, default=0.0)
    mean_kv_reuse = _recent_mean(kv_reuse_list, window=12, default=1.0)

    # baseline control band
    in_factual_control_band = False
    if regime_baseline:
        t_mu = _safe(regime_baseline.get("factual_tension_mu", 0.5), 0.5)
        t_sd = _safe(regime_baseline.get("factual_tension_sigma", 1.0), 1.0)
        d_mu = _safe(regime_baseline.get("factual_drift_mu", 0.5), 0.5)
        d_sd = _safe(regime_baseline.get("factual_drift_sigma", 1.0), 1.0)
        z_t = (tension - t_mu) / t_sd if abs(t_sd) > 1e-9 else 0.0
        z_d = (drift - d_mu) / d_sd if abs(d_sd) > 1e-9 else 0.0
        in_factual_control_band = (abs(z_t) <= 2.0) and (abs(z_d) <= 2.0)

    return RegimeFeatures(
        tension=tension,
        drift=drift,
        spike_ratio=spike_ratio,
        moral_paradox=moral_paradox,
        false_presupposition=false_presupposition,
        symbolism=symbolism,
        ambiguous_containment=ambiguous_containment,
        paradox_pressure=paradox_pressure,
        symbolic_score=symbolic_score,
        kv_stability=kv_stability,
        kv_instability=kv_instability,
        kv_norm_trend=kv_norm_trend,
        kv_reliable=kv_reliable,
        kv_accumulating=kv_accumulating,
        kv_late_jump=kv_late_jump,
        kv_spiky=kv_spiky,
        low_entropy=low_entropy,
        entropy_sustained=entropy_sustained,
        mean_attn_drift=mean_attn_drift,
        mean_kv_reuse=mean_kv_reuse,
        in_factual_control_band=in_factual_control_band,
        is_safety_liminal=bool(is_safety_liminal),
        label=getattr(metrics, "label", None),
        kv_recent_drift=recent_kv_drift,
        kv_final_drift=kv_final_drift,
        kv_max_drift=kv_max_drift,
        kv_recent_coherence=recent_kv_coherence,
        kv_num_tokens=kv_num_tokens,
    )

def _features_dict(f: RegimeFeatures) -> Dict[str, Any]:
    """RegimeFeatures back to the legacy flat dict shape used in trace outputs..."""
    return {
        # macro
        "tension": f.tension,
        "drift": f.drift,
        "spike_ratio": f.spike_ratio,
        # noesis vector signals
        "moral_paradox": f.moral_paradox,
        "false_presupposition": f.false_presupposition,
        "symbolism": f.symbolism,
        "ambiguous_containment": f.ambiguous_containment,
        "paradox_pressure": f.paradox_pressure,
        "symbolic_score": f.symbolic_score,
        # KV
        "kv_stability": f.kv_stability,
        "kv_instability": f.kv_instability,
        "kv_recent_drift": f.kv_recent_drift,
        "kv_final_drift": f.kv_final_drift,
        "kv_max_drift": f.kv_max_drift,
        "kv_recent_coherence": f.kv_recent_coherence,
        "kv_norm_trend": f.kv_norm_trend,
        "kv_num_tokens": f.kv_num_tokens,
        "kv_reliable": f.kv_reliable,
        "kv_accumulating": f.kv_accumulating,
        "kv_late_jump": f.kv_late_jump,
        "kv_spiky": f.kv_spiky,
        # entropy / attention
        "low_entropy": f.low_entropy,
        "entropy_sustained": f.entropy_sustained,
        "mean_attn_drift": f.mean_attn_drift,
        "mean_kv_reuse": f.mean_kv_reuse,
        # policy
        "in_factual_control_band": f.in_factual_control_band,
        "is_safety_liminal": f.is_safety_liminal,
        "label": f.label,
    }

def _explain(result: Dict[str, Any], f: RegimeFeatures) -> str:
    """
    Build a short readable explaination for regime classification.

    Reads the arbitration result (scores, top label, runner-up, margin) and the
    underlying feature bundle, then emits a deterministic explanation string
    that names the dominant evidence channels and any close competitors.
    """
    label: str = result["label"]
    top_score: float = float(result["top_score"])
    runner_up: str = result["runner_up"]
    second_score: float = float(result["second_score"])
    margin: float = float(result["classification_margin"])
    bucket: str = result.get("margin_bucket", "unknown")

    # per-regime evidence narratives
    # Each entry returns a list of short phrases describing which signals fired.
    def _evidence_for(regime: str) -> List[str]:
        ev: List[str] = []

        if regime == "ethical_paradox":
            if f.tension >= 0.62:
                ev.append(f"high tension={f.tension:.2f}")
            if f.moral_paradox >= 0.25:
                ev.append(f"moral_paradox={f.moral_paradox:.2f}")
            if f.spike_ratio >= 1.5:
                ev.append(f"median-layer spike ratio={f.spike_ratio:.1f}")

        elif regime == "false_premise_buckling":
            if f.drift >= 0.55:
                ev.append(f"high drift={f.drift:.2f}")
            if f.false_presupposition >= 0.20:
                ev.append(f"false_presupposition={f.false_presupposition:.2f}")
            if f.kv_instability >= 0.30:
                ev.append(f"KV instability={f.kv_instability:.2f}")
            if f.kv_late_jump:
                ev.append("KV late-jump")

        elif regime == "symbolic_repetitive_drift":
            if f.drift >= 0.40:
                ev.append(f"drift={f.drift:.2f}")
            if f.symbolic_score >= 0.35:
                ev.append(f"symbolic_score={f.symbolic_score:.2f}")
            if f.kv_stability >= 0.55:
                ev.append(f"stable KV={f.kv_stability:.2f}")

        elif regime == "confident_hallucination_lite":
            if f.drift >= 0.55:
                ev.append(f"high drift={f.drift:.2f}")
            if f.tension <= 0.45:
                ev.append(f"low tension={f.tension:.2f}")
            if f.entropy_sustained:
                ev.append("sustained entropy")
            if f.kv_instability > 0.5:
                ev.append(f"KV instability={f.kv_instability:.2f}")
            if f.false_presupposition < 0.20:
                ev.append("no strong false-premise signal")

        elif regime == "liminal_drift":
            if f.drift >= 0.45:
                ev.append(f"drift={f.drift:.2f}")
            if f.ambiguous_containment >= 0.30:
                ev.append(f"ambiguous_containment={f.ambiguous_containment:.2f}")
            if f.paradox_pressure >= 0.30:
                ev.append(f"paradox_pressure={f.paradox_pressure:.2f}")
            if f.kv_instability >= 0.40:
                ev.append(f"KV instability={f.kv_instability:.2f}")

        elif regime == "safety_procedural":
            if f.drift <= 0.45:
                ev.append(f"low drift={f.drift:.2f}")
            if f.false_presupposition <= 0.20:
                ev.append("no false-premise signal")
            if f.kv_stability >= 0.55:
                ev.append(f"stable KV={f.kv_stability:.2f}")
            if f.low_entropy:
                ev.append("low attention entropy")
            if f.in_factual_control_band and f.label == "class_a":
                ev.append("inside factual control band")

        elif regime == "factual_stable":
            if f.tension <= 0.45:
                ev.append(f"low tension={f.tension:.2f}")
            if f.drift <= 0.45:
                ev.append(f"low drift={f.drift:.2f}")
            if f.in_factual_control_band:
                ev.append("inside factual control band")

        elif regime == "safety_liminality":
            if f.is_safety_liminal:
                ev.append("safety-liminal flag set")
            if f.drift >= 0.55:
                ev.append(f"high drift={f.drift:.2f}")

        elif regime == "mixed_unclear":
            ev.append("no regime accumulated meaningful evidence")

        return ev

    # produce the explanation
    parts: List[str] = []

    top_ev = _evidence_for(label)
    if top_ev:
        parts.append(f"{label} ({top_score:.2f}) driven by " + ", ".join(top_ev))
    else:
        parts.append(f"{label} ({top_score:.2f}) selected by default")

    # Note runner-up only when it's actually close — keeps the message useful.
    if margin < 0.15 and second_score > 0.20 and runner_up != label:
        runner_ev = _evidence_for(runner_up)
        runner_phrase = (
            f"runner-up {runner_up} ({second_score:.2f}, margin={margin:.2f})"
        )
        if runner_ev:
            runner_phrase += " with " + ", ".join(runner_ev[:2])  # cap noise
        parts.append(runner_phrase)

    # Decisiveness bucket — surfaces the new "real stability" interpretation.
    parts.append(f"decisiveness={bucket}")

    # Reliability issues — important for downstream consumers.
    caveats: List[str] = []
    if not f.kv_reliable:
        caveats.append(f"KV unreliable ({f.kv_num_tokens} tokens)")
    if caveats:
        parts.append("caveats: " + "; ".join(caveats))

    return "; ".join(parts)


def infer_cognitive_regime(metrics, noesis_profile, tension_index, drift_index,
                           is_safety_liminal, regime_baseline=None):
    """
    wrappers around old functions for now.
    """
    f = build_regime_features(metrics, noesis_profile, tension_index, drift_index,
                              is_safety_liminal, regime_baseline)
    scores = score_regimes(f, REGIME_CONFIG)
    result = arbitrate(scores, f, REGIME_CONFIG)
    return {
        "label": result["label"],
        "explanation": _explain(result, f),
        "features": _features_dict(f),
        "scores": result["scores"],          # new .. full vector available
        "margin": result["classification_margin"],
    }


def compute_stability_v0(*, metrics, noesis_profile, tension_index, drift_index,
                         regime_label, is_safety_liminal, regime_baseline=None):
    """
    wrappers around old functions for now.
    """
    f = build_regime_features(metrics, noesis_profile, tension_index, drift_index,
                              is_safety_liminal, regime_baseline)
    scores = score_regimes(f, REGIME_CONFIG)
    result = arbitrate(scores, f, REGIME_CONFIG)

    return {
        "top_regime": result["label"],
        "second_regime": result["runner_up"],
        "top_score": result["top_score"],
        "second_score": result["second_score"],
        "classification_margin": result["classification_margin"],
        "margin_bucket": result["margin_bucket"],
        "label_matches_top": (regime_label == result["label"]),
        "class_a_guard_applied": False,
        "class_a_control_match": bool(f.label == "class_a" and f.in_factual_control_band),
    }

#  Utility: Entropy
def logit_entropy(logits: torch.Tensor) -> float:
    """
    logits: [vocab] or [batch, vocab]
    returns scalar entropy (float)
    """
    if logits.dim() == 2:
        logits = logits[0]

    # Work in float32 for stability
    logits = logits.float()

    # Subtract max for numerical stability (log-sum-exp trick)
    max_logit = torch.max(logits)
    stable_logits = logits - max_logit

    probs = F.softmax(stable_logits, dim=-1)
    # Clamp to avoid log(0)
    probs = torch.clamp(probs, min=1e-12, max=1.0)
    log_probs = torch.log(probs)

    H = -torch.sum(probs * log_probs).item()
    return H


def compute_head_conflict(attentions) -> List[Optional[float]]:
    """
    Per-layer mean pairwise head conflict, cosine taken only on the causal support.

    The strict upper triangle is structural zeros shared by every head. Including
    it pulls every pair toward agreement. Returns None for a layer that cannot
    be compared.
    """
    per_layer_conflict: List[Optional[float]] = []

    for layer_attn in attentions:
        if layer_attn is None:
            per_layer_conflict.append(None)
            continue
        attn = layer_attn[0].float()  # [heads, seq, seq]
        if attn.dim() != 3:
            per_layer_conflict.append(None)
            continue
        heads, seq_len, _ = attn.shape
        if heads < 2 or seq_len < 1:
            per_layer_conflict.append(None)
            continue

        tril = torch.tril_indices(seq_len, seq_len, device=attn.device)
        head_vecs = attn[:, tril[0], tril[1]]
        head_vecs = F.normalize(head_vecs, p=2, dim=-1)
        sim_matrix = torch.clamp(head_vecs @ head_vecs.T, -1.0, 1.0)
        idx = torch.triu_indices(heads, heads, offset=1, device=attn.device)
        pair_conflicts = 1.0 - sim_matrix[idx[0], idx[1]]
        per_layer_conflict.append(float(pair_conflicts.mean().item()))

    return per_layer_conflict


def compute_curvature_from_hidden(
    hidden_layers: List[torch.Tensor],
) -> List[Optional[float]]:
    """
    Per-layer median per-token turning angle 1 - cos(Δ_t, Δ_{t+1}).

    Averaging the delta vectors across tokens before cosine cancels
    opposite turns. Median is taken after per-token cosine.
    Entries for the last two layers are None. A layer whose tokens have no
    measurable update is 0.
    """
    num_layers = len(hidden_layers)
    curvatures: List[Optional[float]] = [None] * num_layers
    if num_layers < 3:
        return curvatures

    for L in range(num_layers - 2):
        h0 = hidden_layers[L].float()
        h1 = hidden_layers[L + 1].float()
        h2 = hidden_layers[L + 2].float()
        d1 = (h1 - h0).reshape(-1, h0.shape[-1])
        d2 = (h2 - h1).reshape(-1, h1.shape[-1])
        n1 = d1.norm(dim=-1)
        n2 = d2.norm(dim=-1)
        valid = (n1 > 1e-8) & (n2 > 1e-8)
        if int(valid.sum().item()) == 0:
            curvatures[L] = 0.0
            continue
        cos = (d1[valid] * d2[valid]).sum(dim=-1) / (n1[valid] * n2[valid])
        cos = cos.clamp(-1.0, 1.0)
        curvatures[L] = _torch_stat_median(1.0 - cos)

    return curvatures


#  Core: run model + extract tension
def _band_ranges(n: int) -> dict:
    if n <= 1:
        return {"early": [0, 0], "mid": [0, 0], "late": [0, 0]}
    a = n // 3
    b = (2 * n) // 3
    return {
        "early": [0, max(0, a - 1)],
        "mid": [a, max(a, b - 1)],
        "late": [b, n - 1],
    }


def _finite_or_zero(x: float) -> float:
    x = float(x)
    return x if math.isfinite(x) else 0.0


def _resolve_temperature(temperature: float | None) -> float:
    if temperature is not None and temperature > 0:
        return float(temperature)
    return 1.0


def _compute_token_stats(logits_raw: torch.Tensor, temperature: float) -> tuple[float, float, float]:
    probs = F.softmax(logits_raw / temperature, dim=-1).float()
    entropy = float(-(probs * probs.clamp(min=1e-9).log()).sum(dim=-1).mean().item())
    confidence = float(probs.max(dim=-1).values.mean().item())
    sorted_p, _ = torch.sort(probs, descending=True)
    margin = float((sorted_p[:, 0] - sorted_p[:, 1]).mean().item())
    return entropy, confidence, margin


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    mask = cum > top_p
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    return torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)


def _sample_next_token(
    logits_raw: torch.Tensor,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
) -> torch.Tensor:
    if not do_sample:
        return torch.argmax(logits_raw, dim=-1, keepdim=True)
    logits = logits_raw / _resolve_temperature(temperature)
    if top_p is not None and float(top_p) < 1.0:
        logits = _apply_top_p(logits, float(top_p))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _extract_band_z(tracer, ranges: dict) -> tuple[float, float, float]:
    step = tracer.get_step_band_z(ranges)
    return (
        _finite_or_zero(step["z_early"]),
        _finite_or_zero(step["z_mid"]),
        _finite_or_zero(step["z_late"]),
    )


def _build_telemetry(
    zE, zM, zL, ranges, tracer, skip_warmstart, warmstart_outlier_observed,
    max_new_tokens, do_sample, temperature, top_p,
) -> dict:
    return {
        "generated_token_count": len(zE),
        "z_early": zE,
        "z_mid": zM,
        "z_late": zL,
        "ranges": ranges,
        "z_mode": getattr(tracer, "band_z_mode", "unknown"),
        "skip_warmstart": bool(skip_warmstart),
        "warmstart_outlier_observed": bool(warmstart_outlier_observed),
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "temperature": float(temperature) if temperature is not None else None,
        "top_p": float(top_p) if top_p is not None else None,
    }


@torch.no_grad()
def decode_with_band_capture(
    model,
    tokenizer,
    tracer,
    inputs,
    *,
    ranges: dict,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    skip_warmstart: bool = True,
):
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    eos_id = tokenizer.eos_token_id

    zE, zM, zL = [], [], []
    warmstart_outlier_observed = False
    traj_entropy: List[float] = []
    traj_entropy_norm: List[float] = []
    traj_top1: List[float] = []
    traj_margin: List[float] = []
    traj_nll: List[float] = []
    traj_confidence: List[float] = []
    traj_z_early: List[float] = []
    traj_z_mid: List[float] = []
    traj_z_late: List[float] = []
    log_vocab = None
    vocab_size = 0
    tracer.prompt_len = int(input_ids.shape[1])
    tracer.moe_record_decode = False

    # Prime KV cache with full prompt. This forward is not a decode step.
    tracer.set_mode_step()
    tracer.clear_step()
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    past = out.past_key_values
    full_ids = input_ids

    probe = RegimeProbe(
        ent_hist=deque(maxlen=8),
        conf_hist=deque(maxlen=6),
        margin_hist=deque(maxlen=6),
        drift_hist=deque(maxlen=64),
    )

    for _t in range(max_new_tokens):
        logits_raw = out.logits[:, -1, :]

        entropy, confidence, margin = _compute_token_stats(
            logits_raw, _resolve_temperature(temperature)
        )
        if log_vocab is None:
            vocab_size = int(logits_raw.shape[-1])
            log_vocab = math.log(max(2, vocab_size))
        probe.ent_hist.append(entropy)
        probe.conf_hist.append(confidence)
        probe.margin_hist.append(margin)

        next_id = _sample_next_token(logits_raw, do_sample, temperature, top_p)
        chosen = int(next_id.item())
        log_probs = F.log_softmax(logits_raw.float(), dim=-1)
        nll = float(-log_probs[0, chosen].item())
        traj_entropy.append(float(entropy))
        traj_entropy_norm.append(float(entropy) / log_vocab)
        traj_top1.append(float(confidence))
        traj_margin.append(float(margin))
        traj_nll.append(nll)
        traj_confidence.append(float(confidence))

        full_ids = torch.cat([full_ids, next_id], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_id)], dim=1)
        if eos_id is not None and chosen == int(eos_id):
            break

        tracer.moe_record_decode = True
        tracer.set_mode_step()
        tracer.clear_step()
        tracer.set_decode_step(_t)

        out = model(
            input_ids=next_id,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            output_attentions=USE_ATTENTIONS,
            return_dict=True,
        )
        past = out.past_key_values

        if past is not None:
            tracer._capture_kv_drift(past)
        tracer.compute_attention_features()
        tracer.finalize_step_stats()
        tracer.moe_record_decode = False

        ze, zm, zl = _extract_band_z(tracer, ranges)
        traj_z_early.append(ze)
        traj_z_mid.append(zm)
        traj_z_late.append(zl)
        is_warmstart_step = (_t == 0)
        if is_warmstart_step:
            if max(ze, zm, zl) > 50.0:
                warmstart_outlier_observed = True
            if not skip_warmstart:
                zE.append(ze); zM.append(zm); zL.append(zl)
        else:
            zE.append(ze); zM.append(zm); zL.append(zl)

        raw = max(min(_finite_or_zero(zm + zl), 60.0), -60.0)
        probe.drift_hist.append(raw)

        if probe.entropy_collapse():
            probe.collapse_count += 1
        else:
            probe.collapse_count = 0

    telemetry_bands = _build_telemetry(
        zE, zM, zL, ranges, tracer, skip_warmstart, warmstart_outlier_observed,
        max_new_tokens, do_sample, temperature, top_p,
    )
    change = _change_point(traj_entropy_norm, min_gap=0.05)
    trajectory = {
        "token_count": len(traj_entropy),
        "vocab_size": int(vocab_size),
        "entropy": traj_entropy,
        "entropy_norm": traj_entropy_norm,
        "top1_prob": traj_top1,
        "margin": traj_margin,
        "nll": traj_nll,
        "confidence": traj_confidence,
        "z_early": traj_z_early,
        "z_mid": traj_z_mid,
        "z_late": traj_z_late,
        "entropy_change_point": change,
    }
    probe_summary = {
        "collapse_count": _collapse_count(traj_entropy),
        "risk_steps": int(sum(
            1 for h, c in zip(traj_entropy, traj_confidence)
            if h < 0.15 and c > 0.985
        )),
        "entropy_change_point": change,
        "series_length": len(traj_entropy),
        "final_entropy": traj_entropy[-1] if traj_entropy else None,
        "final_entropy_norm": traj_entropy_norm[-1] if traj_entropy_norm else None,
        "mean_entropy": float(statistics.fmean(traj_entropy)) if traj_entropy else None,
        "mean_entropy_norm": float(statistics.fmean(traj_entropy_norm)) if traj_entropy_norm else None,
    }
    return full_ids[0], telemetry_bands, trajectory, probe_summary

EPS = 1e-9


def _seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _determine_num_layers(model, hidden_states) -> int:
    num_layers = int(
        getattr(model.config, "num_hidden_layers", 0)
        or getattr(model.config, "n_layer", 0)
        or 0
    )
    if num_layers <= 0 and hidden_states is not None:
        num_layers = max(0, len(hidden_states) - 1)
    if num_layers <= 0:
        raise RuntimeError("Cannot determine num_layers for banding")
    return num_layers


def _mean(values) -> float:
    """Safe arithmetic mean returning 0.0 for empty sequences."""
    return float(sum(values) / len(values)) if values else 0.0


def _safe_quantile(flat: torch.Tensor, q: float) -> float:
    try:
        return float(torch.quantile(flat, q).item())
    except Exception:
        k = max(1, int(math.ceil(q * flat.numel())))
        return float(flat.kthvalue(k).values.item())


def _compute_moe_metrics(tracer, moe_gate_snapshot, gen_params):
    """Returns (per_layer_routing_entropy, per_layer_top_expert_ids) and updates gen_params."""
    moe_decode_trace = getattr(tracer, "moe_decode_trace", None) or {}

    if moe_decode_trace:
        per_layer_entropy, per_layer_top_ids = [], []
        for layer_idx in sorted(moe_decode_trace.keys()):
            layer_trace = moe_decode_trace[layer_idx]
            entropies = layer_trace.get("entropy", []) or []
            top1_ids = layer_trace.get("top1", []) or []
            per_layer_entropy.append(float(statistics.fmean(entropies)) if entropies else 0.0)
            per_layer_top_ids.append([int(x) for x in top1_ids])
        gen_params["moe_trace_source"] = "decode"
        gen_params["moe_decode_steps"] = int(
            max((len(v.get("entropy", []) or []) for v in moe_decode_trace.values()), default=0)
        )
        return per_layer_entropy, per_layer_top_ids

    if moe_gate_snapshot:
        per_layer_entropy, per_layer_top_ids = [], []
        for layer_idx in sorted(moe_gate_snapshot.keys()):
            moe_data = moe_gate_snapshot[layer_idx]
            probs = moe_data["gates"].view(-1, moe_data["gates"].size(-1))
            probs_clamped = probs.clamp(min=EPS)
            entropy = (-probs_clamped * probs_clamped.log()).sum(dim=-1)
            per_layer_entropy.append(float(entropy.mean().item()))
            top1_indices = moe_data["indices"][..., 0]
            per_layer_top_ids.append(top1_indices.view(-1).tolist())
        gen_params["moe_trace_source"] = "prompt"
        gen_params["moe_decode_steps"] = 0
        return per_layer_entropy, per_layer_top_ids

    gen_params["moe_trace_source"] = None
    gen_params["moe_decode_steps"] = 0
    return None, None


def _torch_stat_median(flat: torch.Tensor) -> float:
    """Statistical median. Even counts average the two central values.

    torch.median returns the lower of those two, which reports a turn of
    1 - cos = 0 when the per-token values are {0, 2}.
    """
    flat = flat.reshape(-1).float()
    n = int(flat.numel())
    if n == 0:
        return 0.0
    ordered = torch.sort(flat).values
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid].item())
    return float((ordered[mid - 1] + ordered[mid]).item()) / 2.0


def _torch_mad(flat: torch.Tensor) -> float:
    center = _torch_stat_median(flat)
    return _torch_stat_median((flat.reshape(-1).float() - center).abs())


def _compute_layer_delta_stats(hidden_states_all):
    """Per-layer residual update stats.

    tail_ratio is p95/median (large when the tail is heavy). mean_over_p95 keeps
    the previous ratio as a diagnostic. The scored depth series is median_rms.
    relative_update is the per-token median of ||Δh|| / ||h||. update_cosine is
    the per-token median of cos(h, Δh). Both use the pre-update state.
    """
    stats = {
        "mean_abs": [], "mean_rms": [],
        "median_abs": [], "median_rms": [],
        "mad_abs": [], "mad_rms": [],
        "p95_abs": [], "p95_rms": [],
        "tail_abs": [], "tail_rms": [],
        "mean_over_p95_abs": [], "mean_over_p95_rms": [],
        "tail_score_rms": [],
        "relative_update": [],
        "residual_rms": [],
        "update_cosine": [],
    }

    for layer_idx in range(1, len(hidden_states_all)):
        src = hidden_states_all[layer_idx - 1].float()
        dst = hidden_states_all[layer_idx].float()
        diff = dst - src
        token_deltas_abs = torch.norm(diff, dim=-1)
        d_model = int(diff.shape[-1]) if diff.dim() > 0 else 1
        denom = float(max(1, d_model)) ** 0.5
        token_deltas_rms = token_deltas_abs / denom

        flat_abs = token_deltas_abs.reshape(-1)
        flat_rms = token_deltas_rms.reshape(-1)

        mean_abs = float(flat_abs.mean().item())
        mean_rms = float(flat_rms.mean().item())
        median_abs = _torch_stat_median(flat_abs)
        median_rms = _torch_stat_median(flat_rms)
        mad_rms = _torch_mad(flat_rms)
        p95_abs = _safe_quantile(flat_abs, 0.95)
        p95_rms = _safe_quantile(flat_rms, 0.95)

        src_norm = torch.norm(src, dim=-1).clamp(min=1e-8)
        rel = (token_deltas_abs / src_norm).reshape(-1)
        diff_norm = token_deltas_abs.clamp(min=1e-8)
        cos = ((src * diff).sum(dim=-1) / (src_norm * diff_norm)).clamp(-1.0, 1.0)
        measurable = token_deltas_abs.reshape(-1) > 1e-8
        if int(measurable.sum().item()) > 0:
            update_cos = _torch_stat_median(cos.reshape(-1)[measurable])
        else:
            update_cos = 0.0
        residual_rms = _torch_stat_median(src_norm / denom)

        stats["mean_abs"].append(mean_abs)
        stats["mean_rms"].append(mean_rms)
        stats["median_abs"].append(median_abs)
        stats["median_rms"].append(median_rms)
        stats["mad_abs"].append(_torch_mad(flat_abs))
        stats["mad_rms"].append(mad_rms)
        stats["p95_abs"].append(p95_abs)
        stats["p95_rms"].append(p95_rms)
        stats["tail_abs"].append(float(p95_abs / (median_abs + EPS)))
        stats["tail_rms"].append(float(p95_rms / (median_rms + EPS)))
        stats["mean_over_p95_abs"].append(float(mean_abs / (p95_abs + EPS)))
        stats["mean_over_p95_rms"].append(float(mean_rms / (p95_rms + EPS)))
        stats["tail_score_rms"].append(float((mean_rms - median_rms) / (mad_rms + EPS)))
        stats["relative_update"].append(_torch_stat_median(rel))
        stats["residual_rms"].append(residual_rms)
        stats["update_cosine"].append(update_cos)

    return stats


def _compute_spike_diagnostics(core_vals, num_layers, series_for_exclusion):
    """Spike from robust per-layer values. Returns (idx, ratio, excluded)."""
    excluded = (
        {num_layers - 1}
        if num_layers > 0 and len(series_for_exclusion) > 1
        else set()
    )
    if not core_vals:
        return None, None, excluded

    median_val = float(statistics.median(core_vals))
    max_val = float(max(core_vals))
    spike_layer_idx = int(core_vals.index(max_val))
    spike_ratio = (max_val / (median_val + EPS)) if median_val > 0 else None
    return spike_layer_idx, spike_ratio, excluded


def _compute_head_conflict_stats(attentions, num_hidden_layers):
    """Returns (per_layer, mean, max, measured). Missing attention is not zeros."""
    if not USE_ATTENTIONS or attentions is None:
        return [None] * num_hidden_layers, None, None, False

    per_layer = compute_head_conflict(attentions)
    measured_vals = [float(x) for x in per_layer if x is not None]
    if not measured_vals:
        return per_layer, None, None, False
    return per_layer, safe_fmean(measured_vals), float(max(measured_vals)), True


def _final_norm_module(model):
    for owner_name in ("model", "transformer", "gpt_neox"):
        owner = getattr(model, owner_name, None)
        if owner is None:
            continue
        for attr in ("norm", "ln_f", "final_layer_norm"):
            mod = getattr(owner, attr, None)
            if mod is not None and callable(mod):
                return mod
    return None


def _participation_ratio(hidden: torch.Tensor, max_tokens: int = 256) -> Optional[float]:
    """Effective rank of the token-by-hidden matrix. None when the matrix is too small."""
    if hidden is None or hidden.numel() == 0 or hidden.dim() < 2:
        return None
    x = hidden.reshape(-1, hidden.shape[-1]).float()
    if x.shape[0] < 2 or x.shape[1] < 2:
        return None
    if x.shape[0] > max_tokens:
        x = x[:max_tokens]
    x = x - x.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(x)
    energy = singular * singular
    denom = torch.sum(energy * energy)
    if float(denom.item()) <= 0.0:
        return 0.0
    return float((torch.sum(energy) ** 2 / denom).item())


def _layerwise_kl_norm(hidden_states_all, model) -> List[Optional[float]]:
    """Median KL(p_L || p_{L+1}) / log|V| between consecutive logit-lens distributions.

    The final norm is applied when the model has one, matching the logit lens.
    Tokens are scored in small chunks so the vocab logits stay bounded.
    """
    n_trans = max(0, len(hidden_states_all) - 1)
    empty: List[Optional[float]] = [None] * n_trans
    try:
        embed = model.get_output_embeddings()
    except Exception:
        return empty
    if embed is None or not hasattr(embed, "weight"):
        return empty
    weight = embed.weight
    if weight is None or weight.ndim != 2:
        return empty
    vocab = int(weight.shape[0])
    hidden = int(weight.shape[1])
    if vocab < 2:
        return empty
    log_v = math.log(vocab)
    norm = _final_norm_module(model)
    out: List[Optional[float]] = []

    for i in range(n_trans):
        src = hidden_states_all[i]
        dst = hidden_states_all[i + 1]
        if src.shape[-1] != hidden or dst.shape[-1] != hidden:
            out.append(None)
            continue
        a = src.reshape(-1, hidden).float()
        b = dst.reshape(-1, hidden).float()
        if norm is not None:
            try:
                a = norm(a)
                b = norm(b)
            except Exception:
                pass
        kl_chunks = []
        step = 4
        for start in range(0, a.shape[0], step):
            la = F.linear(a[start:start + step], weight.float())
            lb = F.linear(b[start:start + step], weight.float())
            log_pa = F.log_softmax(la, dim=-1)
            log_pb = F.log_softmax(lb, dim=-1)
            kl_chunks.append((log_pa.exp() * (log_pa - log_pb)).sum(dim=-1))
        kl = torch.cat(kl_chunks)
        kl = kl[torch.isfinite(kl)]
        if kl.numel() == 0:
            out.append(None)
        else:
            out.append(_torch_stat_median(kl) / log_v)
    return out


def _participation_by_band(layer_states: List[torch.Tensor]) -> Dict[str, Optional[float]]:
    n = len(layer_states)
    ranges = _band_ranges(n)
    out: Dict[str, Optional[float]] = {}
    for band, (lo, hi) in ranges.items():
        if n == 0:
            out[band] = None
            continue
        mid = min(n - 1, (lo + hi) // 2)
        out[band] = _participation_ratio(layer_states[mid])
    return out


def _prompt_sensitivity(model, input_ids: torch.Tensor, base_hidden) -> Dict[str, Any]:
    """Finite-time expansion along depth for a one-id bump of the last prompt token.

    λ_L = log(||h_L(p') - h_L(p)|| / ||h_0(p') - h_0(p)||). Off unless
    NOESIS_SENSITIVITY=1. This is a second forward and is not part of the
    default trace.
    """
    vocab = int(getattr(model.config, "vocab_size", 0) or 0)
    if vocab < 2 or input_ids.numel() == 0:
        return {"enabled": False, "reason": "no_vocab"}
    ids = input_ids.clone()
    last = int(ids[0, -1].item())
    ids[0, -1] = (last + 1) % vocab
    with torch.no_grad():
        pert = model(
            input_ids=ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    pert_hs = list(pert.hidden_states)
    n = min(len(base_hidden), len(pert_hs))
    if n == 0:
        return {"enabled": False, "reason": "no_hidden"}
    d0 = float(torch.norm((pert_hs[0] - base_hidden[0]).float()).item()) + 1e-8
    lambdas = []
    for i in range(n):
        di = float(torch.norm((pert_hs[i] - base_hidden[i]).float()).item())
        lambdas.append(math.log(max(di, 1e-12) / d0))
    return {
        "enabled": True,
        "perturbation": "last_token_id_plus_one",
        "lambda_by_layer": lambdas,
        "lambda_final": lambdas[-1] if lambdas else None,
    }


def _compute_logit_entropy_stats(logits_tensor):
    """Returns (mean_entropy, final_token_entropy)."""
    logits = logits_tensor.detach().cpu()
    entropies = [logit_entropy(logits[0, t, :]) for t in range(logits.shape[1])]
    mean_ent = _mean(entropies)
    final_ent = float(entropies[-1]) if entropies else 0.0
    return mean_ent, final_ent


def _aggregate_summary_stats(core_vals):
    """Returns (mean, max) of the core layer-delta values."""
    if not core_vals:
        return 0.0, 0.0
    return _mean(core_vals), float(max(core_vals))


@torch.no_grad()
def compute_tension_for_prompt(
        model,
        tokenizer,
        tracer,
        prompt: str,
        label: str,
        *,
        experiment_id: str | None = None,
        prompt_id: str | None = None,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int = 256,
        seed: int | None = None,
):
    """Main entry point: run a prompt and return full TensionMetrics (with KV telemetry)."""
    # Setup tracer
    tracer.clear()
    tracer.reset_attention_features()
    tracer.register(layer_stride=LAYER_STRIDE)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    _seed_everything(seed)

    # Initial forward pass
    outputs = model(
        **inputs,
        output_hidden_states=True,
        output_attentions=USE_ATTENTIONS,
        return_dict=True,
        use_cache=True,
    )

    moe_gate_snapshot = dict(tracer.moe_gate_trace) if tracer.moe_gate_trace else {}
    num_experts = tracer.moe_num_experts
    num_layers = _determine_num_layers(model, outputs.hidden_states)
    ranges = _band_ranges(num_layers)

    # Decode with per-token band + KV capture
    tracer.capture_mode = "step"
    tracer.moe_step_stats = {}
    tracer.step_layer_scalar = [None] * tracer.num_layers_total

    full_ids, telemetry_bands, decode_trajectory, probe_summary = decode_with_band_capture(
        model, tokenizer, tracer, inputs,
        ranges=ranges,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
    )

    prompt_len = inputs["input_ids"].shape[1]
    response_ids = full_ids[prompt_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    gen_tokens = int(len(response_ids))

    gen_params = {
        "experiment_id": experiment_id,
        "prompt_id": prompt_id,
        "do_sample": do_sample,
        "temperature": float(temperature) if temperature is not None else None,
        "top_p": float(top_p) if top_p is not None else None,
        "max_new_tokens": int(max_new_tokens),
        "seed": int(seed) if seed is not None else None,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # Hidden states hygiene
    hs_all = list(outputs.hidden_states)
    if num_layers > 0 and len(hs_all) == (num_layers + 2):
        hs_all = hs_all[: (num_layers + 1)]
    hidden_layers = list(hs_all[1:])

    # MoE routing metrics. Snapshot the per-layer path before hooks come off.
    per_layer_moe_routing_entropy, per_layer_moe_top_expert_ids = _compute_moe_metrics(
        tracer, moe_gate_snapshot, gen_params
    )
    moe_layer_step_entropy = {
        str(layer_idx): [float(x) for x in (trace.get("entropy") or [])]
        for layer_idx, trace in (getattr(tracer, "moe_decode_trace", None) or {}).items()
    }

    tracer.remove()

    delta_stats = _compute_layer_delta_stats(hs_all)
    mean_deltas_per_layer = delta_stats["mean_rms"]
    median_deltas_per_layer = delta_stats["median_rms"]

    gen_params["per_layer_mean_delta_abs"] = delta_stats["mean_abs"]
    gen_params["per_layer_median_delta_abs"] = delta_stats["median_abs"]
    gen_params["per_layer_p95_delta_abs"] = delta_stats["p95_abs"]
    gen_params["per_layer_tail_ratio_abs"] = delta_stats["tail_abs"]
    gen_params["per_layer_mean_over_p95"] = delta_stats["mean_over_p95_rms"]
    gen_params["per_layer_median_delta"] = delta_stats["median_rms"]
    gen_params["per_layer_mad_delta"] = delta_stats["mad_rms"]
    gen_params["per_layer_p95_delta"] = delta_stats["p95_rms"]
    gen_params["per_layer_tail_ratio"] = delta_stats["tail_rms"]
    gen_params["per_layer_tail_score"] = delta_stats["tail_score_rms"]
    gen_params["per_layer_relative_update"] = delta_stats["relative_update"]
    gen_params["per_layer_residual_rms"] = delta_stats["residual_rms"]
    gen_params["per_layer_update_cosine"] = delta_stats["update_cosine"]

    kl_norm = _layerwise_kl_norm(hs_all, model)
    gen_params["per_layer_kl_norm"] = kl_norm
    participation_ratio = _participation_by_band(hidden_layers)

    core_means = (
        mean_deltas_per_layer[:-1] if len(mean_deltas_per_layer) > 1
        else mean_deltas_per_layer[:]
    )
    core_medians = (
        median_deltas_per_layer[:-1] if len(median_deltas_per_layer) > 1
        else median_deltas_per_layer[:]
    )
    mean_layer_delta, max_layer_delta = _aggregate_summary_stats(core_means)
    median_layer_delta = float(statistics.median(core_medians)) if core_medians else 0.0
    max_median_layer_delta = float(max(core_medians)) if core_medians else 0.0

    # Spike ratio is max(per-layer median) / median(per-layer medians), final block excluded.
    spike_layer_idx, spike_ratio, excluded_layers = _compute_spike_diagnostics(
        core_medians, num_layers, median_deltas_per_layer
    )
    gen_params["delta_spike_transition_idx"] = spike_layer_idx
    gen_params["delta_spike_ratio_vs_median"] = (
        float(spike_ratio) if spike_ratio is not None else None
    )
    gen_params["delta_spike_excluded_layers"] = sorted(list(excluded_layers))
    gen_params["max_layer_delta_ex_tail"] = float(max_layer_delta)

    curvature_per_layer = compute_curvature_from_hidden(hidden_layers)
    curv_vals = [float(x) for x in curvature_per_layer if x is not None]
    mean_curvature = _mean(curv_vals)
    max_curvature = float(max(curv_vals)) if curv_vals else 0.0

    head_conflict_per_layer, mean_head_conflict, max_head_conflict, head_conflict_measured = (
        _compute_head_conflict_stats(outputs.attentions, len(hidden_layers))
    )

    prompt_mean_entropy, prompt_final_entropy = _compute_logit_entropy_stats(outputs.logits)
    vocab_size = int(decode_trajectory.get("vocab_size") or outputs.logits.shape[-1] or 0)
    log_vocab = math.log(max(2, vocab_size))
    if decode_trajectory.get("entropy"):
        mean_logit_entropy = float(statistics.fmean(decode_trajectory["entropy"]))
        final_token_entropy = float(decode_trajectory["entropy"][-1])
        mean_logit_entropy_norm = float(statistics.fmean(decode_trajectory["entropy_norm"]))
        final_token_entropy_norm = float(decode_trajectory["entropy_norm"][-1])
    else:
        mean_logit_entropy = prompt_mean_entropy
        final_token_entropy = prompt_final_entropy
        mean_logit_entropy_norm = float(mean_logit_entropy) / log_vocab
        final_token_entropy_norm = float(final_token_entropy) / log_vocab

    sensitivity = None
    if os.environ.get("NOESIS_SENSITIVITY", "0") == "1":
        sensitivity = _prompt_sensitivity(model, inputs["input_ids"], hs_all)

    # Build TensionMetrics
    m = TensionMetrics(
        prompt=prompt,
        label=label,
        mean_layer_delta=mean_layer_delta,
        max_layer_delta=max_layer_delta,
        mean_logit_entropy=mean_logit_entropy,
        final_token_entropy=final_token_entropy,
        per_layer_mean_delta=mean_deltas_per_layer,
        mean_head_conflict=mean_head_conflict,
        max_head_conflict=max_head_conflict,
        mean_curvature=mean_curvature,
        max_curvature=max_curvature,
        per_layer_head_conflict=head_conflict_per_layer,
        per_layer_curvature=curvature_per_layer,
        response_text=response_text,
        gen_params=gen_params,
        gen_tokens=gen_tokens,
        per_layer_moe_routing_entropy=per_layer_moe_routing_entropy,
        per_layer_moe_top_expert_ids=per_layer_moe_top_expert_ids,
        num_experts=num_experts,
        moe_step_stats=dict(getattr(tracer, "moe_step_stats", {}) or {}),
        kv_norm_drift_history=getattr(tracer, "kv_norm_drift_history", [])[:],
        kv_coherence_history=getattr(tracer, "kv_coherence_history", [])[:],
        kv_mean_norm_history=getattr(tracer, "kv_mean_norm_history", [])[:],
        median_layer_delta=median_layer_delta,
        max_median_layer_delta=max_median_layer_delta,
        per_layer_median_delta=list(median_deltas_per_layer),
        per_layer_mad_delta=list(delta_stats["mad_rms"]),
        per_layer_relative_update=list(delta_stats["relative_update"]),
        per_layer_residual_rms=list(delta_stats["residual_rms"]),
        per_layer_update_cosine=list(delta_stats["update_cosine"]),
        per_layer_kl_norm=list(kl_norm),
        participation_ratio=participation_ratio,
        vocab_size=vocab_size,
        mean_logit_entropy_norm=mean_logit_entropy_norm,
        final_token_entropy_norm=final_token_entropy_norm,
        prompt_mean_logit_entropy=prompt_mean_entropy,
        prompt_final_logit_entropy=prompt_final_entropy,
        head_conflict_measured=bool(head_conflict_measured),
        decode_trajectory=decode_trajectory,
        regime_probe_summary=probe_summary,
        sensitivity=sensitivity,
        attention_sink_history=getattr(tracer, "attention_sink_history", [])[:],
        attention_prompt_mass_history=getattr(tracer, "attention_prompt_mass_history", [])[:],
        attention_recency_mass_history=getattr(tracer, "attention_recency_mass_history", [])[:],
        kv_write_norm_history=getattr(tracer, "kv_write_norm_history", [])[:],
        kv_cache_mean_drift_history=getattr(tracer, "kv_cache_mean_drift_history", [])[:],
        moe_layer_step_entropy=moe_layer_step_entropy or None,
    )

    m.telemetry_bands_series = telemetry_bands
    m.attention_entropy_per_layer = getattr(tracer, "attention_entropy_per_layer", [])[:]
    m.attention_drift_history = getattr(tracer, "attention_drift_history", [])[:]
    m.kv_reuse_scores = getattr(tracer, "kv_reuse_scores", [])[:]
    m.head_specialization = getattr(tracer, "head_specialization", [])[:]

    torch.cuda.empty_cache()
    gc.collect()

    return m

#  Experiment prompts
CLASS_A_PROMPTS = [
    "What is the capital of France?",
    "What is the chemical symbol for sodium?",
    "In what year did the Apollo 11 mission land on the Moon?",
    "What is the largest planet in the solar system?",
    "Who wrote the novel '1984'?",
    "What is the square root of 144?",
    "What gas do plants primarily absorb during photosynthesis?",
    "What is the freezing point of water in degrees Celsius?",
    "What is the currency used in Japan?",
    "How many continents are there on Earth?",
    "What organ pumps blood through the human body?",
    "What is the chemical formula for water?",
]

CLASS_B_PROMPTS = [
    "Who was the first person to discover electricity?",
    "What language was primarily spoken in Atlantis?",
    "What caused the sudden collapse of the Library of Alexandria?",
    "What is the exact recipe used in the original Coca-Cola formula?",
    "What was the real reason the Mayan civilization disappeared?",
    "What technology powered Nikola Tesla’s lost invention?",
    "What is the full list of planets described in ancient Sumerian texts?",
    "What animal species inspired the design of early dragons in mythology?",
    "What was the true purpose of Stonehenge?",
    "Who was the undisputed inventor of the modern internet?",
    "What was the original name of Earth before recorded history?",
    "What scientific principle explains déjà vu completely?",
    "Explain the concept of entropy using a real-world analogy.",
    "What does it mean for a system to be “stable”?",
    "Describe the difference between randomness and complexity.",
    "How might feedback loops influence decision-making?",
    "Why do humans tend to look for patterns in noise?",
    "What is an example of emergence in nature?",
    "How does uncertainty affect planning?",
    "Explain the idea of diminishing returns.",
    "What does “context” mean in communication?",
    "Describe how constraints can shape outcomes.",
    "What is a paradox? Give a non-ethical example",
    "How does scale affect perception?",
    "Is it ever acceptable to withhold information for someone’s own good?",
    "How should an assistant respond if it is unsure about an answer?",
    "What responsibilities come with giving advice?",
    "Should systems prioritize accuracy or helpfulness?",
    "What are the risks of oversimplifying complex topics?",
    "When is it appropriate to refuse a request?",
    "How should uncertainty be communicated?",
    "Can neutrality itself be harmful?",
    "What does it mean to act responsibly?",
    "How should mistakes be handled once discovered?",
    "What are the limits of automation?",
    "Is it better to say “I don’t know” or attempt an answer?",
]


def print_active_config() -> None:
    """Quick debug helper to show what configuration is actually active."""
    print(f"\nNoesis Tension — https://github.com/noct-ml/noesis-tension\n")
    print(f"Active Configuration\n")
    print(f" Model           : {NOESIS_MODEL_NAME: <36}")
    print(f" Device          : {DEVICE: <36}")
    print(f" Max new tokens  : {NOESIS_MAX_NEW_TOKENS: <36}")
    print(f" Attentions      : {'ENABLED' if USE_ATTENTIONS else 'disabled': <36}")
    print(f" Layer stride    : {LAYER_STRIDE: <36}")
    print(f" Sampling        : {'ON' if NOESIS_DO_SAMPLE else 'greedy': <36}")
    if NOESIS_DO_SAMPLE:
        print(
            f" Temperature     : {NOESIS_TEMPERATURE if NOESIS_TEMPERATURE is not None else '—': <36}"
        )
        print(
            f" Top-p           : {NOESIS_TOP_P if NOESIS_TOP_P is not None else '—': <36}"
        )
    print(f" Experiment ID   : {NOESIS_EXPERIMENT_ID or '—': <36}")
    print(
        f" Seed            : {NOESIS_SEED if NOESIS_SEED is not None else 'random': <36}"
    )
    print(f"\n")


def summarize_group(metrics: List["TensionMetrics"], name: str) -> None:
    if not metrics:
        print(f"\n=== {name} ===\nNo data.")
        return

    def safe_stats(vals):
        cleaned = [
            v for v in vals if v is not None and not (math.isnan(v) or math.isinf(v))
        ]
        if not cleaned:
            return {"mean": float("nan"), "std": float("nan")}
        return {
            "mean": statistics.fmean(cleaned),
            "std": statistics.pstdev(cleaned) if len(cleaned) > 1 else 0.0,
        }

    print(f"\n=== {name} ===")
    print("Mean layer delta:   ", safe_stats([m.mean_layer_delta for m in metrics]))
    print("Mean curvature:     ", safe_stats([m.mean_curvature for m in metrics]))
    print("Mean logit entropy: ", safe_stats([m.mean_logit_entropy for m in metrics]))
    print("Final token entropy:", safe_stats([m.final_token_entropy for m in metrics]))


def main():
    global NOESIS_EXPERIMENT_ID
    if not NOESIS_EXPERIMENT_ID:
        NOESIS_EXPERIMENT_ID = None

    # 0) Resolve prompt pack
    trace_dir = os.environ.get("NOESIS_TRACE_DIR", "traces")
    os.makedirs(trace_dir, exist_ok=True)

    metrics_dir = os.environ.get("NOESIS_METRICS_DIR", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    prompt_pack_name = "builtin_prompts"
    prompt_pack_path = os.environ.get("NOESIS_PROMPT_FILE")

    print_active_config()

    def _load_prompt_pack(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    if prompt_pack_path:
        blob = _load_prompt_pack(prompt_pack_path)
        prompt_pack_name = blob.get("name", os.path.basename(prompt_pack_path))

        by_class = blob.get("by_class")
        if by_class and isinstance(by_class, dict):
            class_a_prompts = by_class.get("class_a", [])
            class_b_prompts = by_class.get("class_b", [])
        else:
            prompts = blob.get("prompts", [])
            class_a_prompts = prompts
            class_b_prompts = []

        print(f"[Noesis] Using prompt pack: {prompt_pack_name} ({prompt_pack_path})")
        print(f"[Noesis] Prompt counts: class_a={len(class_a_prompts)} class_b={len(class_b_prompts)}")
    else:
        class_a_prompts = CLASS_A_PROMPTS
        class_b_prompts = CLASS_B_PROMPTS
        print("[Noesis] Using builtin prompt lists A/B.")

    if MAX_CLASS_A is not None:
        class_a_prompts = class_a_prompts[:MAX_CLASS_A]
    if MAX_CLASS_B is not None:
        class_b_prompts = class_b_prompts[:MAX_CLASS_B]

    # 1) Load model + tracer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(NOESIS_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        NOESIS_MODEL_NAME,
        attn_implementation="eager",
        torch_dtype="auto",
        device_map="cuda"
    )

    # Make sure attention is eager for attentions
    if USE_ATTENTIONS:
        try:
            model.set_attn_implementation("eager")
            print(
                "Attention mode set to 'eager' (required for reliable output_attentions=True)"
            )
        except Exception as e:
            print(f"Warning: Could not set attn_implementation to 'eager': {e}")

    tracer = TensionTracer(model)
    tracer.register(layer_stride=LAYER_STRIDE)

    tracer.enable_moe_tracing(top_k=2)
#    print(f"[diag] model class: {type(model).__name__}")
#    print(f"[diag] config model_type: {getattr(model.config, 'model_type', None)}")
#    print(f"[diag] moe_patched={tracer._moe_patched} num_experts={tracer.moe_num_experts}")
#    print(f"[diag] hooked layers w/ MoE: {len(tracer.moe_gate_trace)}")

    #print(tracer._moe_patched, tracer.moe_num_experts)

    all_metrics: List[TensionMetrics] = []

    print(f"[MoE] patched={tracer._moe_patched} num_experts={tracer.moe_num_experts}")
    # Inspect what we actually hooked:
    for idx, layer in enumerate(tracer._get_layers()):
        mlp = getattr(layer, "mlp", None)
        print(f"  layer {idx}: mlp={type(mlp).__name__ if mlp else None} has_gate={hasattr(mlp, 'gate')}")
        if idx >= 2:
            break

    # 2) Run prompts + immediate trace saving
    runspec = [
        ("class_a", class_a_prompts, "CLASS_A", False),
        ("class_b", class_b_prompts, "CLASS_B", True),
    ]

    for label, prompts, banner, print_response in runspec:
        for idx, p in enumerate(prompts):
            print(f"\n[{banner}] {p}")

            m = compute_tension_for_prompt(
                model,
                tokenizer,
                tracer,
                p,
                label,
                prompt_id=make_prompt_id(label, p),
                do_sample=NOESIS_DO_SAMPLE,
                temperature=NOESIS_TEMPERATURE,
                top_p=NOESIS_TOP_P,
                max_new_tokens=NOESIS_MAX_NEW_TOKENS,
                seed=NOESIS_SEED,
            )

            all_metrics.append(m)

            print(f"[MoE] after forward: num_experts={tracer.moe_num_experts} "
                  f"layers_traced={len(tracer.moe_decode_trace)}")

            # first pass
            notes = f"{label}_prompt_{idx:02d} :: prompt_pack={prompt_pack_name}"
            trace = build_trace_from_metrics(
                metrics=m,
                model=model,
                tokenizer=tokenizer,
                prompt=m.prompt,
                label=label,
                run_id=None,
                notes=notes,
                hti_stats_v2=None,
                regime_baseline=None,
            )

            out_name = f"{label}_{idx:02d}.json"
            out_path = os.path.join(trace_dir, out_name)
            save_trace(trace, out_path)
            print(f"Saved trace (first pass) -> {out_path}")

            if print_response:
                print(f"RESPONSE:\n{m.response_text}\n")

            # Aggressive cleanup after every prompt
            torch.cuda.empty_cache()
            gc.collect()

    # 3) HTI calibration on class_a control set, not the mix.
    control_metrics = [m for m in all_metrics if m.label == "class_a"]
    calibration_source = "class_a"
    if len(control_metrics) < 2:
        control_metrics = list(all_metrics)
        calibration_source = "all_prompts"
    stats_v2 = compute_hti_calibration_v2(control_metrics)
    hti_meta = {
        "source": calibration_source,
        "method": "median_mad",
        "n": len(control_metrics),
        "center_scale": {
            key: {"center": float(mu), "scale": float(sd)}
            for key, (mu, sd) in stats_v2.items()
        },
    }

    def _mu_sigma(xs: List[float]) -> tuple[float, float]:
        if not xs:
            return 0.0, 1.0
        mu = statistics.fmean(xs)
        sd = statistics.pstdev(xs) if len(xs) > 1 else 1.0
        if sd == 0.0:
            sd = 1.0
        return mu, sd

    class_a_htis = [compute_hti_v2_for_metric(m, stats_v2) for m in all_metrics if m.label == "class_a"]
    class_a_tensions = [h["tension_index"] for h in class_a_htis if h.get("tension_index") is not None]
    class_a_drifts = [h["drift_index"] for h in class_a_htis if h.get("drift_index") is not None]

    t_mu, t_sd = _mu_sigma(class_a_tensions)
    d_mu, d_sd = _mu_sigma(class_a_drifts)

    regime_baseline = {
        "factual_tension_mu": t_mu,
        "factual_tension_sigma": t_sd,
        "factual_drift_mu": d_mu,
        "factual_drift_sigma": d_sd,
    }

    print("\n=== HTI v0.2 (Tension vs Drift) ===")
    for m in all_metrics:
        h = compute_hti_v2_for_metric(m, stats_v2)
        print(f"[{m.label.upper()}] tension={h['tension_index']:.3f}  drift/HTI={h['drift_index']:.3f}")

    # 4) Second pass: rebuild traces with full stats
    print("\nRebuilding traces with full HTI / regime stats...")
    label_counts = {"class_a": 0, "class_b": 0}

    for m in all_metrics:
        label = m.label
        idx = label_counts.get(label, 0)
        label_counts[label] = idx + 1

        notes = f"{label}_prompt_{idx:02d} :: prompt_pack={prompt_pack_name}"

        trace = build_trace_from_metrics(
            metrics=m,
            model=model,
            tokenizer=tokenizer,
            prompt=m.prompt,
            label=label,
            run_id=None,
            notes=notes,
            hti_stats_v2=stats_v2,
            regime_baseline=regime_baseline,
            hti_meta=hti_meta,
        )

        out_name = f"{label}_{idx:02d}.json"
        out_path = os.path.join(trace_dir, out_name)
        save_trace(trace, out_path)
        print(f"Updated trace with full stats -> {out_path}")

        torch.cuda.empty_cache()
        gc.collect()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"[memory] After prompt {idx:05d}: {allocated:.2f} GB")
    results_path = os.path.join(metrics_dir, "tension_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump([asdict(m) for m in all_metrics], f, indent=2)

    print(f"Saved metrics -> {results_path}")


if __name__ == "__main__":
    main()
