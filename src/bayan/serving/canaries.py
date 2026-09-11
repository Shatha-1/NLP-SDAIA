"""Lab 7/capstone starter: startup skew and behaviour canaries."""
from pathlib import Path

from bayan.preprocessing.core import preprocess

PINNED_INPUT = "الخدمة ممتازة"


class CanaryFailure(RuntimeError):
    """Raised to fail serving startup fast rather than accept bad traffic."""


def run_startup_canaries(serving_dir: str, rollback_dir: str, predict_fn, rollback_predict_fn) -> None:
    """Assert train/serve skew and artefact health before accepting traffic.

    - both the serving artefact and its fp32 rollback must exist on disk
    - the shared preprocessing contract must not silently produce empty text
      for a known-good pinned input
    - the serving artefact must produce a prediction for the pinned input
    - that prediction must agree with the fp32 rollback artefact -- this is
      what catches a corrupted/broken quantised export before it serves traffic
    """
    if not Path(serving_dir).exists():
        raise CanaryFailure(f"Serving artefact not found: {serving_dir}")
    if not Path(rollback_dir).exists():
        raise CanaryFailure(f"Rollback artefact not found: {rollback_dir}")

    pinned_text = preprocess(PINNED_INPUT)
    if not pinned_text:
        raise CanaryFailure("Pinned canary input produced empty text after preprocessing")

    try:
        serving_pred = predict_fn(PINNED_INPUT)
    except Exception as exc:
        raise CanaryFailure(f"Serving artefact ({serving_dir}) failed on pinned input: {exc}") from exc

    try:
        rollback_pred = rollback_predict_fn(PINNED_INPUT)
    except Exception as exc:
        raise CanaryFailure(f"Rollback artefact ({rollback_dir}) failed on pinned input: {exc}") from exc

    if serving_pred != rollback_pred:
        raise CanaryFailure(
            f"Serving artefact ({serving_dir}) disagrees with fp32 rollback ({rollback_dir}) "
            f"on the pinned canary input: {serving_pred!r} != {rollback_pred!r}"
        )
