"""Report formatting helper functions."""
from __future__ import annotations
import re
from typing import Iterable, Any

def extract_feature_name(condition: str, feature_names: Iterable[str]) -> str:
    """Return the exact feature name present in the LIME condition string.

    The function scans *condition* looking for any name in *feature_names*.
    When none is found it falls back to the last alphanumeric token.
    """
    for fname in feature_names:
        if fname in condition:
            return fname
    tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", condition)
    return tokens[-1] if tokens else condition.strip()


def format_value(val: Any) -> str:
    """Format a numeric or textual value to be user-friendly for reports."""
    if isinstance(val, (int, float)):
        if isinstance(val, float) and not val.is_integer():
            return f"{val:.4f}"
        return f"{int(val)}"
    return str(val)


def get_interpretation(impact: float) -> str:
    """Return a textual interpretation according to the sign of *impact*."""
    if impact > 0.001:
        return "Aumenta probabilidade de ATAQUE"
    if impact < -0.001:
        return "Aumenta probabilidade de NORMAL"
    return "Neutro"
