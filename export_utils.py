"""export_utils.py

Utility functions for saving artifacts and standardising the output format for
both CLFE (Linear Classification Federated & Explainable) and ADF (Federated
Decision Tree) approaches.

This module centralises all file–export-related logic so that the rest of the
codebase can focus on doing the computation.  Keeping these helpers in a single
place makes it easier to:
    • guarantee a consistent directory structure across clients/approaches;
    • change file formats later without touching the business-logic code;
    • share common functionality such as saving JSON metadata or generating
      decision-tree images.

Typical usage inside CLFE/ADF components::

    from export_utils import (
        save_metadata, save_json, save_numpy,
        save_lime_explanation, save_shap_values,
        save_tree_visualisation,
    )

    # Save arbitrary python dict as JSON
    save_metadata(my_dict, output_dir / "meta.json")

    # Save a decision-tree visualisation (ADF)
    save_tree_visualisation(clf, feature_names, class_names, output_dir)

NOTE
----
In order to keep dependencies lightweight and optional, the functions that
require external libraries (e.g. ``dtreeviz`` or ``graphviz``) will attempt to
import them at call time and log a clear error message if the library is
missing instead of crashing the entire program.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _ensure_parent_dir(path: Path | str) -> Path:
    """Create parent directories of *path* if they don't exist and return Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _to_serialisable(obj: Any) -> Any:
    """Convert *obj* to a JSON-serialisable representation if needed."""
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    return obj


# ---------------------------------------------------------------------------
# Public API – generic save helpers
# ---------------------------------------------------------------------------

def save_json(data: Mapping[str, Any], destination: Path | str, **json_kwargs) -> Path:
    """Serialise *data* to *destination* (``.json``).

    Parameters
    ----------
    data : dict
        Data to serialise. Non-serialisable objects (NumPy arrays, ``datetime``)
        are automatically converted to JSON-friendly formats.
    destination : str or Path
        Target file path (will be created/overwritten).
    **json_kwargs
        Additional keyword arguments forwarded to ``json.dump``.
    """
    destination = _ensure_parent_dir(destination)

    serialisable_data = {k: _to_serialisable(v) for k, v in data.items()}
    # Default JSON options: pretty print & UTF-8
    json_kwargs.setdefault("ensure_ascii", False)
    json_kwargs.setdefault("indent", 2)

    try:
        with destination.open("w", encoding="utf-8") as fp:
            json.dump(serialisable_data, fp, **json_kwargs)
        logger.info("Saved JSON to %s", destination)
    except Exception as exc:  # pragma: no cover – log & propagate
        logger.error("Could not save JSON to %s – %s", destination, exc, exc_info=True)
        raise

    return destination


def save_numpy(array: np.ndarray, destination: Path | str) -> Path:
    """Save a NumPy array to *destination* with ``np.save`` (``.npy``)."""
    destination = _ensure_parent_dir(destination)
    try:
        np.save(destination, array)
        logger.info("Saved NumPy array to %s", destination)
    except Exception as exc:
        logger.error("Could not save NumPy array to %s – %s", destination, exc, exc_info=True)
        raise
    return destination


def save_pickle(obj: Any, destination: Path | str) -> Path:
    """Pickle *obj* to *destination* (``.pkl``)."""
    destination = _ensure_parent_dir(destination)
    try:
        with destination.open("wb") as fp:
            pickle.dump(obj, fp)
        logger.info("Saved pickle to %s", destination)
    except Exception as exc:
        logger.error("Could not pickle object to %s – %s", destination, exc, exc_info=True)
        raise
    return destination


# ---------------------------------------------------------------------------
# CLFE helpers (LIME / SHAP)
# ---------------------------------------------------------------------------

def save_lime_explanation(lime_exp, destination: Path | str) -> Path:
    """Save a *lime_exp* explanation object to text & JSON formats.

    If *lime_exp* is ``None`` nothing is saved and *None* is returned.
    """
    if lime_exp is None:
        logger.warning("No LIME explanation provided, skipping save.")
        return None

    destination = Path(destination)
    text_dest = destination.with_suffix(".txt")
    json_dest = destination.with_suffix(".json")

    # Save basic text representation (as_list)
    try:
        with text_dest.open("w", encoding="utf-8") as fp:
            fp.write(str(lime_exp.as_list()))
        logger.info("Saved LIME explanation (txt) to %s", text_dest)
    except Exception as exc:
        logger.error("Could not save LIME txt – %s", exc, exc_info=True)

    # Save full JSON representation if available
    try:
        exp_dict = lime_exp.as_map()
        save_json(exp_dict, json_dest)
    except Exception as exc:
        logger.error("Could not save LIME JSON – %s", exc, exc_info=True)

    return text_dest  # Return the main (txt) path for convenience


def save_shap_values(shap_values: Any, feature_names: Sequence[str], destination: Path | str) -> Path:
    """Save SHAP values along with *feature_names* to a ``.json`` file."""
    if shap_values is None:
        logger.warning("No SHAP values provided, skipping save.")
        return None

    destination = Path(destination).with_suffix(".json")
    shap_data = {
        "feature_names": list(feature_names),
        "shap_values": _to_serialisable(shap_values),
    }
    return save_json(shap_data, destination)


# ---------------------------------------------------------------------------
# ADF helpers (Decision-Tree visualisation)
# ---------------------------------------------------------------------------

def save_tree_visualisation(
    clf,
    feature_names: Sequence[str],
    class_names: Sequence[str],
    destination: Path | str,
    *,
    dpi: int = 200,
    fmt: str = "png",
    **graphviz_kwargs,
) -> Path | None:
    """Generate and save a decision-tree visualisation.

    Uses ``sklearn.tree.export_graphviz`` and the Graphviz ``dot`` command-line
    tool (must be installed) to render an image.

    Returns the path of the generated image or ``None`` if something failed.
    """
    from sklearn import tree as _sk_tree  # Local import to avoid hard dependency
    import subprocess

    destination = Path(destination).with_suffix(f".{fmt}")
    dot_path = destination.with_suffix(".dot")

    # Export to DOT format first
    try:
        _ensure_parent_dir(dot_path)
        _sk_tree.export_graphviz(
            clf,
            out_file=str(dot_path),
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            special_characters=True,
            **graphviz_kwargs,
        )
        logger.info("Exported decision tree DOT to %s", dot_path)
    except Exception as exc:
        logger.error("Could not export decision tree to DOT – %s", exc, exc_info=True)
        return None

    # Render DOT → image using Graphviz
    try:
        cmd = ["dot", f"-T{fmt}", str(dot_path), "-o", str(destination), f"-Gdpi={dpi}"]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info("Rendered decision tree image to %s", destination)
    except FileNotFoundError:
        logger.error(
            "Graphviz 'dot' executable not found. Install Graphviz to enable tree visualisation.")
        return None
    except subprocess.CalledProcessError as exc:
        logger.error("Graphviz rendering failed: %s", exc.stderr.decode())
        return None

    finally:
        # Clean up intermediate DOT file
        try:
            dot_path.unlink(missing_ok=True)
        except Exception:
            pass

    return destination

# ---------------------------------------------------------------------------
# Convenience wrapper for meta information
# ---------------------------------------------------------------------------

def save_metadata(
    client_id: int,
    approach: str,
    dataset_path: str | Path,
    seed: int,
    epochs: int,
    output_dir: Path | str,
    extra_info: Mapping[str, Any] | None = None,
) -> Path:
    """Save a metadata file (``meta.json``) with common fields.

    The returned path points to the created JSON file.
    """
    meta = {
        "client_id": client_id,
        "approach": approach,
        "dataset_path": str(dataset_path),
        "seed": seed,
        "epochs": epochs,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    if extra_info:
        meta.update(extra_info)

    output_dir = Path(output_dir)
    _ensure_parent_dir(output_dir / "dummy")  # Ensure directory exists
    return save_json(meta, output_dir / "meta.json")
