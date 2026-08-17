"""Gradient boosting probability engine for football match analysis.

This engine mirrors the public contract of ``probability_engine.py`` but uses
the models trained by ``app/core/gradient_boosting_models.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "matches_features.csv"
GB_MODELS_DIR = PROJECT_ROOT / "models" / "gradient_boosting_models"

MODEL_PATHS = {
    "home_win": GB_MODELS_DIR / "home_win_gb_model.pkl",
    "away_win": GB_MODELS_DIR / "away_win_gb_model.pkl",
    "draw": GB_MODELS_DIR / "draw_gb_model.pkl",
}

MODEL_VERSION = "gradient_boosting_v1"


def _load_model_artifact(model_path: Path, label: str) -> Any:
    """Load a model artifact or raise a clear error."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {label} model: {model_path}. "
            "Run `python app/core/gradient_boosting_models.py` before using the GB engine."
        )
    return joblib.load(model_path)


def _unwrap_model(artifact: Any) -> Any:
    """Return the estimator from either a bare model or an artifact dict."""
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"]
    return artifact


@lru_cache(maxsize=1)
def _load_models() -> dict[str, Any]:
    """Load all GB outcome models."""
    return {name: _load_model_artifact(path, f"gradient boosting {name}") for name, path in MODEL_PATHS.items()}


def _feature_columns_for_model(model: Any) -> list[str] | None:
    """Return model feature names when sklearn exposes them."""
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        return list(feature_names)
    return None


def _predict_positive_probability(model: Any, features: pd.DataFrame, model_name: str) -> float:
    """Predict the probability of class 1 for a binary model."""
    feature_columns = _feature_columns_for_model(model)
    if feature_columns:
        missing_columns = [column for column in feature_columns if column not in features.columns]
        if missing_columns:
            raise ValueError(
                f"Missing features for {model_name}: {', '.join(missing_columns)}. "
                "Provide the same feature columns used during training."
            )
        features = features.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
        features = features.replace([np.inf, -np.inf], np.nan)

    probabilities = model.predict_proba(features)
    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        raise ValueError(f"Model {model_name} cannot predict positive class 1.")
    positive_index = classes.index(1)
    return float(probabilities[0, positive_index])


def _normalize_probabilities(home_win: float, draw: float, away_win: float) -> dict[str, float]:
    """Normalize raw H/D/A probabilities so they sum to 1."""
    total = home_win + draw + away_win
    if total <= 0:
        raise ValueError("Raw probabilities sum to zero or less; cannot normalize.")
    return {
        "home_win": home_win / total,
        "draw": draw / total,
        "away_win": away_win / total,
    }


def _favorite(probabilities: dict[str, float]) -> tuple[str, float]:
    """Return the favorite side and probability from normalized match probabilities."""
    home_win = probabilities["home_win"]
    away_win = probabilities["away_win"]
    if home_win == away_win:
        return "none", home_win
    if home_win > away_win:
        return "home", home_win
    return "away", away_win


def _predicted_class_argmax(home_win_probability: float, draw_probability: float, away_win_probability: float) -> str:
    """Pick the class with the highest probability."""
    classes = {
        "H": home_win_probability,
        "D": draw_probability,
        "A": away_win_probability,
    }
    return max(classes, key=classes.get)


def _predicted_class_adjusted(
    home_win_probability: float,
    draw_probability: float,
    away_win_probability: float,
) -> str:
    """Apply the draw adjustment rule used by the current engine."""
    argmax_class = _predicted_class_argmax(home_win_probability, draw_probability, away_win_probability)
    max_probability = max(home_win_probability, draw_probability, away_win_probability)
    if draw_probability >= 0.27 and (max_probability - draw_probability) <= 0.08:
        return "D"
    return argmax_class


def _recommended_prediction_class(
    league: str | None,
    predicted_class_argmax: str,
    predicted_class_adjusted: str,
) -> str:
    """Return the league-aware recommendation."""
    if league in {"Bundesliga", "Serie A"}:
        return predicted_class_adjusted
    if league in {"Premier League", "La Liga", "Ligue 1"}:
        return predicted_class_argmax
    return predicted_class_argmax


def _match_profile(
    home_win_probability: float,
    draw_probability: float,
    away_win_probability: float,
    top_two_margin: float,
    uncertainty_score: float,
) -> str:
    """Classify the broad shape of the probability distribution."""
    if home_win_probability >= 0.55:
        return "clear_home_advantage"
    if away_win_probability >= 0.55:
        return "clear_away_advantage"
    if draw_probability >= 0.32:
        return "very_strong_draw_signal"
    if draw_probability >= 0.30:
        return "strong_draw_signal"
    if top_two_margin <= 0.07:
        return "balanced_match"
    if draw_probability >= 0.27 and top_two_margin <= 0.10:
        return "draw_plausible"
    if uncertainty_score >= 0.62:
        return "high_uncertainty"
    return "moderate_advantage"


def _confidence_score(favorite_probability: float, top_two_margin: float, uncertainty_score: float) -> float:
    """Compute a bounded confidence score from the probability shape."""
    score = (favorite_probability * 70) + (top_two_margin * 80) - (uncertainty_score * 35)
    return round(max(0.0, min(100.0, score)), 2)


def _round_probability(value: float) -> float:
    """Round probabilities for JSON-compatible output."""
    return round(float(value), 6)


def _extract_analysis_fields(
    league: str | None,
    home_win_probability: float,
    draw_probability: float,
    away_win_probability: float,
    calibrated_draw_signal: float | None,
) -> dict[str, Any]:
    """Build the analysis section compatible with the existing engine."""
    sorted_probabilities = sorted([home_win_probability, draw_probability, away_win_probability], reverse=True)
    top_two_margin = sorted_probabilities[0] - sorted_probabilities[1]
    uncertainty_score = 1 - sorted_probabilities[0]
    favorite_team, favorite_probability = _favorite(
        {
            "home_win": home_win_probability,
            "draw": draw_probability,
            "away_win": away_win_probability,
        }
    )
    predicted_class_argmax = _predicted_class_argmax(home_win_probability, draw_probability, away_win_probability)
    predicted_class_adjusted = _predicted_class_adjusted(home_win_probability, draw_probability, away_win_probability)
    recommended_prediction_class = _recommended_prediction_class(
        league=league,
        predicted_class_argmax=predicted_class_argmax,
        predicted_class_adjusted=predicted_class_adjusted,
    )
    draw_warning = predicted_class_adjusted == "D" and recommended_prediction_class != "D"
    match_profile = _match_profile(
        home_win_probability=home_win_probability,
        draw_probability=draw_probability,
        away_win_probability=away_win_probability,
        top_two_margin=top_two_margin,
        uncertainty_score=uncertainty_score,
    )
    confidence_score = _confidence_score(
        favorite_probability=favorite_probability,
        top_two_margin=top_two_margin,
        uncertainty_score=uncertainty_score,
    )
    is_draw_plausible = draw_probability >= 0.27 or (calibrated_draw_signal is not None and calibrated_draw_signal >= 0.27)
    is_strong_draw_signal = draw_probability >= 0.30 or (calibrated_draw_signal is not None and calibrated_draw_signal >= 0.30)
    is_very_strong_draw_signal = draw_probability >= 0.32 or (
        calibrated_draw_signal is not None and calibrated_draw_signal >= 0.32
    )

    return {
        "model_version": MODEL_VERSION,
        "main_probability_source": "gradient_boosting_models",
        "draw_signal_source": "none",
        "calibrated_draw_signal": _round_probability(calibrated_draw_signal) if calibrated_draw_signal is not None else None,
        "favorite_team": favorite_team,
        "favorite_probability": _round_probability(favorite_probability),
        "uncertainty_score": _round_probability(uncertainty_score),
        "top_two_margin": _round_probability(top_two_margin),
        "match_profile": match_profile,
        "confidence_score": confidence_score,
        "predicted_class_argmax": predicted_class_argmax,
        "predicted_class_adjusted": predicted_class_adjusted,
        "recommended_prediction_class": recommended_prediction_class,
        "draw_warning": bool(draw_warning),
        "is_draw_plausible": bool(is_draw_plausible),
        "is_strong_draw_signal": bool(is_strong_draw_signal),
        "is_very_strong_draw_signal": bool(is_very_strong_draw_signal),
        "league": league,
    }


def predict_match_probabilities_gb(match_features: dict[str, Any]) -> dict[str, Any]:
    """Predict normalized football match probabilities from one feature dictionary."""
    if not match_features:
        raise ValueError("match_features cannot be empty.")

    features = pd.DataFrame([match_features])
    models = _load_models()
    home_win_model = _unwrap_model(models["home_win"])
    away_win_model = _unwrap_model(models["away_win"])
    draw_model = _unwrap_model(models["draw"])

    raw_home_win = _predict_positive_probability(home_win_model, features, "home_win_gb_model")
    raw_away_win = _predict_positive_probability(away_win_model, features, "away_win_gb_model")
    raw_draw = _predict_positive_probability(draw_model, features, "draw_gb_model")

    normalized = _normalize_probabilities(raw_home_win, raw_draw, raw_away_win)
    home_win_probability = normalized["home_win"]
    draw_probability = normalized["draw"]
    away_win_probability = normalized["away_win"]

    league = match_features.get("league")
    calibrated_draw_signal = None

    analysis = _extract_analysis_fields(
        league=league,
        home_win_probability=home_win_probability,
        draw_probability=draw_probability,
        away_win_probability=away_win_probability,
        calibrated_draw_signal=calibrated_draw_signal,
    )

    return {
        "probabilities": {
            "home": {
                "win": _round_probability(home_win_probability),
                "draw": _round_probability(draw_probability),
                "loss": _round_probability(away_win_probability),
                "no_loss": _round_probability(home_win_probability + draw_probability),
            },
            "away": {
                "win": _round_probability(away_win_probability),
                "draw": _round_probability(draw_probability),
                "loss": _round_probability(home_win_probability),
                "no_loss": _round_probability(away_win_probability + draw_probability),
            },
            "match": {
                "home_win": _round_probability(home_win_probability),
                "draw": _round_probability(draw_probability),
                "away_win": _round_probability(away_win_probability),
            },
        },
        "raw_probabilities": {
            "home_win": _round_probability(raw_home_win),
            "draw": _round_probability(raw_draw),
            "away_win": _round_probability(raw_away_win),
        },
        "analysis": analysis,
    }


def predict_from_latest_features_csv(row_index: int = -1) -> dict[str, Any]:
    """Predict probabilities from one row of ``matches_features.csv``."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_PATH}. "
            "Run `python app/data_pipeline/build_features.py` first."
        )

    features = pd.read_csv(FEATURES_PATH)
    if features.empty:
        raise ValueError(f"Feature file is empty: {FEATURES_PATH}")

    try:
        row = features.iloc[row_index]
    except IndexError as exc:
        raise IndexError(f"row_index {row_index} is out of range for {len(features)} rows.") from exc

    ignored_columns = {
        "Date",
        "date",
        "match_date",
        "HomeTeam",
        "home_team",
        "AwayTeam",
        "away_team",
        "FTR",
        "result",
        "target",
    }
    league_value = None
    for column in ("league", "League"):
        if column in row.index and pd.notna(row[column]):
            league_value = row[column]
            break
    match_features = {key: value for key, value in row.to_dict().items() if key not in ignored_columns}
    if league_value is not None:
        match_features["league"] = league_value
    result = predict_match_probabilities_gb(match_features)

    print(f"Using row_index={row_index} from: {FEATURES_PATH}")
    if "HomeTeam" in row and "AwayTeam" in row:
        print(f"Match: {row['HomeTeam']} vs {row['AwayTeam']}")
    if "Date" in row:
        print(f"Date: {row['Date']}")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Predict match probabilities from the GB models.")
    parser.add_argument("--row-index", type=int, default=-1, help="Row index from matches_features.csv. Defaults to -1.")
    return parser.parse_args()


def main() -> None:
    """Run a prediction from the latest features CSV row."""
    args = parse_args()
    predict_from_latest_features_csv(row_index=args.row_index)


if __name__ == "__main__":
    main()
