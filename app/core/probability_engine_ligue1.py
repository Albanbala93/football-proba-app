"""Ligue 1 specialist probability engine.

This engine mirrors the public contract of ``probability_engine.py`` but uses
only the specialist models trained on Ligue 1 data.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "matches_features.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "ligue1_specialist_models"
REPORT_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_specialist_models_report.csv"

MODEL_VERSION = "ligue1_specialist_v1"
DRAW_PROBABILITY_THRESHOLD = 0.28
DRAW_TOP_TWO_MARGIN_THRESHOLD = 0.08

MODEL_NAMES = {
    "home_win": "home_win_ligue1_model",
    "draw": "draw_ligue1_model",
    "away_win": "away_win_ligue1_model",
}
DEFAULT_MODEL_PATHS = {
    key: MODELS_DIR / f"{model_name}_classic.pkl" for key, model_name in MODEL_NAMES.items()
}


def _round_probability(value: float) -> float:
    """Round probabilities for JSON-compatible output."""
    return round(float(value), 6)


def _model_path_from_report(model_name: str) -> Path | None:
    """Return the best reported model path for one specialist model when available."""
    if not REPORT_PATH.exists():
        return None

    report = pd.read_csv(REPORT_PATH)
    required_columns = {"model", "variant", "brier_score", "log_loss", "roc_auc"}
    if not required_columns.issubset(report.columns):
        return None

    rows = report[report["model"].astype(str) == model_name].copy()
    if rows.empty:
        return None

    rows["brier_score"] = pd.to_numeric(rows["brier_score"], errors="coerce")
    rows["log_loss"] = pd.to_numeric(rows["log_loss"], errors="coerce")
    rows["roc_auc"] = pd.to_numeric(rows["roc_auc"], errors="coerce")
    rows = rows.sort_values(["brier_score", "log_loss", "roc_auc"], ascending=[True, True, False])

    variant = str(rows.iloc[0]["variant"])
    path = MODELS_DIR / f"{model_name}_{variant}.pkl"
    return path if path.exists() else None


def _resolve_model_path(key: str) -> Path:
    """Resolve the best available model path, falling back to classic."""
    model_name = MODEL_NAMES[key]
    report_path = _model_path_from_report(model_name)
    if report_path is not None:
        return report_path

    classic_path = DEFAULT_MODEL_PATHS[key]
    if classic_path.exists():
        return classic_path

    raise FileNotFoundError(
        f"Missing Ligue 1 specialist model for {key}: {classic_path}. "
        "Run `python app/core/ligue1_specialist_models.py` first."
    )


@lru_cache(maxsize=1)
def load_ligue1_models() -> dict[str, Any]:
    """Load Ligue 1 specialist models."""
    return {key: joblib.load(_resolve_model_path(key)) for key in MODEL_NAMES}


@lru_cache(maxsize=1)
def loaded_model_paths() -> dict[str, str]:
    """Return resolved model paths for diagnostics."""
    return {key: str(_resolve_model_path(key)) for key in MODEL_NAMES}


def _feature_columns_for_model(model: Any) -> list[str] | None:
    """Return model feature names when scikit-learn exposes them."""
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
        if features.isna().any().any():
            missing_values = features.columns[features.isna().any()].tolist()
            raise ValueError(f"Invalid or missing numeric values for {model_name}: {', '.join(missing_values)}")

    probabilities = model.predict_proba(features)
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    else:
        classes = list(model.named_steps["classifier"].classes_)
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
    top_two_margin: float,
) -> str:
    """Apply the prudent Ligue 1 draw adjustment rule."""
    argmax_class = _predicted_class_argmax(home_win_probability, draw_probability, away_win_probability)
    if draw_probability >= DRAW_PROBABILITY_THRESHOLD and top_two_margin <= DRAW_TOP_TWO_MARGIN_THRESHOLD:
        return "D"
    return argmax_class


def _favorite(probabilities: dict[str, float]) -> tuple[str, float]:
    """Return favorite side and probability from normalized match probabilities."""
    home_win = probabilities["home_win"]
    away_win = probabilities["away_win"]
    if home_win == away_win:
        return "none", home_win
    if home_win > away_win:
        return "home", home_win
    return "away", away_win


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
    """Compute a bounded 0-100 confidence score from probability shape."""
    score = (favorite_probability * 70) + (top_two_margin * 80) - (uncertainty_score * 35)
    return round(max(0.0, min(100.0, score)), 2)


def _analysis_fields(
    home_win_probability: float,
    draw_probability: float,
    away_win_probability: float,
) -> dict[str, Any]:
    """Build the analysis section compatible with existing engines."""
    sorted_probabilities = sorted([home_win_probability, draw_probability, away_win_probability], reverse=True)
    top_two_margin = sorted_probabilities[0] - sorted_probabilities[1]
    uncertainty_score = 1 - sorted_probabilities[0]
    normalized = {
        "home_win": home_win_probability,
        "draw": draw_probability,
        "away_win": away_win_probability,
    }
    favorite_team, favorite_probability = _favorite(normalized)
    predicted_class_argmax = _predicted_class_argmax(
        home_win_probability,
        draw_probability,
        away_win_probability,
    )
    predicted_class_adjusted = _predicted_class_adjusted(
        home_win_probability,
        draw_probability,
        away_win_probability,
        top_two_margin,
    )
    recommended_prediction_class = predicted_class_adjusted
    draw_warning = predicted_class_adjusted == "D" and predicted_class_argmax != "D"
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

    return {
        "model_version": MODEL_VERSION,
        "main_probability_source": "ligue1_specialist_models",
        "draw_signal_source": "ligue1_prudent_draw_rule",
        "calibrated_draw_signal": None,
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
        "is_draw_plausible": bool(draw_probability >= 0.27),
        "is_strong_draw_signal": bool(draw_probability >= 0.30),
        "is_very_strong_draw_signal": bool(draw_probability >= 0.32),
        "draw_probability_threshold": DRAW_PROBABILITY_THRESHOLD,
        "draw_top_two_margin_threshold": DRAW_TOP_TWO_MARGIN_THRESHOLD,
        "selected_model_paths": loaded_model_paths(),
    }


def predict_match_probabilities_ligue1(match_features: dict[str, Any]) -> dict[str, Any]:
    """Predict normalized Ligue 1 match probabilities from one feature dictionary."""
    if not match_features:
        raise ValueError("match_features cannot be empty.")

    features = pd.DataFrame([match_features])
    models = load_ligue1_models()

    raw_home_win = _predict_positive_probability(models["home_win"], features, "home_win_ligue1_model")
    raw_draw = _predict_positive_probability(models["draw"], features, "draw_ligue1_model")
    raw_away_win = _predict_positive_probability(models["away_win"], features, "away_win_ligue1_model")

    normalized = _normalize_probabilities(raw_home_win, raw_draw, raw_away_win)
    home_win_probability = normalized["home_win"]
    draw_probability = normalized["draw"]
    away_win_probability = normalized["away_win"]
    analysis = _analysis_fields(
        home_win_probability=home_win_probability,
        draw_probability=draw_probability,
        away_win_probability=away_win_probability,
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


def predict_from_latest_ligue1_features() -> dict[str, Any]:
    """Predict probabilities from the latest Ligue 1 row in matches_features.csv."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURES_PATH}")

    features = pd.read_csv(FEATURES_PATH)
    if "league" not in features.columns:
        raise ValueError("Feature file has no league column.")

    ligue1 = features[features["league"].astype(str) == "Ligue 1"].copy()
    if ligue1.empty:
        raise ValueError("No Ligue 1 rows found in matches_features.csv.")

    if "Date" in ligue1.columns:
        ligue1["Date"] = pd.to_datetime(ligue1["Date"], errors="coerce")
        ligue1 = ligue1.sort_values("Date").reset_index(drop=True)

    row = ligue1.iloc[-1]
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
    match_features = {key: value for key, value in row.to_dict().items() if key not in ignored_columns}
    result = predict_match_probabilities_ligue1(match_features)

    print(f"Using latest Ligue 1 row from: {FEATURES_PATH}")
    if "HomeTeam" in row and "AwayTeam" in row:
        print(f"Match: {row['HomeTeam']} vs {row['AwayTeam']}")
    if "Date" in row:
        print(f"Date: {row['Date']}")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Ligue 1 engine smoke testing."""
    return argparse.ArgumentParser(description="Predict probabilities from the latest Ligue 1 feature row.").parse_args()


def main() -> None:
    """Run a smoke prediction from the latest Ligue 1 feature row."""
    parse_args()
    predict_from_latest_ligue1_features()


if __name__ == "__main__":
    main()
