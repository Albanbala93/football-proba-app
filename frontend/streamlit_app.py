"""Streamlit interface for testing the football probability engine."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from datetime import date, datetime
import html
import re
import unicodedata

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.probability_engine import predict_match_probabilities  # noqa: E402
from app.core.probability_engine_gb import predict_match_probabilities_gb  # noqa: E402
from app.core.probability_engine_ligue1_api_xg import FEATURE_COLUMNS as LIGUE1_API_XG_FEATURE_COLUMNS  # noqa: E402
from app.core.probability_engine_ligue1_api_xg import predict_ligue1_api_xg  # noqa: E402
from app.core.probability_engine_ligue1_api_xg_v2 import FEATURE_COLUMNS as LIGUE1_API_XG_V2_FEATURE_COLUMNS  # noqa: E402
from app.core.probability_engine_ligue1_api_xg_v2 import predict_ligue1_api_xg_v2  # noqa: E402
from app.core.upcoming_match_predictor import predict_upcoming_match  # noqa: E402
from app.core.upcoming_match_predictor import build_upcoming_match_features  # noqa: E402


FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "matches_features.csv"
CURRENT_TEAMS_PATH = PROJECT_ROOT / "data" / "reference" / "current_teams.csv"
UPCOMING_FIXTURES_PATH = PROJECT_ROOT / "data" / "reference" / "upcoming_fixtures.csv"
LIGUE1_API_XG_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "api_football_ligue1_2025_prematch_features.csv"
LIGUE1_API_XG_FORMATION_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "api_football_ligue1_2025_formation_performance_features.csv"
LIGUE1_API_XG_INJURY_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "api_football_ligue1_2025_injury_impact_features.csv"
LIGUE1_API_XG_MATCH_STAKES_PATH = PROJECT_ROOT / "data" / "processed" / "api_football_ligue1_2025_match_stakes_features.csv"
MATCH_PROFILES_ANALYSIS_PATH = PROJECT_ROOT / "data" / "predictions" / "match_profiles_analysis.csv"
UPCOMING_PREDICTIONS_LOG_PATH = PROJECT_ROOT / "data" / "predictions" / "upcoming_predictions_log.csv"
UPCOMING_PREDICTIONS_EVALUATED_PATH = PROJECT_ROOT / "data" / "predictions" / "upcoming_predictions_evaluated.csv"
LIGUE1_API_XG_LIVE_TRACKING_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_api_xg_live_tracking.csv"
RESULT_COLUMNS = ["result", "FTR", "target"]
DATE_COLUMNS = ["Date", "date", "match_date"]
HOME_TEAM_COLUMNS = ["HomeTeam", "home_team"]
AWAY_TEAM_COLUMNS = ["AwayTeam", "away_team"]
UPCOMING_LOG_COLUMNS = [
    "prediction_created_at",
    "match_date",
    "league",
    "home_team",
    "away_team",
    "mode",
    "odds_source",
    "odds_home",
    "odds_draw",
    "odds_away",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "home_no_loss_probability",
    "away_no_loss_probability",
    "favorite_team",
    "favorite_probability",
    "match_profile",
    "confidence_score",
    "is_draw_plausible",
    "is_strong_draw_signal",
    "is_very_strong_draw_signal",
    "calibrated_draw_signal",
    "model_version",
]
EVALUATION_TABLE_COLUMNS = [
    "match_date",
    "league",
    "home_team",
    "away_team",
    "mode",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "predicted_class",
    "recommended_prediction_class",
    "actual_result",
    "is_correct",
    "is_correct_recommended",
    "evaluation_status",
    "match_profile",
    "confidence_score",
]
LIGUE1_API_XG_LIVE_TRACKING_COLUMNS = [
    "prediction_datetime",
    "match_date",
    "league",
    "home_team",
    "away_team",
    "model_version",
    "predicted_class",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "confidence_score",
    "top_probability",
    "top_two_margin",
    "readability_level",
    "recommendation_status",
    "reliability_score",
    "actual_result",
    "is_correct",
    "engine_label",
]

UI_STYLE = """
<style>
    .app-header {
        padding: 18px 20px;
        margin-bottom: 18px;
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .app-header h1 {
        margin: 0 0 6px 0;
        font-size: 1.8rem;
        font-weight: 800;
        color: #111827;
    }
    .app-header p {
        margin: 0;
        color: #6B7280;
        font-size: 0.98rem;
    }
    .section-card, .result-card, .prob-card, .readability-card {
        padding: 18px;
        margin-bottom: 16px;
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .result-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
    }
    .prob-card {
        background: #FFFFFF;
    }
    .readability-card {
        background: #FFFFFF;
    }
    .status-good {
        background: #ECFDF5 !important;
        border-color: #A7F3D0 !important;
    }
    .status-medium {
        background: #FFFBEB !important;
        border-color: #FDE68A !important;
    }
    .status-bad {
        background: #FEF2F2 !important;
        border-color: #FECACA !important;
    }
    .muted-text {
        color: #6B7280;
        font-size: 0.92rem;
    }
    .metric-label {
        color: #6B7280;
        font-size: 0.86rem;
        text-transform: uppercase;
        letter-spacing: 0.02em;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #111827;
        font-size: 1.2rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .card-title {
        margin: 0 0 10px 0;
        font-size: 1.02rem;
        font-weight: 800;
        color: #111827;
    }
    .card-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-left: 6px;
        vertical-align: middle;
        background: #EEF2FF;
        color: #3730A3;
    }
    .card-badge-good {
        background: #DCFCE7;
        color: #166534;
    }
    .card-badge-medium {
        background: #FEF3C7;
        color: #92400E;
    }
    .card-badge-bad {
        background: #FEE2E2;
        color: #991B1B;
    }
    ul.fp-bullets {
        margin: 0.35rem 0 0 1.2rem;
        padding: 0;
    }
    ul.fp-bullets li {
        margin-bottom: 0.25rem;
    }
</style>
"""
QUICK_READINGS = {
    "clear_home_advantage": "Avantage domicile clair. Historiquement, ce profil fait partie des signaux les plus fiables du modele.",
    "clear_away_advantage": "Avantage exterieur clair. Historiquement, ce profil fait partie des signaux les plus fiables du modele.",
    "moderate_advantage": "Avantage modere. Le signal existe, mais l'historique montre une fiabilite limitee.",
    "balanced_match": "Match tres equilibre. Aucune issue ne domine clairement : prudence.",
    "draw_plausible": "Nul plausible. Le signal existe, mais l'echantillon historique reste limite.",
    "strong_draw_signal": "Signal nul renforce. A surveiller, mais le volume historique reste faible.",
    "very_strong_draw_signal": "Signal nul tres fort. Attention : tres faible volume historique, a interpreter avec prudence.",
    "high_uncertainty": "Match a forte incertitude. Le modele signale une zone moins fiable.",
}
SIGNAL_LEVELS = {
    "clear_home_advantage": "Signal fort",
    "clear_away_advantage": "Signal fort",
    "moderate_advantage": "Signal faible",
    "balanced_match": "Prudence",
    "high_uncertainty": "Prudence",
    "draw_plausible": "Signal nul a surveiller",
    "strong_draw_signal": "Signal nul a surveiller",
    "very_strong_draw_signal": "Signal nul a surveiller",
}
DRAW_PLAUSIBLE_FALLBACK_MATCHES = 98
DRAW_PLAUSIBLE_FALLBACK_RATE = 0.2755
DRAW_ADJUSTMENT_THRESHOLD = 0.27
DRAW_ADJUSTMENT_MAX_GAP = 0.08
DRAW_WARNING_BACKTEST_RATE = 0.34
ADJUSTED_RECOMMENDED_LEAGUES = {"Bundesliga", "Serie A"}
PSEUDO_ODDS_WARNING = "Les pseudo-cotes sont dérivées de l’Elo et ne remplacent pas des cotes réelles."
HISTORICAL_ADVANCED_REQUIRED_COLUMNS = [
    "home_shots_for_5",
    "home_shots_against_5",
    "home_shots_on_target_for_5",
    "home_shots_on_target_against_5",
    "home_corners_for_5",
    "home_corners_against_5",
    "home_fouls_for_5",
    "home_fouls_against_5",
    "home_yellow_cards_5",
    "home_red_cards_5",
    "away_shots_for_5",
    "away_shots_against_5",
    "away_shots_on_target_for_5",
    "away_shots_on_target_against_5",
    "away_corners_for_5",
    "away_corners_against_5",
    "away_fouls_for_5",
    "away_fouls_against_5",
    "away_yellow_cards_5",
    "away_red_cards_5",
]
GB_MODEL_PATHS = [
    PROJECT_ROOT / "models" / "gradient_boosting_models" / "home_win_gb_model.pkl",
    PROJECT_ROOT / "models" / "gradient_boosting_models" / "away_win_gb_model.pkl",
    PROJECT_ROOT / "models" / "gradient_boosting_models" / "draw_gb_model.pkl",
]
LIGUE1_API_XG_TEAM_ALIASES = {
    "paris sg": "paris saint germain",
    "psg": "paris saint germain",
    "paris saint germain": "paris saint germain",
    "paris saint-germain": "paris saint germain",
    "paris saint germain fc": "paris saint germain",
    "olympique marseille": "marseille",
    "marseille": "marseille",
    "stade brestois 29": "stade brestois 29",
    "stade brestois": "stade brestois 29",
    "brest": "stade brestois 29",
    "le havre": "le havre",
    "lehavre": "le havre",
    "le havre ac": "le havre",
    "paris fc": "paris fc",
    "lens": "lens",
    "lyon": "lyon",
    "monaco": "monaco",
    "nice": "nice",
    "toulouse": "toulouse",
    "nantes": "nantes",
    "strasbourg": "strasbourg",
    "montpellier": "montpellier",
    "reims": "reims",
    "angers": "angers",
    "auxerre": "auxerre",
    "lorient": "lorient",
    "metz": "metz",
    "rennes": "rennes",
}


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first existing column from a candidate list."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


@st.cache_data
def load_features() -> pd.DataFrame:
    """Load match features used by the probability engine."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURES_PATH}")
    return pd.read_csv(FEATURES_PATH)


@st.cache_data
def load_current_teams() -> pd.DataFrame:
    """Load the current team reference table when available."""
    if not CURRENT_TEAMS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CURRENT_TEAMS_PATH)


@st.cache_data
def load_upcoming_fixtures() -> pd.DataFrame:
    """Load the upcoming fixtures reference table when available."""
    if not UPCOMING_FIXTURES_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(UPCOMING_FIXTURES_PATH)


@st.cache_data
def load_match_profiles_analysis() -> pd.DataFrame:
    """Load historical match_profile performance when available."""
    if not MATCH_PROFILES_ANALYSIS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(MATCH_PROFILES_ANALYSIS_PATH)


@st.cache_data
def load_upcoming_predictions_evaluated() -> pd.DataFrame:
    """Load evaluated upcoming predictions when available."""
    if not UPCOMING_PREDICTIONS_EVALUATED_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(UPCOMING_PREDICTIONS_EVALUATED_PATH)


@st.cache_data
def load_ligue1_api_xg_live_tracking() -> pd.DataFrame:
    """Load live-tracked Ligue 1 API xG predictions when available."""
    if not LIGUE1_API_XG_LIVE_TRACKING_PATH.exists():
        return pd.DataFrame()
    tracking = pd.read_csv(LIGUE1_API_XG_LIVE_TRACKING_PATH)
    for column in LIGUE1_API_XG_LIVE_TRACKING_COLUMNS:
        if column not in tracking.columns:
            tracking[column] = ""
    return tracking[LIGUE1_API_XG_LIVE_TRACKING_COLUMNS]


@st.cache_data
def load_ligue1_api_xg_features() -> pd.DataFrame:
    """Load experimental Ligue 1 API xG prematch features when available."""
    if not LIGUE1_API_XG_FEATURES_PATH.exists():
        return pd.DataFrame()
    features = pd.read_csv(LIGUE1_API_XG_FEATURES_PATH)
    if "date" in features.columns:
        features["api_date_only"] = features["date"].apply(to_date_only)
    if "home_team_name" in features.columns:
        features["home_team_key"] = features["home_team_name"].map(normalize_team_name_for_matching)
        features["home_team_key_basic"] = features["home_team_name"].map(normalize_team_name_basic)
    if "away_team_name" in features.columns:
        features["away_team_key"] = features["away_team_name"].map(normalize_team_name_for_matching)
        features["away_team_key_basic"] = features["away_team_name"].map(normalize_team_name_basic)
    return features


def _prepare_ligue1_api_xg_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Normalize common columns used by the Ligue 1 API-Football engines."""
    if features.empty:
        return features
    prepared = features.copy()
    if "date" in prepared.columns:
        prepared["api_date_only"] = prepared["date"].apply(to_date_only)
    if "home_team_name" in prepared.columns:
        prepared["home_team_key"] = prepared["home_team_name"].map(normalize_team_name_for_matching)
        prepared["home_team_key_basic"] = prepared["home_team_name"].map(normalize_team_name_basic)
    if "away_team_name" in prepared.columns:
        prepared["away_team_key"] = prepared["away_team_name"].map(normalize_team_name_for_matching)
        prepared["away_team_key_basic"] = prepared["away_team_name"].map(normalize_team_name_basic)
    return prepared


def _prepare_ligue1_fixture_reference_frame(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Normalize fixture reference columns for Ligue 1 selection."""
    if fixtures.empty:
        return fixtures
    prepared = fixtures.copy()
    if "home_team_name" not in prepared.columns and "home_team" in prepared.columns:
        prepared["home_team_name"] = prepared["home_team"]
    if "away_team_name" not in prepared.columns and "away_team" in prepared.columns:
        prepared["away_team_name"] = prepared["away_team"]
    return _prepare_ligue1_api_xg_feature_frame(prepared)


@st.cache_data
def load_ligue1_api_xg_v2_features() -> tuple[pd.DataFrame, str | None]:
    """Load the Ligue 1 API xG v2 feature frame with preferred and fallback sources."""
    source_paths = [
        LIGUE1_API_XG_INJURY_FEATURES_PATH,
        LIGUE1_API_XG_FORMATION_FEATURES_PATH,
        LIGUE1_API_XG_FEATURES_PATH,
    ]
    for source_path in source_paths:
        if source_path.exists():
            features = pd.read_csv(source_path)
            return _prepare_ligue1_api_xg_feature_frame(features), str(source_path)
    return pd.DataFrame(), None


@st.cache_data
def load_ligue1_match_stakes_features() -> pd.DataFrame:
    """Load Ligue 1 match stakes features when available."""
    if not LIGUE1_API_XG_MATCH_STAKES_PATH.exists():
        return pd.DataFrame()
    features = pd.read_csv(LIGUE1_API_XG_MATCH_STAKES_PATH)
    if "date" in features.columns:
        features["api_date_only"] = features["date"].apply(to_date_only)
    if "home_team_name" in features.columns:
        features["home_team_key"] = features["home_team_name"].map(normalize_team_name_for_matching)
        features["home_team_key_basic"] = features["home_team_name"].map(normalize_team_name_basic)
    if "away_team_name" in features.columns:
        features["away_team_key"] = features["away_team_name"].map(normalize_team_name_for_matching)
        features["away_team_key_basic"] = features["away_team_name"].map(normalize_team_name_basic)
    return features


def build_match_label(row: pd.Series, date_column: str | None, home_column: str | None, away_column: str | None) -> str:
    """Build a readable selectbox label."""
    date_value = row[date_column] if date_column else "unknown date"
    home_team = row[home_column] if home_column else "Home"
    away_team = row[away_column] if away_column else "Away"
    league_part = f" - {row['league']}" if "league" in row.index and pd.notna(row["league"]) else ""
    return f"{date_value}{league_part} - {home_team} vs {away_team}"


def to_date_only(value: Any) -> str | None:
    """Convert a datetime-like value to YYYY-MM-DD without timezone."""
    parsed_value = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed_value):
        return None
    return parsed_value.strftime("%Y-%m-%d")


def strip_accents(value: str) -> str:
    """Remove accents from a string."""
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(character for character in normalized if not unicodedata.combining(character))


def normalize_team_name_basic(value: Any) -> str:
    """Normalize a team name without applying manual aliases."""
    text = str(value or "").lower().strip()
    text = strip_accents(text)
    text = text.replace("&", " and ")
    text = re.sub(r"['’`-]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(st|st\.)\b", "saint", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_team_name(value: Any) -> str:
    """Normalize a team name with a small manual alias map."""
    basic = normalize_team_name_basic(value)
    return LIGUE1_API_XG_TEAM_ALIASES.get(basic, basic)


def normalize_team_name_for_matching(value: Any) -> str:
    """Backward-compatible alias for normalized team matching."""
    return normalize_team_name(value)


def predicted_class(match_probabilities: dict[str, float]) -> str:
    """Return H, D, or A from normalized match probabilities."""
    probabilities_by_class = prediction_probabilities_by_class(match_probabilities)
    return max(probabilities_by_class, key=probabilities_by_class.get)


def prediction_probabilities_by_class(match_probabilities: dict[str, float]) -> dict[str, float]:
    """Convert engine probability keys to H/D/A class keys."""
    return {
        "H": match_probabilities["home_win"],
        "D": match_probabilities["draw"],
        "A": match_probabilities["away_win"],
    }


def adjusted_prediction_class(match_probabilities: dict[str, float], argmax_class: str) -> str:
    """Apply the same draw adjustment rule used by the backtest."""
    probabilities_by_class = prediction_probabilities_by_class(match_probabilities)
    max_probability = max(probabilities_by_class.values())
    draw_probability = probabilities_by_class["D"]
    if draw_probability >= DRAW_ADJUSTMENT_THRESHOLD and (max_probability - draw_probability) <= DRAW_ADJUSTMENT_MAX_GAP:
        return "D"
    return argmax_class


def recommended_prediction_class(league: Any, argmax_class: str, adjusted_class: str) -> str:
    """Select the recommended class according to the league backtest policy."""
    if isinstance(league, str) and league in ADJUSTED_RECOMMENDED_LEAGUES:
        return adjusted_class
    return argmax_class


def prediction_classes(match_probabilities: dict[str, float], league: Any) -> dict[str, Any]:
    """Return argmax, adjusted, recommended, and draw warning values."""
    argmax_class = predicted_class(match_probabilities)
    adjusted_class = adjusted_prediction_class(match_probabilities, argmax_class)
    recommended_class = recommended_prediction_class(league, argmax_class, adjusted_class)
    return {
        "predicted_class_argmax": argmax_class,
        "predicted_class_adjusted": adjusted_class,
        "recommended_prediction_class": recommended_class,
        "draw_warning": adjusted_class == "D" and recommended_class != "D",
    }


def profile_sentence(profile: str, home_team: str, away_team: str) -> str:
    """Generate a short interpretation sentence from match_profile."""
    sentences = {
        "clear_home_advantage": f"{home_team} ressort avec un avantage net dans les probabilites du modele.",
        "clear_away_advantage": f"{away_team} ressort avec un avantage net malgre le deplacement.",
        "balanced_match": "Le match est equilibre: les deux probabilites principales sont tres proches.",
        "draw_plausible": "Le nul est une issue plausible et proche des meilleures options du modele.",
        "strong_draw_signal": "Le signal de nul est marque, mais il doit rester confronte aux probabilites de victoire.",
        "very_strong_draw_signal": "Le signal de nul est tres marque; verifier la fiabilite historique avant interpretation.",
        "high_uncertainty": "Le modele signale une forte incertitude: aucune issue ne se detache clairement.",
        "moderate_advantage": "Une equipe se detache legerement, mais l'avantage reste modere.",
    }
    return sentences.get(profile, "Le profil du match reste difficile a interpreter automatiquement.")


def probability_metric(label: str, value: float) -> None:
    """Display one probability as a percentage metric."""
    st.metric(label, f"{value * 100:.1f}%")


def percent(value: float) -> str:
    """Format a ratio as a percentage."""
    return f"{value * 100:.1f}%"


def inject_ui_styles() -> None:
    """Inject a light Streamlit style layer."""
    st.markdown(UI_STYLE, unsafe_allow_html=True)


def render_html_card(title: str, content: str, css_class: str = "section-card", badge: str | None = None) -> None:
    """Render a visible HTML card."""
    safe_title = html.escape(str(title))
    safe_badge = f'<span class="card-badge">{html.escape(str(badge))}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="card-title">{safe_title}{safe_badge}</div>
            <div>{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_card(title: str, body: Any | None = None, icon: str | None = None) -> None:
    """Render a lightweight visual block kept for backward compatibility."""
    title_prefix = f"{icon} " if icon else ""
    try:
        container = st.container(border=True)
    except TypeError:
        container = st.container()
    with container:
        st.markdown(f"**{title_prefix}{title}**")
        if callable(body):
            body()
        elif body is not None:
            st.markdown(body, unsafe_allow_html=True)


def render_probability_cards(home_prob: float, draw_prob: float, away_prob: float, predicted_class: str) -> None:
    """Render three visible probability cards side by side."""
    probabilities = {
        "Domicile": float(home_prob),
        "Nul": float(draw_prob),
        "Extérieur": float(away_prob),
    }
    top_label = max(probabilities, key=probabilities.get)
    predicted_lookup = {"H": "Domicile", "D": "Nul", "A": "Extérieur"}
    top_label = predicted_lookup.get(predicted_class, top_label)
    badge_html = '<span class="card-badge card-badge-good">Issue la plus probable</span>'
    cols = st.columns(3)
    for idx, (label, value) in enumerate(probabilities.items()):
        css_class = "prob-card"
        badge = badge_html if label == top_label else None
        content = f'<div class="metric-label">{html.escape(label)}</div><div class="metric-value">{value * 100:.1f}%</div>'
        if badge:
            content += f'<div style="margin-top:10px;">{badge}</div>'
        with cols[idx]:
            render_html_card("Probabilité", content, css_class=css_class)


def parse_optional_odd(value: str) -> float | None:
    """Parse an optional odd entered in the UI."""
    normalized_value = value.strip().replace(",", ".")
    if not normalized_value:
        return None
    try:
        return float(normalized_value)
    except ValueError:
        return None


def favorite_label(favorite_team: str, home_team: str, away_team: str) -> str:
    """Return a readable favorite label from the engine favorite value."""
    labels = {
        "home": home_team,
        "away": away_team,
        "draw": "Nul",
    }
    return labels.get(str(favorite_team), str(favorite_team))


def all_gb_models_available() -> bool:
    """Return whether the GB models exist on disk."""
    return all(path.exists() for path in GB_MODEL_PATHS)


def derived_profile_and_confidence(match_probabilities: dict[str, float]) -> tuple[str, float]:
    """Derive profile and confidence when an engine does not provide them."""
    home_win = float(match_probabilities["home_win"])
    draw = float(match_probabilities["draw"])
    away_win = float(match_probabilities["away_win"])
    sorted_probabilities = sorted([home_win, draw, away_win], reverse=True)
    top_two_margin = sorted_probabilities[0] - sorted_probabilities[1]
    uncertainty_score = 1 - sorted_probabilities[0]

    if home_win >= 0.55:
        profile = "clear_home_advantage"
    elif away_win >= 0.55:
        profile = "clear_away_advantage"
    elif draw >= 0.32:
        profile = "very_strong_draw_signal"
    elif draw >= 0.30:
        profile = "strong_draw_signal"
    elif top_two_margin <= 0.07:
        profile = "balanced_match"
    elif draw >= 0.27 and top_two_margin <= 0.10:
        profile = "draw_plausible"
    elif uncertainty_score >= 0.62:
        profile = "high_uncertainty"
    else:
        profile = "moderate_advantage"

    favorite_probability = max(home_win, away_win)
    confidence_score = (favorite_probability * 70) + (top_two_margin * 80) - (uncertainty_score * 35)
    return profile, round(max(0.0, min(100.0, confidence_score)), 2)


def engine_comparison_row(engine_name: str, prediction_payload: dict[str, Any], league: Any) -> dict[str, Any]:
    """Normalize one engine payload into the comparison table."""
    if "prediction" in prediction_payload:
        engine_prediction = prediction_payload["prediction"]
        analysis = engine_prediction.get("analysis", {})
        match_probabilities = engine_prediction["probabilities"]["match"]
    else:
        analysis = prediction_payload.get("analysis", {})
        match_probabilities = prediction_payload["probabilities"]["match"]

    probabilities_by_class = prediction_probabilities_by_class(match_probabilities)
    argmax_class = max(probabilities_by_class, key=probabilities_by_class.get)
    adjusted_class = adjusted_prediction_class(match_probabilities, argmax_class)
    recommended_class = recommended_prediction_class(league, argmax_class, adjusted_class)

    profile = analysis.get("match_profile")
    confidence_score = analysis.get("confidence_score")
    if profile is None or confidence_score is None:
        profile, confidence_score = derived_profile_and_confidence(match_probabilities)

    return {
        "engine_name": engine_name,
        "home_win_probability": float(match_probabilities["home_win"]),
        "draw_probability": float(match_probabilities["draw"]),
        "away_win_probability": float(match_probabilities["away_win"]),
        "predicted_class_argmax": argmax_class,
        "recommended_prediction_class": recommended_class,
        "draw_warning": adjusted_class == "D" and recommended_class != "D",
        "match_profile": profile,
        "confidence_score": confidence_score,
        "is_draw_plausible": analysis.get("is_draw_plausible", profile in {"draw_plausible", "strong_draw_signal", "very_strong_draw_signal"}),
        "model_version": analysis.get("model_version", prediction_payload.get("model_version", "not_available")),
    }


def _fixture_id_from_row(row: pd.Series) -> Any | None:
    """Extract a fixture identifier from a row when available."""
    for column in ["fixture_id", "fixtureId", "id"]:
        if column in row.index and pd.notna(row[column]):
            value = row[column]
            try:
                if float(value).is_integer():
                    return int(value)
            except (TypeError, ValueError, OverflowError):
                pass
            return value
    return None


def _normalize_debug_frame(features: pd.DataFrame, match_date: date) -> pd.DataFrame:
    """Return a small debug frame around the target date."""
    if features.empty or "date" not in features.columns:
        return pd.DataFrame()

    debug_frame = features.copy()
    debug_frame["date_debug"] = pd.to_datetime(debug_frame["date"], errors="coerce", utc=True).dt.tz_localize(None)
    target_ts = pd.Timestamp(match_date)
    debug_frame["date_distance_days"] = (debug_frame["date_debug"].dt.normalize() - target_ts).abs().dt.days
    columns = [column for column in ["fixture_id", "date", "home_team_name", "away_team_name"] if column in debug_frame.columns]
    debug_frame = debug_frame.sort_values(["date_distance_days", "date_debug"]).head(10)
    return debug_frame[columns]


def inspect_ligue1_api_xg_match_features(
    home_team: str,
    away_team: str,
    match_date: date,
    fixture_id: Any | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Inspect the API-Football table and return match features plus debug details."""
    features = load_ligue1_api_xg_features()
    selected_date_only = to_date_only(match_date)
    debug: dict[str, Any] = {
        "file_found": bool(not features.empty),
        "row_count": int(len(features)),
        "columns": features.columns.tolist() if not features.empty else [],
        "searched_date": selected_date_only or match_date.isoformat(),
        "searched_home_team": home_team,
        "searched_away_team": away_team,
        "searched_fixture_id": fixture_id,
        "matches_found": 0,
        "candidate_method": None,
        "available_matches_same_date": pd.DataFrame(),
        "available_matches_near_date": pd.DataFrame(),
    }

    required_columns = {"api_date_only", *LIGUE1_API_XG_FEATURE_COLUMNS}
    if features.empty or not required_columns.issubset(features.columns):
        return None, debug

    search_date_key = selected_date_only
    home_basic = normalize_team_name_basic(home_team)
    away_basic = normalize_team_name_basic(away_team)
    home_normalized = normalize_team_name(home_team)
    away_normalized = normalize_team_name(away_team)

    if fixture_id is not None and "fixture_id" in features.columns:
        fixture_matches = features[features["fixture_id"].astype(str) == str(fixture_id)]
        if not fixture_matches.empty:
            debug["candidate_method"] = "fixture_id"
            debug["matches_found"] = int(len(fixture_matches))
            debug["available_matches_same_date"] = _normalize_debug_frame(features[features["api_date_only"] == search_date_key], match_date)
            debug["available_matches_near_date"] = _normalize_debug_frame(features, match_date)
            return fixture_matches.iloc[0][LIGUE1_API_XG_FEATURE_COLUMNS].to_dict(), debug

    same_date = features[features["api_date_only"] == search_date_key].copy()
    debug["available_matches_same_date"] = (
        same_date[["fixture_id", "date", "home_team_name", "away_team_name"]]
        if not same_date.empty and {"fixture_id", "date", "home_team_name", "away_team_name"}.issubset(same_date.columns)
        else pd.DataFrame()
    )
    debug["available_matches_near_date"] = _normalize_debug_frame(features, match_date)

    exact_rows = same_date[
        (same_date["home_team_name"].astype(str).str.strip().str.lower() == str(home_team).strip().lower())
        & (same_date["away_team_name"].astype(str).str.strip().str.lower() == str(away_team).strip().lower())
    ]
    if not exact_rows.empty:
        debug["candidate_method"] = "date_exact_raw"
        debug["matches_found"] = int(len(exact_rows))
        return exact_rows.iloc[0][LIGUE1_API_XG_FEATURE_COLUMNS].to_dict(), debug

    basic_rows = same_date[
        (same_date["home_team_key_basic"] == home_basic)
        & (same_date["away_team_key_basic"] == away_basic)
    ]
    if not basic_rows.empty:
        debug["candidate_method"] = "date_basic_normalization"
        debug["matches_found"] = int(len(basic_rows))
        return basic_rows.iloc[0][LIGUE1_API_XG_FEATURE_COLUMNS].to_dict(), debug

    normalized_rows = same_date[
        (same_date["home_team_key"] == home_normalized)
        & (same_date["away_team_key"] == away_normalized)
    ]
    if not normalized_rows.empty:
        debug["candidate_method"] = "date_normalized_with_aliases"
        debug["matches_found"] = int(len(normalized_rows))
        return normalized_rows.iloc[0][LIGUE1_API_XG_FEATURE_COLUMNS].to_dict(), debug

    mapped_home = normalize_team_name(home_team)
    mapped_away = normalize_team_name(away_team)
    manual_rows = same_date[
        (same_date["home_team_key_basic"] == mapped_home)
        & (same_date["away_team_key_basic"] == mapped_away)
    ]
    if not manual_rows.empty:
        debug["candidate_method"] = "date_manual_mapping"
        debug["matches_found"] = int(len(manual_rows))
        return manual_rows.iloc[0][LIGUE1_API_XG_FEATURE_COLUMNS].to_dict(), debug

    debug["candidate_method"] = "not_found"
    return None, debug


def find_ligue1_api_xg_match_features(
    home_team: str,
    away_team: str,
    match_date: date,
    fixture_id: Any | None = None,
) -> dict[str, Any] | None:
    """Return API-Football xG features for one Ligue 1 match when available."""
    match_features, _ = inspect_ligue1_api_xg_match_features(home_team, away_team, match_date, fixture_id=fixture_id)
    return match_features


def inspect_ligue1_api_xg_v2_match_features(
    home_team: str,
    away_team: str,
    match_date: date,
    fixture_id: Any | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Inspect the preferred Ligue 1 API xG v2 table and return debug details."""
    features, loaded_file = load_ligue1_api_xg_v2_features()
    selected_date_only = to_date_only(match_date)
    debug: dict[str, Any] = {
        "file_found": bool(not features.empty),
        "loaded_file": loaded_file,
        "row_count": int(len(features)),
        "columns": features.columns.tolist() if not features.empty else [],
        "searched_date": selected_date_only or match_date.isoformat(),
        "searched_home_team": home_team,
        "searched_away_team": away_team,
        "searched_fixture_id": fixture_id,
        "matches_found": 0,
        "match_found": False,
        "candidate_method": None,
        "missing_features": [],
        "available_matches_same_date": pd.DataFrame(),
        "available_matches_near_date": pd.DataFrame(),
    }

    required_columns = {"api_date_only", *LIGUE1_API_XG_V2_FEATURE_COLUMNS}
    if features.empty:
        return None, debug

    missing_columns = [column for column in required_columns if column not in features.columns]
    debug["missing_features"] = missing_columns
    if missing_columns:
        return None, debug

    search_date_key = selected_date_only
    home_basic = normalize_team_name_basic(home_team)
    away_basic = normalize_team_name_basic(away_team)
    home_normalized = normalize_team_name(home_team)
    away_normalized = normalize_team_name(away_team)

    if fixture_id is not None and "fixture_id" in features.columns:
        fixture_matches = features[features["fixture_id"].astype(str) == str(fixture_id)]
        if not fixture_matches.empty:
            debug["candidate_method"] = "fixture_id"
            debug["matches_found"] = int(len(fixture_matches))
            debug["match_found"] = True
            debug["available_matches_same_date"] = _normalize_debug_frame(features[features["api_date_only"] == search_date_key], match_date)
            debug["available_matches_near_date"] = _normalize_debug_frame(features, match_date)
            return fixture_matches.iloc[0][LIGUE1_API_XG_V2_FEATURE_COLUMNS].to_dict(), debug

    same_date = features[features["api_date_only"] == search_date_key].copy()
    debug["available_matches_same_date"] = (
        same_date[["fixture_id", "date", "home_team_name", "away_team_name"]]
        if not same_date.empty and {"fixture_id", "date", "home_team_name", "away_team_name"}.issubset(same_date.columns)
        else pd.DataFrame()
    )
    debug["available_matches_near_date"] = _normalize_debug_frame(features, match_date)

    exact_rows = same_date[
        (same_date["home_team_name"].astype(str).str.strip().str.lower() == str(home_team).strip().lower())
        & (same_date["away_team_name"].astype(str).str.strip().str.lower() == str(away_team).strip().lower())
    ]
    if not exact_rows.empty:
        debug["candidate_method"] = "date_exact_raw"
        debug["matches_found"] = int(len(exact_rows))
        debug["match_found"] = True
        return exact_rows.iloc[0][LIGUE1_API_XG_V2_FEATURE_COLUMNS].to_dict(), debug

    basic_rows = same_date[
        (same_date["home_team_key_basic"] == home_basic)
        & (same_date["away_team_key_basic"] == away_basic)
    ]
    if not basic_rows.empty:
        debug["candidate_method"] = "date_basic_normalization"
        debug["matches_found"] = int(len(basic_rows))
        debug["match_found"] = True
        return basic_rows.iloc[0][LIGUE1_API_XG_V2_FEATURE_COLUMNS].to_dict(), debug

    normalized_rows = same_date[
        (same_date["home_team_key"] == home_normalized)
        & (same_date["away_team_key"] == away_normalized)
    ]
    if not normalized_rows.empty:
        debug["candidate_method"] = "date_normalized_with_aliases"
        debug["matches_found"] = int(len(normalized_rows))
        debug["match_found"] = True
        return normalized_rows.iloc[0][LIGUE1_API_XG_V2_FEATURE_COLUMNS].to_dict(), debug

    mapped_home = normalize_team_name(home_team)
    mapped_away = normalize_team_name(away_team)
    manual_rows = same_date[
        (same_date["home_team_key_basic"] == mapped_home)
        & (same_date["away_team_key_basic"] == mapped_away)
    ]
    if not manual_rows.empty:
        debug["candidate_method"] = "date_manual_mapping"
        debug["matches_found"] = int(len(manual_rows))
        debug["match_found"] = True
        return manual_rows.iloc[0][LIGUE1_API_XG_V2_FEATURE_COLUMNS].to_dict(), debug

    debug["candidate_method"] = "not_found"
    return None, debug


def find_ligue1_api_xg_v2_match_features(
    home_team: str,
    away_team: str,
    match_date: date,
    fixture_id: Any | None = None,
) -> dict[str, Any] | None:
    """Return API-Football v2 features for one Ligue 1 match when available."""
    match_features, _ = inspect_ligue1_api_xg_v2_match_features(home_team, away_team, match_date, fixture_id=fixture_id)
    return match_features


LIGUE1_MATCH_CONTEXT_LABELS = {
    "title_race": "Course au titre",
    "europe_race": "Course a l'Europe",
    "relegation_battle": "Lutte pour le maintien",
    "mixed_stakes": "Enjeux mixtes",
    "low_stakes": "Enjeu faible",
    "standard": "Standard",
}


def find_ligue1_match_stakes_features(
    home_team: str,
    away_team: str,
    match_date: date,
    fixture_id: Any | None = None,
) -> dict[str, Any] | None:
    """Return the qualitative match stakes row when available."""
    features = load_ligue1_match_stakes_features()
    if features.empty:
        return None

    selected_date_only = to_date_only(match_date)
    required_columns = {"api_date_only", "home_team_key", "away_team_key", "home_team_key_basic", "away_team_key_basic"}
    if not required_columns.issubset(features.columns):
        return None

    search_date_key = selected_date_only
    home_basic = normalize_team_name_basic(home_team)
    away_basic = normalize_team_name_basic(away_team)
    home_normalized = normalize_team_name(home_team)
    away_normalized = normalize_team_name(away_team)

    if fixture_id is not None and "fixture_id" in features.columns:
        fixture_matches = features[features["fixture_id"].astype(str) == str(fixture_id)]
        if not fixture_matches.empty:
            return fixture_matches.iloc[0].to_dict()

    same_date = features[features["api_date_only"] == search_date_key].copy()
    exact_rows = same_date[
        (same_date["home_team_name"].astype(str).str.strip().str.lower() == str(home_team).strip().lower())
        & (same_date["away_team_name"].astype(str).str.strip().str.lower() == str(away_team).strip().lower())
    ]
    if not exact_rows.empty:
        return exact_rows.iloc[0].to_dict()

    basic_rows = same_date[
        (same_date["home_team_key_basic"] == home_basic)
        & (same_date["away_team_key_basic"] == away_basic)
    ]
    if not basic_rows.empty:
        return basic_rows.iloc[0].to_dict()

    normalized_rows = same_date[
        (same_date["home_team_key"] == home_normalized)
        & (same_date["away_team_key"] == away_normalized)
    ]
    if not normalized_rows.empty:
        return normalized_rows.iloc[0].to_dict()

    return None


def render_ligue1_match_context(
    home_team: str,
    away_team: str,
    match_date: date,
    fixture_id: Any | None = None,
) -> None:
    """Render a qualitative context block for Ligue 1 matches."""
    context_row = find_ligue1_match_stakes_features(home_team, away_team, match_date, fixture_id=fixture_id)
    with st.expander("Contexte du match", expanded=False):
        st.warning(
            "Signal contextuel affiché à titre indicatif. Non intégré au moteur de probabilité car l’ablation a "
            "dégradé log_loss et Brier."
        )
        if context_row is None:
            st.info("Aucun contexte d'enjeu disponible pour ce match.")
            return

        label = str(context_row.get("match_context_label", "standard"))
        readable_label = LIGUE1_MATCH_CONTEXT_LABELS.get(label, label)
        metrics_cols = st.columns(3)
        metrics_cols[0].metric("Contexte", readable_label)
        metrics_cols[1].metric("Motivation domicile", f"{float(context_row.get('home_total_motivation_score', 0.0)):.2f}")
        metrics_cols[2].metric("Motivation extérieur", f"{float(context_row.get('away_total_motivation_score', 0.0)):.2f}")

        details_cols = st.columns(2)
        with details_cols[0]:
            st.write(f"Rang domicile : {context_row.get('home_rank_before', 'n/a')}")
            st.write(f"Points domicile : {context_row.get('home_points_before', 'n/a')}")
            st.write(f"Pression titre domicile : {context_row.get('home_title_pressure', 'n/a')}")
            st.write(f"Pression Europe domicile : {context_row.get('home_europe_pressure', 'n/a')}")
            st.write(f"Pression maintien domicile : {context_row.get('home_relegation_pressure', 'n/a')}")
        with details_cols[1]:
            st.write(f"Rang extérieur : {context_row.get('away_rank_before', 'n/a')}")
            st.write(f"Points extérieur : {context_row.get('away_points_before', 'n/a')}")
            st.write(f"Pression titre extérieur : {context_row.get('away_title_pressure', 'n/a')}")
            st.write(f"Pression Europe extérieur : {context_row.get('away_europe_pressure', 'n/a')}")
            st.write(f"Pression maintien extérieur : {context_row.get('away_relegation_pressure', 'n/a')}")

        st.write(f"Différence de motivation : {context_row.get('motivation_diff', 'n/a')}")


def render_ligue1_composition_rotation_context(
    home_team: str,
    away_team: str,
    match_date: date,
    fixture_id: Any | None = None,
) -> None:
    """Render qualitative composition / rotation context for Ligue 1 matches."""
    features = load_ligue1_lineup_strength_features()
    if features.empty:
        with st.expander("Composition / rotation", expanded=False):
            st.info("Aucune donnée de composition / rotation disponible pour ce match.")
        return

    selected_date_only = to_date_only(match_date)
    required_columns = {"api_date_only", "home_team_key", "away_team_key", "home_team_key_basic", "away_team_key_basic"}
    if not required_columns.issubset(features.columns):
        with st.expander("Composition / rotation", expanded=False):
            st.info("Aucune donnée de composition / rotation disponible pour ce match.")
        return

    context_row = None
    if fixture_id is not None and "fixture_id" in features.columns:
        fixture_matches = features[features["fixture_id"].astype(str) == str(fixture_id)]
        if not fixture_matches.empty:
            context_row = fixture_matches.iloc[0].to_dict()
    if context_row is None:
        same_date = features[features["api_date_only"] == selected_date_only].copy()
        exact_rows = same_date[
            (same_date["home_team_name"].astype(str).str.strip().str.lower() == str(home_team).strip().lower())
            & (same_date["away_team_name"].astype(str).str.strip().str.lower() == str(away_team).strip().lower())
        ]
        if not exact_rows.empty:
            context_row = exact_rows.iloc[0].to_dict()
        else:
            basic_rows = same_date[
                (same_date["home_team_key_basic"] == normalize_team_name_basic(home_team))
                & (same_date["away_team_key_basic"] == normalize_team_name_basic(away_team))
            ]
            if not basic_rows.empty:
                context_row = basic_rows.iloc[0].to_dict()
            else:
                normalized_rows = same_date[
                    (same_date["home_team_key"] == normalize_team_name(home_team))
                    & (same_date["away_team_key"] == normalize_team_name(away_team))
                ]
                if not normalized_rows.empty:
                    context_row = normalized_rows.iloc[0].to_dict()

    with st.expander("Composition / rotation", expanded=False):
        if context_row is None:
            st.info("Compositions non disponibles pour ce match.")
            return

        st.warning(
            "Ce signal est affiché à titre contextuel. Il n’est pas intégré au moteur de probabilité car l’ablation "
            "a dégradé log_loss et Brier."
        )
        lineup_available = int(pd.to_numeric(pd.Series([context_row.get("lineup_available")]), errors="coerce").iloc[0] or 0)
        home_rotation_score = float(pd.to_numeric(pd.Series([context_row.get("home_rotation_score")]), errors="coerce").iloc[0] or 0.0)
        away_rotation_score = float(pd.to_numeric(pd.Series([context_row.get("away_rotation_score")]), errors="coerce").iloc[0] or 0.0)
        rotation_diff = float(pd.to_numeric(pd.Series([context_row.get("rotation_diff")]), errors="coerce").iloc[0] or 0.0)

        metrics_cols = st.columns(3)
        metrics_cols[0].metric("Lineup disponible", "oui" if lineup_available == 1 else "non")
        metrics_cols[1].metric("Rotation domicile", f"{home_rotation_score:.2f}")
        metrics_cols[2].metric("Rotation extérieur", f"{away_rotation_score:.2f}")

        details_cols = st.columns(2)
        with details_cols[0]:
            st.write(f"Titulaires attendus domicile : {context_row.get('home_expected_starters_count', 'n/a')}")
            st.write(f"Regular starters en XI domicile : {context_row.get('home_actual_regular_starters_in_xi', 'n/a')}")
            st.write(f"Force composition domicile : {context_row.get('home_lineup_strength_score', 'n/a')}")
        with details_cols[1]:
            st.write(f"Titulaires attendus extérieur : {context_row.get('away_expected_starters_count', 'n/a')}")
            st.write(f"Regular starters en XI extérieur : {context_row.get('away_actual_regular_starters_in_xi', 'n/a')}")
            st.write(f"Force composition extérieur : {context_row.get('away_lineup_strength_score', 'n/a')}")

        st.write(f"Rotation diff : {rotation_diff:.2f}")
        st.write(f"Lineup strength diff : {context_row.get('lineup_strength_diff', 'n/a')}")

        interpretations: list[str] = []
        if lineup_available == 0:
            interpretations.append("Compositions non disponibles pour ce match.")
        if home_rotation_score >= 3:
            interpretations.append("Rotation importante côté domicile.")
        if away_rotation_score >= 3:
            interpretations.append("Rotation importante côté extérieur.")
        if abs(rotation_diff) >= 2:
            interpretations.append("Écart notable de rotation entre les deux équipes.")

        if interpretations:
            st.markdown("**Interprétation**")
            for item in interpretations:
                st.write(f"- {item}")
        else:
            st.write("Aucune alerte de composition/rotation marquée détectée.")


def render_ligue1_api_xg_mapping_debug(debug: dict[str, Any]) -> None:
    """Render a temporary debug expander for API-Football match mapping."""
    with st.expander("Debug mapping API-Football", expanded=False):
        st.write(f"Fichier trouve : {'oui' if debug.get('file_found') else 'non'}")
        st.write(f"Nombre de lignes : {debug.get('row_count', 0)}")
        columns = debug.get("columns", [])
        st.write("Colonnes disponibles :")
        st.code(", ".join(columns) if columns else "not_available")
        st.write(f"Date recherchee : {debug.get('searched_date', 'not_available')}")
        st.write(f"Equipe domicile recherchee : {debug.get('searched_home_team', 'not_available')}")
        st.write(f"Equipe exterieure recherchee : {debug.get('searched_away_team', 'not_available')}")
        if debug.get("searched_fixture_id") is not None:
            st.write(f"Fixture id recherche : {debug.get('searched_fixture_id')}")
        st.write(f"Methode de mapping : {debug.get('candidate_method', 'not_available')}")
        st.write(f"Matchs trouves : {debug.get('matches_found', 0)}")

        same_date = debug.get("available_matches_same_date")
        if isinstance(same_date, pd.DataFrame) and not same_date.empty:
            st.write("Matchs disponibles a cette date :")
            st.dataframe(same_date, use_container_width=True)
            available_teams: list[str] = []
            if "home_team_name" in same_date.columns:
                available_teams.extend(same_date["home_team_name"].dropna().astype(str).tolist())
            if "away_team_name" in same_date.columns:
                available_teams.extend(same_date["away_team_name"].dropna().astype(str).tolist())
            available_teams = sorted({team.strip() for team in available_teams if team and team.strip()})
            st.write("Equipes disponibles a cette date :")
            st.write(", ".join(available_teams) if available_teams else "aucune")
        else:
            st.write("Matchs disponibles a cette date : aucun")

        nearby = debug.get("available_matches_near_date")
        if isinstance(nearby, pd.DataFrame) and not nearby.empty:
            st.write("10 exemples autour de la date recherchee :")
            st.dataframe(nearby, use_container_width=True)
        else:
            st.write("10 exemples autour de la date recherchee : indisponibles")


def render_ligue1_api_xg_v2_mapping_debug(debug: dict[str, Any]) -> None:
    """Render a temporary debug expander for the Ligue 1 API xG v2 table."""
    with st.expander("Debug mapping API-Football v2", expanded=False):
        st.write(f"Fichier charge : {debug.get('loaded_file') or 'not_available'}")
        st.write(f"Fichier trouve : {'oui' if debug.get('file_found') else 'non'}")
        st.write(f"Nombre de lignes : {debug.get('row_count', 0)}")
        columns = debug.get("columns", [])
        st.write("Colonnes disponibles :")
        st.code(", ".join(columns) if columns else "not_available")
        loaded_file = debug.get("loaded_file")
        missing_features = debug.get("missing_features", []) or []
        injury_features = {
            "home_injury_impact_score",
            "away_injury_impact_score",
            "injury_impact_diff",
            "home_likely_starter_injuries_count",
            "away_likely_starter_injuries_count",
            "likely_starter_injuries_diff",
        }
        if loaded_file == str(LIGUE1_API_XG_INJURY_FEATURES_PATH) and not any(feature in missing_features for feature in injury_features):
            st.info("Features utilisees : xG, tirs, stabilite de formation, blessures ponderees.")
        elif any(feature in missing_features for feature in injury_features):
            st.warning("Les blessures ponderees ne sont pas disponibles. Relancez le script d’impact des blessures.")
        missing_features = debug.get("missing_features", [])
        st.write("Colonnes manquantes :")
        st.write(", ".join(missing_features) if missing_features else "aucune")
        st.write(f"Date recherchee : {debug.get('searched_date', 'not_available')}")
        st.write(f"Equipe domicile recherchee : {debug.get('searched_home_team', 'not_available')}")
        st.write(f"Equipe exterieure recherchee : {debug.get('searched_away_team', 'not_available')}")
        if debug.get("searched_fixture_id") is not None:
            st.write(f"Fixture id recherche : {debug.get('searched_fixture_id')}")
        st.write(f"Match trouve : {'oui' if debug.get('match_found') else 'non'}")
        st.write(f"Methode de mapping : {debug.get('candidate_method', 'not_available')}")

        same_date = debug.get("available_matches_same_date")
        if isinstance(same_date, pd.DataFrame) and not same_date.empty:
            st.write("Matchs disponibles a cette date :")
            st.dataframe(same_date, use_container_width=True)
        else:
            st.write("Matchs disponibles a cette date : aucun")

        nearby = debug.get("available_matches_near_date")
        if isinstance(nearby, pd.DataFrame) and not nearby.empty:
            st.write("10 exemples autour de la date recherchee :")
            st.dataframe(nearby, use_container_width=True)
        else:
            st.write("10 exemples autour de la date recherchee : indisponibles")


def append_ligue1_api_xg_comparison_row(
    rows: list[dict[str, Any]],
    warnings: list[str],
    home_team: str,
    away_team: str,
    match_date: date,
    selected_league: str | None,
    fixture_id: Any | None = None,
) -> None:
    """Append experimental Ligue 1 API xG engine output when applicable."""
    if selected_league != "Ligue 1":
        return

    match_features = find_ligue1_api_xg_match_features(home_team, away_team, match_date, fixture_id=fixture_id)
    if match_features is None:
        warnings.append("Features API-Football indisponibles pour ce match.")
        return

    try:
        prediction = predict_ligue1_api_xg(match_features)
    except Exception as exc:
        warnings.append(f"Ligue 1 API xG experimental indisponible: {exc}")
        return

    rows.append(
        {
            "engine_name": "Ligue 1 API xG (experimental)",
            "engine_status": "experimental",
            "home_win_probability": float(prediction["home_win_probability"]),
            "draw_probability": float(prediction["draw_probability"]),
            "away_win_probability": float(prediction["away_win_probability"]),
            "predicted_class_argmax": prediction["predicted_class_argmax"],
            "recommended_prediction_class": prediction["predicted_class_argmax"],
            "draw_warning": False,
            "match_profile": "experimental_api_xg",
            "confidence_score": prediction["confidence_score"],
            "is_draw_plausible": prediction["draw_probability"] >= 0.27,
            "model_version": prediction["model_version"],
        }
    )


def append_ligue1_api_xg_v2_comparison_row(
    rows: list[dict[str, Any]],
    warnings: list[str],
    home_team: str,
    away_team: str,
    match_date: date,
    selected_league: str | None,
    fixture_id: Any | None = None,
) -> None:
    """Append experimental Ligue 1 API xG v2 engine output when applicable."""
    if selected_league != "Ligue 1":
        return

    match_features = find_ligue1_api_xg_v2_match_features(home_team, away_team, match_date, fixture_id=fixture_id)
    if match_features is None:
        warnings.append("Features API-Football indisponibles pour ce match.")
        return

    try:
        prediction = predict_ligue1_api_xg_v2(match_features)
    except Exception as exc:
        warnings.append(f"Ligue 1 API xG v2 experimental indisponible: {exc}")
        return

    rows.append(
        {
            "engine_name": "Ligue 1 API xG v2 (experimental)",
            "engine_status": "experimental",
            "home_win_probability": float(prediction["home_win_probability"]),
            "draw_probability": float(prediction["draw_probability"]),
            "away_win_probability": float(prediction["away_win_probability"]),
            "predicted_class_argmax": prediction["predicted_class_argmax"],
            "recommended_prediction_class": prediction["predicted_class_argmax"],
            "draw_warning": False,
            "match_profile": "experimental_api_xg_v2",
            "confidence_score": prediction["confidence_score"],
            "is_draw_plausible": prediction["draw_probability"] >= 0.27,
            "model_version": prediction["model_version"],
        }
    )


def build_ligue1_api_xg_v2_prediction_result(
    home_team: str,
    away_team: str,
    match_date: date,
    selected_league: str | None,
    fixture_id: Any | None = None,
) -> dict[str, Any] | None:
    """Build a synthetic prediction payload for the experimental Ligue 1 API xG v2 engine."""
    if selected_league != "Ligue 1":
        return None

    match_features = find_ligue1_api_xg_v2_match_features(home_team, away_team, match_date, fixture_id=fixture_id)
    if match_features is None:
        return None

    prediction = predict_ligue1_api_xg_v2(match_features)
    home_prob = float(prediction["home_win_probability"])
    draw_prob = float(prediction["draw_probability"])
    away_prob = float(prediction["away_win_probability"])
    favorite_probability = max(home_prob, draw_prob, away_prob)
    if favorite_probability == home_prob:
        favorite_team: str | None = home_team
    elif favorite_probability == away_prob:
        favorite_team = away_team
    else:
        favorite_team = "draw"

    probabilities = {
        "match": {
            "home_win": home_prob,
            "draw": draw_prob,
            "away_win": away_prob,
        },
        "home": {
            "win": home_prob,
            "draw": draw_prob,
            "loss": away_prob,
            "no_loss": home_prob + draw_prob,
        },
        "away": {
            "win": away_prob,
            "draw": draw_prob,
            "loss": home_prob,
            "no_loss": away_prob + draw_prob,
        },
    }
    analysis = {
        "predicted_class_argmax": prediction["predicted_class_argmax"],
        "predicted_class_adjusted": prediction["predicted_class_argmax"],
        "recommended_prediction_class": prediction["predicted_class_argmax"],
        "draw_warning": False,
        "match_profile": "experimental_api_xg_v2",
        "confidence_score": prediction["confidence_score"],
        "favorite_team": favorite_team,
        "favorite_probability": favorite_probability,
        "uncertainty_score": 1 - favorite_probability,
        "top_two_margin": favorite_probability - sorted([home_prob, draw_prob, away_prob], reverse=True)[1],
        "model_version": prediction["model_version"],
        "is_draw_plausible": draw_prob >= 0.27,
        "calibrated_draw_signal": None,
    }
    return {
        "match": {
            "home_team": home_team,
            "away_team": away_team,
            "mode": "experimental",
            "league": selected_league,
            "warning": None,
        },
        "league": selected_league,
        "odds_source": "api_football",
        "prediction": {
            "probabilities": probabilities,
            "analysis": analysis,
            "match": {
                "home_team": home_team,
                "away_team": away_team,
                "mode": "experimental",
                "league": selected_league,
                "warning": None,
            },
            "league": selected_league,
        },
        "features": match_features,
    }


def compute_match_readability(prediction: dict[str, Any], match_features: dict[str, Any] | None) -> dict[str, Any]:
    """Compute a simple readability score for the Ligue 1 API xG v2 engine."""
    probabilities = prediction.get("prediction", {}).get("probabilities", {}).get("match", {})
    analysis = prediction.get("prediction", {}).get("analysis", {})

    home_prob = float(probabilities.get("home_win", 0.0))
    draw_prob = float(probabilities.get("draw", 0.0))
    away_prob = float(probabilities.get("away_win", 0.0))
    top_probability = max(home_prob, draw_prob, away_prob)
    top_two_margin = float(top_probability - sorted([home_prob, draw_prob, away_prob], reverse=True)[1])
    confidence_score = float(analysis.get("confidence_score", 0.0))

    api_expected_goals_diff_5 = None
    home_formation_stability_5 = None
    away_formation_stability_5 = None
    formation_stability_diff_5 = None
    if isinstance(match_features, dict):
        api_expected_goals_diff_5 = pd.to_numeric(pd.Series([match_features.get("api_expected_goals_diff_5")]), errors="coerce").iloc[0]
        home_formation_stability_5 = pd.to_numeric(pd.Series([match_features.get("home_formation_stability_5")]), errors="coerce").iloc[0]
        away_formation_stability_5 = pd.to_numeric(pd.Series([match_features.get("away_formation_stability_5")]), errors="coerce").iloc[0]
        formation_stability_diff_5 = pd.to_numeric(pd.Series([match_features.get("formation_stability_diff_5")]), errors="coerce").iloc[0]

    reasons: list[str] = []
    if top_probability >= 0.50:
        reasons.append("Probabilité dominante nette")
    if top_two_margin < 0.05:
        reasons.append("Écart faible entre les issues")
    if draw_prob >= 0.32:
        reasons.append("Signal nul élevé")
    if pd.notna(api_expected_goals_diff_5) and abs(float(api_expected_goals_diff_5)) >= 0.40:
        reasons.append("Avantage xG récent net")
    if (
        pd.notna(home_formation_stability_5)
        and pd.notna(away_formation_stability_5)
        and float(home_formation_stability_5) >= 0.6
        and float(away_formation_stability_5) >= 0.6
    ):
        reasons.append("Formations stables")
    if pd.notna(formation_stability_diff_5) and abs(float(formation_stability_diff_5)) >= 0.4:
        reasons.append("Stabilité tactique déséquilibrée")

    if top_probability >= 0.50 and top_two_margin >= 0.10 and draw_prob < 0.30:
        readability_level = "forte"
        recommendation_status = "exploitable"
    elif top_probability < 0.43 or top_two_margin < 0.05 or draw_prob >= 0.32:
        readability_level = "faible"
        recommendation_status = "à éviter"
    elif top_probability >= 0.43 or top_two_margin >= 0.06:
        readability_level = "moyenne"
        recommendation_status = "prudence"
    else:
        readability_level = "faible"
        recommendation_status = "à éviter"

    return {
        "readability_level": readability_level,
        "recommendation_status": recommendation_status,
        "reasons": reasons,
        "top_probability": top_probability,
        "top_two_margin": top_two_margin,
        "draw_probability": draw_prob,
        "confidence_score": confidence_score,
        "api_expected_goals_diff_5": api_expected_goals_diff_5,
        "home_formation_stability_5": home_formation_stability_5,
        "away_formation_stability_5": away_formation_stability_5,
        "formation_stability_diff_5": formation_stability_diff_5,
    }


def _readability_bonus(readability_level: str) -> int:
    """Return the readability bonus used in the reliability score."""
    if readability_level == "forte":
        return 15
    if readability_level == "moyenne":
        return 5
    if readability_level == "faible":
        return -15
    return 0


def build_ligue1_reliable_match_selection(
    selected_count: int,
    upcoming_fixtures: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str], int]:
    """Score Ligue 1 API xG v2 candidate fixtures and keep the most reliable ones."""
    warnings: list[str] = []
    candidate_rows: list[dict[str, Any]] = []
    features_frame, source_path = load_ligue1_api_xg_v2_features()
    if features_frame.empty:
        return pd.DataFrame(), ["Aucun match Ligue 1 disponible pour la sélection."], 0

    candidate_df = features_frame.copy()
    if "league" in candidate_df.columns:
        candidate_df = candidate_df[candidate_df["league"].astype(str) == "Ligue 1"].copy()
    if candidate_df.empty:
        return pd.DataFrame(), ["Aucun match Ligue 1 disponible pour la sélection."], 0

    if upcoming_fixtures is not None and not upcoming_fixtures.empty:
        reference_fixtures = _prepare_ligue1_fixture_reference_frame(upcoming_fixtures)
        reference_fixtures = reference_fixtures[reference_fixtures.get("league", pd.Series(dtype=str)).astype(str) == "Ligue 1"].copy()
        if not reference_fixtures.empty:
            if "fixture_id" in candidate_df.columns and "fixture_id" in reference_fixtures.columns:
                merged_df = candidate_df.merge(
                    reference_fixtures[["fixture_id"]].dropna().drop_duplicates(),
                    on="fixture_id",
                    how="inner",
                )
            else:
                merge_keys = [column for column in ["api_date_only", "home_team_key", "away_team_key"] if column in candidate_df.columns and column in reference_fixtures.columns]
                merged_df = candidate_df.merge(
                    reference_fixtures[merge_keys].dropna().drop_duplicates(),
                    on=merge_keys,
                    how="inner",
                ) if merge_keys else pd.DataFrame()
            if not merged_df.empty:
                candidate_df = merged_df
            else:
                warnings.append("Aucun match à venir exact trouvé, fallback sur les matchs non joués du fichier API-Football.")

    if candidate_df.empty and "status_short" in features_frame.columns:
        non_final_statuses = {"FT", "AET", "PEN", "CANC", "ABD", "AWD"}
        candidate_df = features_frame[
            features_frame["status_short"].astype(str).str.upper().map(lambda value: value not in non_final_statuses)
        ].copy()

    if candidate_df.empty and "api_date_only" in features_frame.columns:
        today_only = date.today().isoformat()
        candidate_df = features_frame[features_frame["api_date_only"].fillna("") >= today_only].copy()

    if candidate_df.empty:
        return pd.DataFrame(), ["Aucun match Ligue 1 disponible pour générer une sélection."], 0

    if "api_date_only" in candidate_df.columns:
        candidate_df = candidate_df.sort_values(["api_date_only", "home_team_name", "away_team_name"], na_position="last")

    for _, row in candidate_df.iterrows():
        home_team = str(row.get("home_team_name", "")).strip()
        away_team = str(row.get("away_team_name", "")).strip()
        if not home_team or not away_team:
            continue

        raw_date = row.get("date")
        if pd.isna(raw_date):
            raw_date = row.get("api_date_only")
        date_only = to_date_only(raw_date)
        if date_only is None:
            continue
        try:
            match_date = datetime.strptime(date_only, "%Y-%m-%d").date()
        except ValueError:
            continue

        fixture_id = _fixture_id_from_row(row)
        prediction_result = build_ligue1_api_xg_v2_prediction_result(
            home_team,
            away_team,
            match_date,
            "Ligue 1",
            fixture_id=fixture_id,
        )
        if prediction_result is None:
            continue

        match_features = prediction_result.get("features")
        readability = compute_match_readability(prediction_result, match_features)
        probabilities = prediction_result["prediction"]["probabilities"]["match"]
        analysis = prediction_result["prediction"]["analysis"]
        top_probability = float(readability["top_probability"])
        top_two_margin = float(readability["top_two_margin"])
        draw_probability = float(readability["draw_probability"])
        confidence_score = float(readability["confidence_score"])
        bonus_lisibilite = _readability_bonus(str(readability["readability_level"]))
        penalty_draw = 10 if draw_probability >= 0.32 else 0
        penalty_low_margin = 10 if top_two_margin < 0.05 else 0
        reliability_score = round(
            (top_probability * 100)
            + (top_two_margin * 80)
            + (confidence_score * 0.3)
            + bonus_lisibilite
            - penalty_draw
            - penalty_low_margin,
            2,
        )

        candidate_rows.append(
            {
                "date": date_only,
                "home_team_name": home_team,
                "away_team_name": away_team,
                "predicted_class": analysis.get("predicted_class_argmax", ""),
                "home_win_probability": float(probabilities["home_win"]),
                "draw_probability": float(probabilities["draw"]),
                "away_win_probability": float(probabilities["away_win"]),
                "top_probability": top_probability,
                "top_two_margin": top_two_margin,
                "confidence_score": confidence_score,
                "readability_level": readability["readability_level"],
                "recommendation_status": readability["recommendation_status"],
                "reliability_score": reliability_score,
                "principales_raisons_de_lisibilite": ", ".join(readability["reasons"]) if readability["reasons"] else "Aucune raison additionnelle détectée.",
            }
        )

    if not candidate_rows:
        return pd.DataFrame(), ["Aucun match Ligue 1 disponible pour générer une sélection."], 0

    candidate_df = pd.DataFrame(candidate_rows)
    analyzed_count = len(candidate_df)
    priority_map = {"exploitable": 0, "prudence": 1, "à éviter": 2}
    candidate_df["selection_priority"] = candidate_df["recommendation_status"].map(priority_map).fillna(3).astype(int)
    candidate_df = candidate_df.sort_values(
        ["selection_priority", "reliability_score", "top_probability", "confidence_score"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    preferred_df = candidate_df[candidate_df["recommendation_status"].isin(["exploitable", "prudence"])].copy()
    selected_df = preferred_df.head(selected_count).copy()
    if len(selected_df) < selected_count:
        fallback_needed = selected_count - len(selected_df)
        fallback_df = candidate_df[candidate_df["recommendation_status"] == "à éviter"].head(fallback_needed)
        if not fallback_df.empty:
            warnings.append(
                "Pas assez de matchs exploitables ou prudence pour remplir la sélection. "
                "Quelques matchs 'à éviter' ont été inclus."
            )
        selected_df = pd.concat([selected_df, fallback_df], ignore_index=True)

    selected_df = selected_df.copy()
    selected_df.insert(0, "rang", range(1, len(selected_df) + 1))
    selected_df = selected_df.drop(columns=["selection_priority"], errors="ignore")

    if len(selected_df) < selected_count:
        warnings.append(
            f"Seulement {len(selected_df)} matchs ont pu être sélectionnés sur {selected_count} demandés."
        )

    return selected_df, warnings, analyzed_count


def build_upcoming_comparison_table(
    home_team: str,
    away_team: str,
    match_date: date,
    selected_league: str | None,
    odds_home: float | None,
    odds_draw: float | None,
    odds_away: float | None,
    fixture_id: Any | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a comparison table with no-odds, with-odds, and optional Ligue 1 API xG rows."""
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    match_date_str = match_date.isoformat()

    try:
        no_odds_result = predict_upcoming_match(
            home_team,
            away_team,
            match_date_str,
            mode="no-odds",
            league=selected_league,
        )
        rows.append(engine_comparison_row("Moteur principal sans cotes", no_odds_result, selected_league))
    except Exception as exc:
        warnings.append(f"Moteur principal sans cotes indisponible: {exc}")

    if all(value is not None and value > 1.0 for value in [odds_home, odds_draw, odds_away]):
        try:
            with_odds_result = predict_upcoming_match(
                home_team,
                away_team,
                match_date_str,
                mode="with-odds",
                odds_home=odds_home,
                odds_draw=odds_draw,
                odds_away=odds_away,
                league=selected_league,
            )
            rows.append(engine_comparison_row("Moteur principal avec cotes reelles", with_odds_result, selected_league))
        except Exception as exc:
            warnings.append(f"Moteur principal avec cotes reelles indisponible: {exc}")
    elif any(value is not None for value in [odds_home, odds_draw, odds_away]):
        warnings.append("Le moteur principal avec cotes reelles requiert trois cotes strictement superieures a 1.")

    if selected_league == "Ligue 1":
        match_features = find_ligue1_api_xg_v2_match_features(home_team, away_team, match_date, fixture_id=fixture_id)
        if match_features is None:
            warnings.append("Features API-Football indisponibles pour ce match.")
        else:
            try:
                prediction = predict_ligue1_api_xg_v2(match_features)
                rows.append(
                    {
                        "engine_name": "Ligue 1 API xG experimental",
                        "engine_status": "experimental",
                        "home_win_probability": float(prediction["home_win_probability"]),
                        "draw_probability": float(prediction["draw_probability"]),
                        "away_win_probability": float(prediction["away_win_probability"]),
                        "predicted_class_argmax": prediction["predicted_class_argmax"],
                        "recommended_prediction_class": prediction["predicted_class_argmax"],
                        "draw_warning": False,
                        "match_profile": "experimental_api_xg_v2",
                        "confidence_score": prediction["confidence_score"],
                        "is_draw_plausible": prediction["draw_probability"] >= 0.27,
                        "model_version": prediction["model_version"],
                    }
                )
            except Exception as exc:
                warnings.append(f"Ligue 1 API xG experimental indisponible: {exc}")

    return pd.DataFrame(rows), warnings


def build_engine_comparison_table(
    home_team: str,
    away_team: str,
    match_date: date,
    selected_league: str | None,
    mode: str,
    odds_home: float | None,
    odds_draw: float | None,
    odds_away: float | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the comparison table for the available engines."""
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    match_date_str = match_date.isoformat()

    if mode == "no-odds":
        warnings.append(
            "logistic_v3 indisponible en mode no-odds : ce moteur attend des cotes reelles."
        )
        try:
            no_odds_result = predict_upcoming_match(
                home_team,
                away_team,
                match_date_str,
                mode="no-odds",
                league=selected_league,
            )
            rows.append(engine_comparison_row("no_odds_elo_v1", no_odds_result, selected_league))
        except Exception as exc:
            warnings.append(f"no_odds_elo_v1 indisponible: {exc}")

        if all_gb_models_available():
            warnings.append(
                "gradient_boosting_v1 requiert des cotes reelles. "
                "Passez en with-odds pour le comparer."
            )
        else:
            warnings.append("gradient_boosting_v1 indisponible: les modeles sauvegardes sont manquants.")
        append_ligue1_api_xg_comparison_row(
            rows=rows,
            warnings=warnings,
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            selected_league=selected_league,
        )
        append_ligue1_api_xg_v2_comparison_row(
            rows=rows,
            warnings=warnings,
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            selected_league=selected_league,
        )
        return pd.DataFrame(rows), warnings

    try:
        classic_result = predict_upcoming_match(
            home_team,
            away_team,
            match_date_str,
            mode=mode,
            odds_home=odds_home,
            odds_draw=odds_draw,
            odds_away=odds_away,
            league=selected_league,
        )
        rows.append(engine_comparison_row("logistic_v3", classic_result, selected_league))
    except Exception as exc:
        warnings.append(f"logistic_v3 indisponible: {exc}")

    try:
        no_odds_result = predict_upcoming_match(
            home_team,
            away_team,
            match_date_str,
            mode="no-odds",
            league=selected_league,
        )
        rows.append(engine_comparison_row("no_odds_elo_v1", no_odds_result, selected_league))
    except Exception as exc:
        warnings.append(f"no_odds_elo_v1 indisponible: {exc}")

    if all_gb_models_available():
        try:
            match_features, resolved_league, league_warning = build_upcoming_match_features(
                home_team=home_team,
                away_team=away_team,
                match_date=match_date_str,
                odds_mode="manual" if mode == "with-odds" else "elo_pseudo",
                manual_odds={"home": odds_home, "draw": odds_draw, "away": odds_away},
                league=selected_league,
            )
            if league_warning:
                warnings.append(league_warning)
            gb_result = predict_match_probabilities_gb(match_features)
            gb_result["analysis"]["league"] = resolved_league
            rows.append(engine_comparison_row("gradient_boosting_v1", gb_result, resolved_league))
        except Exception as exc:
            warnings.append(f"gradient_boosting_v1 indisponible: {exc}")
    else:
        warnings.append("gradient_boosting_v1 indisponible: les modeles sauvegardes sont manquants.")

    append_ligue1_api_xg_comparison_row(
        rows=rows,
        warnings=warnings,
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        selected_league=selected_league,
    )
    append_ligue1_api_xg_v2_comparison_row(
        rows=rows,
        warnings=warnings,
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        selected_league=selected_league,
    )
    return pd.DataFrame(rows), warnings


def validate_historical_features(match_features: dict[str, Any]) -> list[str]:
    """Return missing advanced historical columns before calling the engine."""
    return [column for column in HISTORICAL_ADVANCED_REQUIRED_COLUMNS if column not in match_features]


def render_quick_reading(profile: str) -> None:
    """Render the compact business interpretation for a match profile."""
    st.subheader("Lecture rapide")
    signal_level = SIGNAL_LEVELS.get(profile, "Signal non qualifie")
    st.markdown(f"**Niveau de signal : {signal_level}**")
    st.info(QUICK_READINGS.get(profile, "Profil non disponible pour ce match."))


def render_upcoming_summary(
    result: dict[str, Any],
    home_team: str,
    away_team: str,
    readability: dict[str, Any] | None = None,
) -> None:
    """Render a concise summary for an upcoming match prediction."""
    probabilities = result["probabilities"]
    match_probabilities = probabilities["match"]
    analysis = result["analysis"]
    profile = str(analysis.get("match_profile", "not_available"))
    favorite_team = str(analysis.get("favorite_team", "not_available"))
    favorite_probability = analysis.get("favorite_probability")
    class_details = prediction_classes(match_probabilities, result.get("league"))
    if favorite_probability is None:
        favorite_probability = max(match_probabilities.values())
    predicted_class = class_details["predicted_class_argmax"]
    readability_level = readability.get("readability_level") if readability else "not_available"
    recommendation_status = readability.get("recommendation_status") if readability else "not_available"
    class_to_probability_key = {"H": "home_win", "D": "draw", "A": "away_win"}
    class_to_label = {"H": home_team, "D": "Nul", "A": away_team}
    dominant_probability = float(match_probabilities.get(class_to_probability_key.get(predicted_class, "home_win"), favorite_probability))
    dominant_label = class_to_label.get(predicted_class, favorite_label(favorite_team, home_team, away_team))
    confidence_score = analysis.get("confidence_score")
    result_content = f"""
        <div class="metric-label">Pronostic principal</div>
        <div class="metric-value">{html.escape(str(dominant_label))}</div>
        <div style="height:10px;"></div>
        <div class="metric-label">Probabilité dominante</div>
        <div class="metric-value">{dominant_probability * 100:.1f}%</div>
        <div style="height:10px;"></div>
        <div class="metric-label">Moteur utilisé</div>
        <div class="metric-value">{html.escape(str(analysis.get('model_version', 'not_available')))}</div>
        <div style="height:10px;"></div>
        <div class="metric-label">Confidence score</div>
        <div class="metric-value">{float(confidence_score):.1f}/100</div>
        <div style="height:10px;"></div>
        <div class="metric-label">Lisibilité</div>
        <div class="metric-value">{html.escape(str(readability_level))}</div>
        <div style="height:10px;"></div>
        <div class="metric-label">Recommandation</div>
        <div class="metric-value">{html.escape(str(recommendation_status))}</div>
    """
    render_html_card("Résultat principal", result_content, css_class="result-card")


def build_upcoming_log_row(
    prediction_result: dict[str, Any],
    odds_home: float | None,
    odds_draw: float | None,
    odds_away: float | None,
) -> dict[str, Any]:
    """Build one CSV log row from an upcoming match prediction."""
    match = prediction_result["match"]
    features = prediction_result.get("features", {})
    prediction = prediction_result["prediction"]
    probabilities = prediction["probabilities"]
    analysis = prediction["analysis"]
    favorite_probability = analysis.get("favorite_probability")
    if favorite_probability is None:
        favorite_probability = max(probabilities["match"].values())

    return {
        "prediction_created_at": datetime.now().isoformat(timespec="seconds"),
        "match_date": match["date"],
        "league": match.get("league"),
        "home_team": match["home_team"],
        "away_team": match["away_team"],
        "mode": match.get("mode"),
        "odds_source": match.get("odds_source"),
        "odds_home": features.get("B365H", odds_home),
        "odds_draw": features.get("B365D", odds_draw),
        "odds_away": features.get("B365A", odds_away),
        "home_win_probability": probabilities["match"]["home_win"],
        "draw_probability": probabilities["match"]["draw"],
        "away_win_probability": probabilities["match"]["away_win"],
        "home_no_loss_probability": probabilities["home"]["no_loss"],
        "away_no_loss_probability": probabilities["away"]["no_loss"],
        "favorite_team": analysis.get("favorite_team"),
        "favorite_probability": favorite_probability,
        "match_profile": analysis.get("match_profile"),
        "confidence_score": analysis.get("confidence_score"),
        "is_draw_plausible": analysis.get("is_draw_plausible"),
        "is_strong_draw_signal": analysis.get("is_strong_draw_signal"),
        "is_very_strong_draw_signal": analysis.get("is_very_strong_draw_signal"),
        "calibrated_draw_signal": analysis.get("calibrated_draw_signal"),
        "model_version": analysis.get("model_version"),
    }


def append_upcoming_prediction_log(row: dict[str, Any]) -> None:
    """Append one upcoming prediction to the CSV log without overwriting history."""
    UPCOMING_PREDICTIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not UPCOMING_PREDICTIONS_LOG_PATH.exists()
    pd.DataFrame([row], columns=UPCOMING_LOG_COLUMNS).to_csv(
        UPCOMING_PREDICTIONS_LOG_PATH,
        mode="a",
        header=write_header,
        index=False,
    )


def _prediction_class_from_probabilities(probabilities: dict[str, float]) -> str:
    """Return the top class from a probability dictionary."""
    return max(probabilities, key=probabilities.get)


def build_ligue1_api_xg_live_tracking_row(state: dict[str, Any]) -> dict[str, Any] | None:
    """Build one live-tracking row from the experimental Ligue 1 API xG state."""
    prediction_result = state.get("prediction_result")
    if not prediction_result:
        return None

    match = prediction_result.get("match", {})
    prediction = prediction_result.get("prediction", {})
    probabilities = prediction.get("probabilities", {}).get("match", {})
    analysis = prediction.get("analysis", {})
    if not probabilities:
        return None

    home_prob = float(probabilities.get("home_win", 0.0))
    draw_prob = float(probabilities.get("draw", 0.0))
    away_prob = float(probabilities.get("away_win", 0.0))
    top_probability = max(home_prob, draw_prob, away_prob)
    sorted_probabilities = sorted([home_prob, draw_prob, away_prob], reverse=True)
    top_two_margin = float(sorted_probabilities[0] - sorted_probabilities[1]) if len(sorted_probabilities) >= 2 else 0.0
    predicted_class = analysis.get("predicted_class_argmax") or _prediction_class_from_probabilities(
        {"H": home_prob, "D": draw_prob, "A": away_prob}
    )

    return {
        "prediction_datetime": datetime.now().isoformat(timespec="seconds"),
        "match_date": match.get("date") or state.get("match_date") or "",
        "league": match.get("league") or state.get("selected_league") or "Ligue 1",
        "home_team": match.get("home_team", ""),
        "away_team": match.get("away_team", ""),
        "model_version": analysis.get("model_version", "ligue1_api_xg_v2"),
        "predicted_class": predicted_class,
        "home_win_probability": home_prob,
        "draw_probability": draw_prob,
        "away_win_probability": away_prob,
        "confidence_score": float(analysis.get("confidence_score", 0.0)),
        "top_probability": top_probability,
        "top_two_margin": top_two_margin,
        "readability_level": state.get("readability", {}).get("readability_level", ""),
        "recommendation_status": state.get("readability", {}).get("recommendation_status", ""),
        "reliability_score": state.get("reliability_score", ""),
        "actual_result": state.get("actual_result", ""),
        "is_correct": state.get("is_correct", ""),
        "engine_label": "Ligue 1 API xG - recommandé",
    }


def build_ligue1_api_xg_live_tracking_rows_from_selection(selection_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build live-tracking rows for a reliable selection table."""
    if selection_df.empty:
        return []

    rows: list[dict[str, Any]] = []
    for _, row in selection_df.iterrows():
        rows.append(
            {
                "prediction_datetime": datetime.now().isoformat(timespec="seconds"),
                "match_date": row.get("date", ""),
                "league": "Ligue 1",
                "home_team": row.get("home_team_name", ""),
                "away_team": row.get("away_team_name", ""),
                "model_version": "ligue1_api_xg_v2",
                "predicted_class": row.get("predicted_class", ""),
                "home_win_probability": row.get("home_win_probability", ""),
                "draw_probability": row.get("draw_probability", ""),
                "away_win_probability": row.get("away_win_probability", ""),
                "confidence_score": row.get("confidence_score", ""),
                "top_probability": row.get("top_probability", ""),
                "top_two_margin": row.get("top_two_margin", ""),
                "readability_level": row.get("readability_level", ""),
                "recommendation_status": row.get("recommendation_status", ""),
                "reliability_score": row.get("reliability_score", ""),
                "actual_result": row.get("actual_result", ""),
                "is_correct": row.get("is_correct", ""),
        "engine_label": "Ligue 1 API xG - sélection fiable",
            }
        )
    return rows


def append_ligue1_api_xg_live_tracking(row: dict[str, Any]) -> bool:
    """Append a live-tracking row if it is not an exact duplicate."""
    if not row:
        return False

    LIGUE1_API_XG_LIVE_TRACKING_PATH.parent.mkdir(parents=True, exist_ok=True)
    current = load_ligue1_api_xg_live_tracking()
    dedupe_columns = ["match_date", "home_team", "away_team", "model_version"]
    if not current.empty and set(dedupe_columns).issubset(current.columns):
        duplicate_mask = pd.Series(True, index=current.index)
        for column in dedupe_columns:
            duplicate_mask &= current[column].astype(str) == str(row.get(column, ""))
        if bool(duplicate_mask.any()):
            return False

    write_header = not LIGUE1_API_XG_LIVE_TRACKING_PATH.exists()
    pd.DataFrame([row], columns=LIGUE1_API_XG_LIVE_TRACKING_COLUMNS).to_csv(
        LIGUE1_API_XG_LIVE_TRACKING_PATH,
        mode="a",
        header=write_header,
        index=False,
    )
    load_ligue1_api_xg_live_tracking.clear()
    return True


def append_ligue1_api_xg_live_tracking_rows(rows: list[dict[str, Any]]) -> int:
    """Append multiple live-tracking rows while skipping exact duplicates."""
    if not rows:
        return 0

    saved_count = 0
    for row in rows:
        if append_ligue1_api_xg_live_tracking(row):
            saved_count += 1
    return saved_count


def _evaluate_ligue1_api_xg_live_tracking(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived evaluation columns for live-tracked Ligue 1 API xG predictions."""
    if df.empty:
        return df

    evaluated = df.copy()
    if "actual_result" in evaluated.columns:
        evaluated["actual_result"] = evaluated["actual_result"].fillna("").astype(str).str.upper().str.strip()
    else:
        evaluated["actual_result"] = ""

    if "predicted_class" in evaluated.columns:
        evaluated["predicted_class"] = evaluated["predicted_class"].fillna("").astype(str).str.upper().str.strip()
    else:
        evaluated["predicted_class"] = ""

    evaluated["is_correct"] = (
        evaluated["actual_result"].ne("")
        & evaluated["predicted_class"].ne("")
        & (evaluated["actual_result"] == evaluated["predicted_class"])
    )
    return evaluated


def _confidence_bucket(value: Any) -> str:
    """Map a confidence score to the requested bucket labels."""
    score = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(score):
        return "unknown"
    if score < 45:
        return "faible"
    if score < 60:
        return "moyen"
    return "fort"


def _top_probability_bucket(value: Any) -> str:
    """Map a dominant probability to the requested bucket labels."""
    score = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(score):
        return "unknown"
    if score < 0.43:
        return "< 0.43"
    if score < 0.50:
        return "0.43-0.50"
    if score < 0.60:
        return "0.50-0.60"
    return "> 0.60"


def render_upcoming_prediction_result(
    state: dict[str, Any],
    profiles_analysis: pd.DataFrame | None = None,
    show_save_button: bool = True,
) -> None:
    """Render a stored upcoming prediction and its save action."""
    prediction_result = state["prediction_result"]
    match = prediction_result["match"]
    result = {
        **prediction_result["prediction"],
        "odds_source": match.get("odds_source", "none"),
        "league": match.get("league"),
    }
    analysis_caption = (
        "Ligue 1 API xG : moteur expérimental spécialisé Ligue 1 basé sur xG, tirs et stabilité de formation."
        if str(match.get("mode", state.get("mode"))).strip() == "experimental"
        else None
    )

    if match.get("warning"):
        st.error(match["warning"])
    render_upcoming_summary(result, match["home_team"], match["away_team"])
    render_probability_blocks(result, match["home_team"], match["away_team"])
    match_probabilities = result["probabilities"]["match"]
    class_details = prediction_classes(match_probabilities, result.get("league"))
    analysis = result["analysis"]
    readability = None
    with st.expander("Détails avancés API-Football", expanded=False):
        if str(match.get("mode", state.get("mode"))).strip() == "experimental":
            match_features = state.get("match_features") or prediction_result.get("features")
            readability = compute_match_readability(prediction_result, match_features)
            readability_class = {
                "exploitable": "status-good",
                "prudence": "status-medium",
                "à éviter": "status-bad",
            }.get(readability["recommendation_status"], "section-card")
            reasons_html = "".join(
                f"<li>{html.escape(reason)}</li>" for reason in (readability.get("reasons") or [])
            ) or "<li>Aucune raison additionnelle détectée.</li>"
            readability_content = f"""
                <div class="metric-label">Niveau</div>
                <div class="metric-value">{html.escape(str(readability['readability_level']))}</div>
                <div style="height:10px;"></div>
                <div class="metric-label">Recommandation</div>
                <div class="metric-value">{html.escape(str(readability['recommendation_status']))}</div>
                <div style="height:10px;"></div>
                <div class="metric-label">Top probability</div>
                <div class="metric-value">{readability['top_probability'] * 100:.1f}%</div>
                <div class="metric-label" style="margin-top:10px;">Top two margin</div>
                <div class="metric-value">{readability['top_two_margin']:.3f}</div>
                <div class="metric-label" style="margin-top:10px;">Confidence</div>
                <div class="metric-value">{readability['confidence_score']:.1f}/100</div>
                <div class="metric-label" style="margin-top:10px;">Raisons principales</div>
                <ul class="fp-bullets">{reasons_html}</ul>
            """
            render_html_card("Lisibilité du match", readability_content, css_class=f"readability-card {readability_class}")
            details_lines = []
            if pd.notna(readability.get("api_expected_goals_diff_5")):
                details_lines.append(f"xG récent diff : **{float(readability['api_expected_goals_diff_5']):.2f}**")
            if pd.notna(readability.get("home_formation_stability_5")) or pd.notna(readability.get("away_formation_stability_5")):
                home_stability = readability.get("home_formation_stability_5")
                away_stability = readability.get("away_formation_stability_5")
                stability_parts = []
                if pd.notna(home_stability):
                    stability_parts.append(f"home {float(home_stability):.2f}")
                if pd.notna(away_stability):
                    stability_parts.append(f"away {float(away_stability):.2f}")
                if pd.notna(readability.get("formation_stability_diff_5")):
                    stability_parts.append(f"diff {float(readability['formation_stability_diff_5']):.2f}")
                details_lines.append("Stabilité formation : " + " | ".join(stability_parts))
            if details_lines:
                st.write(" | ".join(details_lines))
        render_analysis_quality_box(
            mode=match.get("mode", state.get("mode")),
            odds_source=result.get("odds_source"),
            recommended_classes=[class_details["recommended_prediction_class"]],
            profile=str(analysis.get("match_profile", "not_available")),
            profiles_analysis=profiles_analysis,
            draw_warning=bool(class_details["draw_warning"]),
            is_draw_plausible=bool(analysis.get("is_draw_plausible")),
            caption=analysis_caption,
        )

    if show_save_button and st.button("Enregistrer cette prediction"):
        row = build_upcoming_log_row(
            prediction_result,
            state.get("odds_home"),
            state.get("odds_draw"),
            state.get("odds_away"),
        )
        append_upcoming_prediction_log(row)
        st.success(f"Prediction enregistree dans {UPCOMING_PREDICTIONS_LOG_PATH}")


def render_profile_reliability(profile: str, profiles_analysis: pd.DataFrame) -> None:
    """Show historical backtest performance for the selected match profile."""
    st.subheader("Fiabilite historique du profil")

    if profiles_analysis.empty or "match_profile" not in profiles_analysis.columns:
        st.info("Aucune analyse historique des profils n'est disponible pour l'instant.")
        return

    profile_rows = profiles_analysis[profiles_analysis["match_profile"] == profile]
    if profile_rows.empty:
        st.info(f"Aucune donnee historique disponible pour le profil `{profile}`.")
        return

    profile_row = profile_rows.iloc[0]
    matches = int(profile_row["matches"])
    accuracy = float(profile_row["accuracy_1N2"])
    if accuracy >= 0.65:
        reliability_label = "Fiabilite historique elevee"
    elif accuracy >= 0.50:
        reliability_label = "Fiabilite historique moyenne"
    else:
        reliability_label = "Fiabilite historique faible"

    st.write(f"Profil : **{profile}**")
    st.write(f"Sur le backtest : **{matches} matchs**")
    st.write(f"Evaluation : **{reliability_label}**")
    if matches < 30:
        st.warning("Echantillon faible")

    cols = st.columns(5)
    cols[0].metric("Accuracy historique", percent(accuracy))
    cols[1].metric("Victoire domicile reelle", percent(float(profile_row["actual_home_win_rate"])))
    cols[2].metric("Nul reel", percent(float(profile_row["actual_draw_rate"])))
    cols[3].metric("Victoire exterieure reelle", percent(float(profile_row["actual_away_win_rate"])))
    if "avg_confidence_score" in profile_row:
        cols[4].metric("Confidence moyenne", f"{float(profile_row['avg_confidence_score']):.1f}/100")


def render_draw_signal_box(analysis: dict[str, Any]) -> None:
    """Show extra context when the engine flags draw plausibility."""
    if not analysis.get("is_draw_plausible"):
        return

    st.subheader("Signal nul")
    calibrated_draw_signal = analysis.get("calibrated_draw_signal")
    if calibrated_draw_signal is not None:
        st.metric("Calibrated draw signal", percent(float(calibrated_draw_signal)))

    st.info(
        "Sur le backtest, le signal large `is_draw_plausible` a donne "
        f"{percent(DRAW_PLAUSIBLE_FALLBACK_RATE)} de nuls reels sur "
        f"{DRAW_PLAUSIBLE_FALLBACK_MATCHES} matchs."
    )


def normalized_data_source(mode: Any, odds_source: Any = None) -> str:
    """Return the user-facing data source label for an upcoming prediction."""
    mode_text = str(mode or "").strip()
    if mode_text in {"with-odds", "no-odds", "pseudo-odds"}:
        return mode_text
    if mode_text in {"experimental", "api_football"}:
        return "api-football"

    odds_source_text = str(odds_source or "").strip()
    if odds_source_text in {"manual", "with-odds"}:
        return "with-odds"
    if odds_source_text in {"elo_pseudo", "pseudo-odds"}:
        return "pseudo-odds"
    return "no-odds"


def engine_consensus_label(recommended_classes: list[str]) -> str:
    """Classify agreement across available engines."""
    classes = [str(value) for value in recommended_classes if pd.notna(value) and str(value).strip()]
    if not classes:
        return "non disponible"

    counts = pd.Series(classes).value_counts()
    if len(counts) == 1:
        return "fort"
    if int(counts.iloc[0]) > len(classes) / 2:
        return "moyen"
    return "faible"


def profile_reliability_row(profile: str, profiles_analysis: pd.DataFrame) -> pd.Series | None:
    """Return the historical row for a match profile when available."""
    if profiles_analysis.empty or "match_profile" not in profiles_analysis.columns:
        return None

    lookup_profile = str(profile).replace(" (majoritaire)", "")
    if lookup_profile.startswith("divergent:"):
        return None

    profile_rows = profiles_analysis[profiles_analysis["match_profile"].astype(str) == lookup_profile]
    if profile_rows.empty:
        return None
    return profile_rows.iloc[0]


def render_analysis_quality_box(
    mode: Any,
    odds_source: Any,
    recommended_classes: list[str],
    profile: str,
    profiles_analysis: pd.DataFrame | None,
    draw_warning: bool,
    is_draw_plausible: bool,
    caption: str | None = None,
) -> None:
    """Render a compact reliability box for upcoming-match analysis."""
    if profiles_analysis is None:
        profiles_analysis = pd.DataFrame()

    reliability = profile_reliability_row(profile, profiles_analysis)
    if reliability is None or "accuracy_1N2" not in reliability:
        accuracy_html = "not_available"
        matches_html = "not_available"
    else:
        accuracy_html = percent(float(reliability["accuracy_1N2"]))
        matches_html = f"{int(reliability['matches'])} matchs"

    status_class = "status-bad" if draw_warning or is_draw_plausible else "status-good"
    status_text = "Signal nul à surveiller" if draw_warning or is_draw_plausible else "Pas de signal nul particulier"
    content = f"""
        <div class="metric-label">Source des données</div>
        <div class="metric-value">{html.escape(normalized_data_source(mode, odds_source))}</div>
        <div style="height:10px;"></div>
        <div class="metric-label">Consensus des moteurs</div>
        <div class="metric-value">{html.escape(engine_consensus_label(recommended_classes))}</div>
        <div style="height:10px;"></div>
        <div class="metric-label">Profil du match</div>
        <div class="metric-value">{html.escape(profile)}</div>
        <div style="height:10px;"></div>
        <div class="metric-label">Accuracy historique</div>
        <div class="metric-value">{accuracy_html}</div>
        <div style="height:10px;"></div>
        <div class="metric-label">Échantillon historique</div>
        <div class="metric-value">{html.escape(matches_html)}</div>
        <div style="height:10px;"></div>
        <div class="{status_class}" style="padding:12px;border-radius:12px;">
            <div class="metric-label">Signal nul</div>
            <div class="metric-value">{html.escape(status_text)}</div>
        </div>
        <div style="height:10px;"></div>
        <div class="muted-text">{html.escape(caption or "Les données xG, compositions probables et blessures ne sont pas encore intégrées au moteur principal. Les pseudo-cotes Elo ont été retirées de l'interface pour éviter la confusion avec de vraies cotes bookmaker.")}</div>
    """
    render_html_card("Qualité de l'analyse", content, css_class=f"readability-card {status_class}")


def render_probability_blocks(
    result: dict[str, Any],
    home_team: str,
    away_team: str,
    profiles_analysis: pd.DataFrame | None = None,
    actual_result: str | None = None,
) -> None:
    """Render probabilities and analysis shared by both tabs."""
    match_probabilities = result["probabilities"]["match"]
    home_probabilities = result["probabilities"]["home"]
    away_probabilities = result["probabilities"]["away"]
    analysis = result["analysis"]
    class_details = prediction_classes(match_probabilities, result.get("league"))
    prediction = class_details["predicted_class_argmax"]
    top_issue_label = {"home_win": "Domicile", "draw": "Nul", "away_win": "Extérieur"}[max(match_probabilities, key=match_probabilities.get)]
    st.markdown("### Probabilités principales")
    render_probability_cards(
        match_probabilities["home_win"],
        match_probabilities["draw"],
        match_probabilities["away_win"],
        prediction,
    )

    st.markdown("### Probabilités par équipe")
    team_cols = st.columns(2)
    with team_cols[0]:
        render_html_card(
            home_team,
            f"""
                <div class="metric-label">Victoire</div><div class="metric-value">{home_probabilities["win"] * 100:.1f}%</div>
                <div class="metric-label" style="margin-top:10px;">Nul</div><div class="metric-value">{home_probabilities["draw"] * 100:.1f}%</div>
                <div class="metric-label" style="margin-top:10px;">Défaite</div><div class="metric-value">{home_probabilities["loss"] * 100:.1f}%</div>
                <div class="metric-label" style="margin-top:10px;">Non-défaite</div><div class="metric-value">{home_probabilities["no_loss"] * 100:.1f}%</div>
            """,
            css_class="section-card",
        )
    with team_cols[1]:
        render_html_card(
            away_team,
            f"""
                <div class="metric-label">Victoire</div><div class="metric-value">{away_probabilities["win"] * 100:.1f}%</div>
                <div class="metric-label" style="margin-top:10px;">Nul</div><div class="metric-value">{away_probabilities["draw"] * 100:.1f}%</div>
                <div class="metric-label" style="margin-top:10px;">Défaite</div><div class="metric-value">{away_probabilities["loss"] * 100:.1f}%</div>
                <div class="metric-label" style="margin-top:10px;">Non-défaite</div><div class="metric-value">{away_probabilities["no_loss"] * 100:.1f}%</div>
            """,
            css_class="section-card",
        )

    def _analysis_body() -> None:
        profile = analysis.get("match_profile", "not_available")
        confidence_score = analysis.get("confidence_score")
        metrics = st.columns(6)
        metrics[0].metric("Predicted argmax", prediction)
        metrics[1].metric("Match profile", profile)
        metrics[2].metric("Confidence", f"{float(confidence_score):.1f}/100" if confidence_score is not None else "not_available")
        metrics[3].metric("Favorite", analysis.get("favorite_team", "not_available"))
        metrics[4].metric("Uncertainty", f"{float(analysis.get('uncertainty_score', 0)):.3f}")
        metrics[5].metric("Top two margin", f"{float(analysis.get('top_two_margin', 0)):.3f}")

        recommendation_cols = st.columns(4)
        recommendation_cols[0].metric("Predicted adjusted", class_details["predicted_class_adjusted"])
        recommendation_cols[1].metric("Prediction recommandee", class_details["recommended_prediction_class"])
        recommendation_cols[2].metric("Draw warning", str(class_details["draw_warning"]))
        if actual_result:
            recommendation_cols[3].metric(
                "Recommended correcte",
                str(class_details["recommended_prediction_class"] == actual_result),
            )
        else:
            recommendation_cols[3].metric("Recommended correcte", "not_available")

        if class_details["draw_warning"]:
            st.warning(
                "Signal nul a surveiller. Ce signal a historiquement donne environ "
                f"{percent(DRAW_WARNING_BACKTEST_RATE)} de nuls sur le backtest Big 5."
            )

        extra_cols = st.columns(3)
        extra_cols[0].metric("Model version", str(analysis.get("model_version", "not_available")))
        extra_cols[1].metric("Odds source", str(result.get("odds_source", "not_available")))
        calibrated_draw_signal = analysis.get("calibrated_draw_signal")
        extra_cols[2].metric(
            "Calibrated draw signal",
            percent(float(calibrated_draw_signal)) if calibrated_draw_signal is not None else "not_available",
        )

        st.markdown("**Signal nul**")
        st.write(f"Draw plausible: **{analysis.get('is_draw_plausible', 'not_available')}**")
        st.write(f"Strong draw signal: **{analysis.get('is_strong_draw_signal', 'not_available')}**")
        st.write(f"Very strong draw signal: **{analysis.get('is_very_strong_draw_signal', 'not_available')}**")

        if profile != "not_available":
            st.markdown("**Lecture du modèle**")
            st.write(profile_sentence(profile, home_team, away_team))
            if profiles_analysis is not None:
                render_profile_reliability(profile, profiles_analysis)
            render_draw_signal_box(analysis)

        if actual_result:
            is_correct = prediction == actual_result
            is_correct_recommended = class_details["recommended_prediction_class"] == actual_result
            st.markdown("**Résultat réel**")
            st.write(f"Résultat réel: **{actual_result}**")
            st.write(f"Prédiction recommandée correcte: **{is_correct_recommended}**")
            if is_correct:
                st.success("Prédiction correcte.")
            else:
                st.warning("Prédiction incorrecte.")

    with st.expander("Détails du moteur", expanded=False):
        _analysis_body()


def comparison_profile_label(comparison_df: pd.DataFrame) -> str:
    """Return a compact profile label for a comparison table."""
    if "match_profile" not in comparison_df.columns:
        return "not_available"

    profiles = comparison_df["match_profile"].dropna().astype(str)
    profiles = profiles[profiles.str.strip() != ""]
    if profiles.empty:
        return "not_available"

    counts = profiles.value_counts()
    if len(counts) == 1:
        return str(counts.index[0])
    if int(counts.iloc[0]) > len(profiles) / 2:
        return f"{counts.index[0]} (majoritaire)"
    return "divergent: " + " / ".join(sorted(profiles.unique()))


def render_engine_comparison_result(state: dict[str, Any], profiles_analysis: pd.DataFrame | None = None) -> None:
    """Render the comparison table for upcoming-match engines."""
    warnings = state.get("warnings", [])
    for warning in warnings:
        st.warning(warning)

    comparison_df = state.get("comparison_df", pd.DataFrame())
    if comparison_df.empty:
        st.error("Aucun moteur comparé n'est disponible pour ce match.")
        return

    recommended_classes = (
        comparison_df["recommended_prediction_class"].dropna().astype(str).tolist()
        if "recommended_prediction_class" in comparison_df.columns
        else []
    )
    draw_warning = (
        comparison_df["draw_warning"].astype(bool).any()
        if "draw_warning" in comparison_df.columns
        else False
    )
    is_draw_plausible = (
        comparison_df["is_draw_plausible"].astype(bool).any()
        if "is_draw_plausible" in comparison_df.columns
        else False
    )
    render_analysis_quality_box(
        mode=state.get("mode"),
        odds_source=None,
        recommended_classes=recommended_classes,
        profile=comparison_profile_label(comparison_df),
        profiles_analysis=profiles_analysis,
        draw_warning=bool(draw_warning),
        is_draw_plausible=bool(is_draw_plausible),
    )
    with st.expander("Comparaison complète des moteurs", expanded=False):
        display_columns = [
            column
            for column in [
                "engine_name",
                "engine_status",
                "home_win_probability",
                "draw_probability",
                "away_win_probability",
                "predicted_class_argmax",
                "recommended_prediction_class",
                "match_profile",
                "confidence_score",
                "model_version",
            ]
            if column in comparison_df.columns
        ]
        table = comparison_df[display_columns].copy()
        for column in ["home_win_probability", "draw_probability", "away_win_probability"]:
            if column in table.columns:
                table[column] = table[column].astype(float).map(lambda value: f"{value * 100:.1f}%")
        if "confidence_score" in table.columns:
            table["confidence_score"] = table["confidence_score"].astype(float).map(lambda value: f"{value:.1f}/100")
        st.dataframe(table, use_container_width=True)


def render_historical_tab(df_features: pd.DataFrame, profiles_analysis: pd.DataFrame) -> None:
    """Render the historical match explorer tab."""
    date_column = first_existing_column(df_features, DATE_COLUMNS)
    home_column = first_existing_column(df_features, HOME_TEAM_COLUMNS)
    away_column = first_existing_column(df_features, AWAY_TEAM_COLUMNS)
    result_column = first_existing_column(df_features, RESULT_COLUMNS)

    if "league" in df_features.columns:
        league_options = ["all"] + sorted(df_features["league"].dropna().astype(str).unique().tolist())
        selected_league = st.selectbox(
            "Championnat", league_options, format_func=lambda value: "Tous les championnats" if value == "all" else value
        )
        if selected_league != "all":
            df_features = df_features[df_features["league"].astype(str) == selected_league].copy()

    if df_features.empty:
        st.info("Aucun match disponible pour ce filtre.")
        return

    if date_column:
        df_features = df_features.copy()
        df_features[date_column] = pd.to_datetime(df_features[date_column], errors="coerce")
        df_features = df_features.sort_values(date_column).reset_index(drop=True)

    display_columns = [column for column in [date_column, "league", home_column, away_column, result_column] if column and column in df_features.columns]
    display_df = df_features[display_columns].copy() if display_columns else df_features.copy()
    labels = [build_match_label(row, date_column, home_column, away_column) for _, row in display_df.iterrows()]
    selected_label = st.selectbox("Match", labels, index=len(labels) - 1)
    selected_index = labels.index(selected_label)
    selected_row = df_features.loc[selected_index]

    home_team = str(selected_row[home_column]) if home_column else "Equipe domicile"
    away_team = str(selected_row[away_column]) if away_column else "Equipe exterieure"
    st.info("L'analyse du match historique ne s'exécute qu'après clic sur le bouton ci-dessous.")

    if st.button("Analyser ce match historique", type="primary"):
        match_features = selected_row.to_dict()
        missing_advanced_columns = validate_historical_features(match_features)

        with st.expander("Debug — colonnes transmises au moteur"):
            st.write(f"Chemin du fichier chargé : {FEATURES_PATH}")
            st.write(f"df.shape : {df_features.shape}")
            st.write(f"df.columns.tolist() : {df_features.columns.tolist()}")
            st.write(f"list(match_features.keys()) : {list(match_features.keys())}")

        if missing_advanced_columns:
            st.error(
                "Les features avancées sont absentes du CSV chargé. Relancez python scripts/rebuild_all.py."
            )
            st.write("Colonnes manquantes :", missing_advanced_columns)
            st.stop()

        try:
            engine_result = predict_match_probabilities(match_features)
        except Exception as exc:
            st.error(str(exc))
            return

        result = {
            **engine_result,
            "odds_source": engine_result.get("analysis", {}).get("main_probability_source", "historical_features"),
            "league": selected_row["league"] if "league" in selected_row.index else None,
        }
        actual_result = str(selected_row[result_column]).upper() if result_column else None
        render_probability_blocks(result, home_team, away_team, profiles_analysis, actual_result)


def team_options(features: pd.DataFrame) -> list[str]:
    """Return sorted team names from feature data."""
    home_column = first_existing_column(features, HOME_TEAM_COLUMNS)
    away_column = first_existing_column(features, AWAY_TEAM_COLUMNS)
    if not home_column or not away_column:
        return []
    teams = pd.concat([features[home_column], features[away_column]], ignore_index=True).dropna().astype(str)
    teams = teams.str.strip()
    teams = teams[teams != ""]
    return sorted(teams.unique().tolist())


def current_team_options(current_teams: pd.DataFrame, league: str | None) -> tuple[list[str], str | None]:
    """Return current teams and the reference season for the selected league."""
    required_columns = {"league", "season", "team"}
    if current_teams.empty or not required_columns.issubset(current_teams.columns) or league is None:
        return [], None

    league_teams = current_teams[current_teams["league"].astype(str) == league].copy()
    if league_teams.empty:
        return [], None

    seasons = sorted(league_teams["season"].dropna().astype(str).unique().tolist())
    season = seasons[-1] if seasons else None
    if season is not None:
        league_teams = league_teams[league_teams["season"].astype(str) == season].copy()

    teams = league_teams["team"].dropna().astype(str).str.strip()
    teams = teams[teams != ""]
    return sorted(teams.unique().tolist()), season


def fixture_options(fixtures: pd.DataFrame) -> list[str]:
    """Return sorted league names available in upcoming fixtures."""
    if fixtures.empty or "league" not in fixtures.columns:
        return []
    return sorted(fixtures["league"].dropna().astype(str).unique().tolist())


def fixture_label(row: pd.Series) -> str:
    """Build the upcoming fixture selectbox label."""
    return f"{row['date']} - {row['home_team']} vs {row['away_team']}"


def optional_float(value: Any) -> float | None:
    """Return a float for populated numeric values, otherwise None."""
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fixture_has_odds(row: pd.Series) -> bool:
    """Return whether a fixture has all three usable odds."""
    odds = [optional_float(row.get(column)) for column in ["odds_home", "odds_draw", "odds_away"]]
    return all(value is not None and value > 1.0 for value in odds)


def league_options(features: pd.DataFrame) -> list[str]:
    """Return sorted league names when available."""
    if "league" not in features.columns:
        return []
    return sorted(features["league"].dropna().astype(str).unique().tolist())


def _render_upcoming_tab_legacy(features: pd.DataFrame) -> None:
    """Render the upcoming match analysis tab."""
    selected_league = None
    leagues = league_options(features)
    if leagues:
        selected_league = st.selectbox("Championnat", leagues)
        features = features[features["league"].astype(str) == selected_league].copy()

    current_teams = load_current_teams()
    reference_season = None
    if current_teams.empty and not CURRENT_TEAMS_PATH.exists():
        st.warning("Liste actuelle des équipes indisponible, fallback sur l’historique.")
        teams = team_options(features)
    else:
        teams, reference_season = current_team_options(current_teams, selected_league)
        if not teams:
            teams = team_options(features)

    if not teams:
        st.error("Impossible de charger la liste des equipes depuis matches_features.csv.")
        return
    if len(teams) < 2:
        st.error("Il faut au moins deux equipes distinctes dans matches_features.csv.")
        return

    if reference_season:
        st.caption(f"Équipes affichées : saison {reference_season}")

    home_team = st.selectbox("Equipe domicile", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
    away_options = [team for team in teams if team != home_team]
    away_default_team = "Chelsea" if "Chelsea" in away_options else away_options[0]
    away_team = st.selectbox("Equipe exterieure", away_options, index=away_options.index(away_default_team))
    match_date = st.date_input("Date du match", value=date.today())
    mode = st.radio(
        "Mode",
        ["no-odds", "with-odds"],
        format_func={
            "no-odds": "no-odds : sans cotes",
            "with-odds": "with-odds : avec cotes reelles",
        }.get,
        horizontal=True,
    )
    display_mode = st.radio(
        "Affichage",
        ["Analyse simple", "Comparer les moteurs"],
        horizontal=True,
    )
    comparison_mode = display_mode == "Comparer les moteurs"

    odds_home = odds_draw = odds_away = None
    if mode == "with-odds":
        odds_cols = st.columns(3)
        odds_home = parse_optional_odd(odds_cols[0].text_input("Cote victoire domicile", value=""))
        odds_draw = parse_optional_odd(odds_cols[1].text_input("Cote nul", value=""))
        odds_away = parse_optional_odd(odds_cols[2].text_input("Cote victoire exterieure", value=""))
    current_prediction_key = (
        selected_league,
        home_team,
        away_team,
        match_date.isoformat(),
        engine_choice,
        mode,
        comparison_mode,
        odds_home,
        odds_draw,
        odds_away,
    )

    if st.button("Analyser le match", type="primary"):
        if home_team == away_team:
            st.error("Selection impossible : l'equipe domicile et l'equipe exterieure doivent etre differentes.")
            return

        if (use_main_engine or comparison_mode) and mode == "with-odds":
            odds = {
                "victoire domicile": odds_home,
                "nul": odds_draw,
                "victoire exterieure": odds_away,
            }
            missing_or_invalid = [label for label, value in odds.items() if value is None or value <= 1.0]
            if missing_or_invalid:
                st.error(
                    "Le mode with-odds necessite trois cotes renseignees et strictement superieures a 1 : "
                    + ", ".join(missing_or_invalid)
                    + "."
                )
                return

        if use_experimental_engine:
            prediction_result = build_ligue1_api_xg_v2_prediction_result(home_team, away_team, match_date, selected_league)
            if prediction_result is None:
                st.error("Features API-Football indisponibles pour ce match. Utilisez le moteur principal.")
                return

            st.session_state["upcoming_prediction_state"] = {
                "prediction_key": current_prediction_key,
                "engine_choice": engine_choice,
                "mode": mode,
                "prediction_result": prediction_result,
                "odds_home": None,
                "odds_draw": None,
                "odds_away": None,
            }
        elif comparison_mode:
            comparison_df, warnings = build_engine_comparison_table(
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                selected_league=selected_league,
                mode=mode,
                odds_home=odds_home,
                odds_draw=odds_draw,
                odds_away=odds_away,
            )
            st.session_state["upcoming_prediction_state"] = {
                "prediction_key": current_prediction_key,
                "comparison_mode": True,
                "comparison_df": comparison_df,
                "warnings": warnings,
            }
        else:
            try:
                prediction_result = predict_upcoming_match(
                    home_team,
                    away_team,
                    match_date.isoformat(),
                    mode=mode,
                    odds_home=odds_home,
                    odds_draw=odds_draw,
                    odds_away=odds_away,
                    league=selected_league,
                )
            except Exception as exc:
                st.error(str(exc))
                return

            st.session_state["upcoming_prediction_state"] = {
                "prediction_key": current_prediction_key,
                "prediction_result": prediction_result,
                "odds_home": odds_home,
                "odds_draw": odds_draw,
                "odds_away": odds_away,
            }

    prediction_state = st.session_state.get("upcoming_prediction_state")
    if prediction_state and prediction_state.get("prediction_key") == current_prediction_key:
        if prediction_state.get("comparison_mode"):
            render_engine_comparison_result(prediction_state)
        else:
            render_upcoming_prediction_result(prediction_state)


def render_upcoming_tab(features: pd.DataFrame, profiles_analysis: pd.DataFrame) -> None:
    """Render the upcoming match analysis tab."""
    upcoming_fixtures = load_upcoming_fixtures()
    input_mode_options = ["Sélectionner un match à venir", "Saisie manuelle"]
    default_mode_index = 1 if upcoming_fixtures.empty else 0
    input_mode = st.radio(
        "Mode de saisie",
        input_mode_options,
        index=default_mode_index,
        horizontal=True,
    )

    selected_league = None
    home_team = ""
    away_team = ""
    match_date = date.today()
    selected_fixture_id: Any | None = None
    default_mode = "no-odds"
    odds_defaults = {"home": "", "draw": "", "away": ""}

    use_manual_input = input_mode == "Saisie manuelle"
    if input_mode == "Sélectionner un match à venir":
        fixtures = upcoming_fixtures.copy()
        if fixtures.empty:
            st.warning("Aucune fixture à venir disponible. Utilisez la saisie manuelle.")
            use_manual_input = True
        else:
            fixture_leagues = fixture_options(fixtures)
            selected_league = st.selectbox("Championnat", fixture_leagues, key="upcoming_fixture_league")
            league_fixtures = fixtures[fixtures["league"].astype(str) == selected_league].copy()
            league_fixtures = league_fixtures.sort_values(["date", "home_team"]).reset_index(drop=True)

            labels = [fixture_label(row) for _, row in league_fixtures.iterrows()]
            selected_label = st.selectbox("Match", labels, key="upcoming_fixture_match")
            selected_fixture = league_fixtures.loc[labels.index(selected_label)]

            home_team = str(selected_fixture["home_team"])
            away_team = str(selected_fixture["away_team"])
            parsed_match_date = pd.to_datetime(selected_fixture["date"], errors="coerce")
            match_date = parsed_match_date.date() if pd.notna(parsed_match_date) else date.today()
            selected_fixture_id = _fixture_id_from_row(selected_fixture)

            if fixture_has_odds(selected_fixture):
                default_mode = "with-odds"
                odds_defaults = {
                    "home": str(optional_float(selected_fixture["odds_home"])),
                    "draw": str(optional_float(selected_fixture["odds_draw"])),
                    "away": str(optional_float(selected_fixture["odds_away"])),
                }

            st.caption(f"{selected_league} - {home_team} vs {away_team} - {match_date.isoformat()}")

    if use_manual_input:
        leagues = league_options(features)
        if leagues:
            selected_league = st.selectbox("Championnat", leagues, key="upcoming_manual_league")
            features = features[features["league"].astype(str) == selected_league].copy()

        current_teams = load_current_teams()
        reference_season = None
        if current_teams.empty and not CURRENT_TEAMS_PATH.exists():
            st.warning("Liste actuelle des équipes indisponible, fallback sur l’historique.")
            teams = team_options(features)
        else:
            teams, reference_season = current_team_options(current_teams, selected_league)
            if not teams:
                teams = team_options(features)

        if not teams:
            st.error("Impossible de charger la liste des equipes depuis matches_features.csv.")
            return
        if len(teams) < 2:
            st.error("Il faut au moins deux equipes distinctes dans matches_features.csv.")
            return

        if reference_season:
            st.caption(f"Équipes affichées : saison {reference_season}")

        home_team = st.selectbox(
            "Equipe domicile",
            teams,
            index=teams.index("Arsenal") if "Arsenal" in teams else 0,
            key="upcoming_manual_home_team",
        )
        away_options = [team for team in teams if team != home_team]
        away_default_team = "Chelsea" if "Chelsea" in away_options else away_options[0]
        away_team = st.selectbox(
            "Equipe exterieure",
            away_options,
            index=away_options.index(away_default_team),
            key="upcoming_manual_away_team",
        )
        match_date = st.date_input("Date du match", value=date.today(), key="upcoming_manual_match_date")

    if selected_league == "Ligue 1":
        st.caption("Bloc sélection fiable chargé pour Ligue 1")
        selection_intro = """
            <p class="muted-text">Analyse automatiquement les matchs Ligue 1 disponibles avec le moteur API xG v2 et classe les matchs selon leur fiabilité.</p>
        """
        render_html_card("Sélection des matchs les plus fiables", selection_intro, css_class="section-card")
        selected_reliable_count = st.selectbox(
            "Nombre de matchs à sélectionner",
            [4, 5, 6, 7, 8],
            index=0,
            key="ligue1_reliable_selection_count",
        )
        if st.button("Générer la sélection fiable", key="ligue1_reliable_selection_button"):
            selected_df, selection_warnings, analyzed_count = build_ligue1_reliable_match_selection(
                int(selected_reliable_count),
                upcoming_fixtures=upcoming_fixtures,
            )
            st.session_state["ligue1_reliable_selection"] = {
                "selected_count": int(selected_reliable_count),
                "selection_df": selected_df,
                "warnings": selection_warnings,
                "analyzed_count": analyzed_count,
            }

        selection_state = st.session_state.get("ligue1_reliable_selection")
        if selection_state:
            selection_df = selection_state.get("selection_df")
            if isinstance(selection_df, pd.DataFrame) and not selection_df.empty:
                warnings_list = selection_state.get("warnings") or []
                if warnings_list:
                    for warning in warnings_list:
                        st.warning(warning)

                selected_count_value = len(selection_df)
                analyzed_count_value = int(selection_state.get("analyzed_count", len(selection_df)))
                status_counts = selection_df["recommendation_status"].value_counts()
                mean_top_probability = float(selection_df["top_probability"].mean()) if not selection_df.empty else 0.0
                mean_reliability_score = float(selection_df["reliability_score"].mean()) if not selection_df.empty else 0.0

                summary_cols = st.columns(4)
                summary_specs = [
                        ("Matchs analysés", f"{analyzed_count_value}", "section-card"),
                    ("Matchs sélectionnés", f"{selected_count_value}", "section-card"),
                    ("Probabilité dominante moyenne", percent(mean_top_probability), "section-card"),
                    ("Score fiabilité moyen", f"{mean_reliability_score:.2f}", "section-card"),
                ]
                for column, (title, value, css_class) in zip(summary_cols, summary_specs, strict=False):
                    with column:
                        render_html_card(title, f'<div class="metric-value">{html.escape(str(value))}</div>', css_class=css_class)

                render_html_card(
                    "Répartition des statuts",
                    f'<div class="metric-value">{int(status_counts.get("exploitable", 0))} exploitables · {int(status_counts.get("prudence", 0))} prudence · {int(status_counts.get("à éviter", 0))} à éviter</div>',
                    css_class="section-card",
                )

                display_columns = [
                    "rang",
                    "date",
                    "home_team_name",
                    "away_team_name",
                    "predicted_class",
                    "home_win_probability",
                    "draw_probability",
                    "away_win_probability",
                    "top_probability",
                    "top_two_margin",
                    "confidence_score",
                    "readability_level",
                    "recommendation_status",
                    "reliability_score",
                    "principales_raisons_de_lisibilite",
                ]
                available_display_columns = [column for column in display_columns if column in selection_df.columns]
                table = selection_df[available_display_columns].copy()
                table = table.rename(
                    columns={
                        "predicted_class": "Pronostic",
                        "top_probability": "Probabilité dominante",
                        "top_two_margin": "Écart 1er/2e",
                        "readability_level": "Lisibilité",
                        "recommendation_status": "Statut",
                        "reliability_score": "Score fiabilité",
                        "principales_raisons_de_lisibilite": "Raisons",
                        "home_team_name": "Domicile",
                        "away_team_name": "Extérieur",
                        "confidence_score": "Confiance",
                        "home_win_probability": "Proba domicile",
                        "draw_probability": "Proba nul",
                        "away_win_probability": "Proba extérieur",
                        "date": "Date",
                    }
                )
                st.dataframe(table, use_container_width=True, hide_index=True)

                if (selection_df["recommendation_status"] == "à éviter").any():
                    st.warning("Quelques matchs 'à éviter' ont été inclus.")

                if st.button("Enregistrer la sélection", key="ligue1_reliable_selection_save_button"):
                    tracking_rows = build_ligue1_api_xg_live_tracking_rows_from_selection(selection_df)
                    saved_count = append_ligue1_api_xg_live_tracking_rows(tracking_rows)
                    if saved_count:
                        st.success(
                            f"{saved_count} prédiction(s) enregistrée(s) dans {LIGUE1_API_XG_LIVE_TRACKING_PATH}"
                        )
                    else:
                        st.info("Sélection déjà enregistrée ou aucune ligne valide à sauvegarder.")
            else:
                st.info("Cliquez sur 'Générer la sélection fiable' pour calculer les matchs les plus fiables.")
        with st.expander("Détails de calcul de la fiabilité", expanded=False):
            st.markdown(
                """
                `reliability_score = top_probability * 100 + top_two_margin * 80 + confidence_score * 0.3 + bonus_lisibilite - penalty_draw - penalty_low_margin`

                - `bonus_lisibilite = 15` si lisibilité forte
                - `bonus_lisibilite = 5` si lisibilité moyenne
                - `bonus_lisibilite = -15` si lisibilité faible
                - `penalty_draw = 10` si `draw_probability >= 0.32`, sinon `0`
                - `penalty_low_margin = 10` si `top_two_margin < 0.05`, sinon `0`
                """
            )

    ligue1_api_xg_v2_match_features: dict[str, Any] | None = None
    ligue1_api_xg_v2_mapping_debug: dict[str, Any] | None = None
    if selected_league == "Ligue 1":
        ligue1_api_xg_v2_match_features, ligue1_api_xg_v2_mapping_debug = inspect_ligue1_api_xg_v2_match_features(
            home_team,
            away_team,
            match_date,
            fixture_id=selected_fixture_id,
        )

    experimental_available = ligue1_api_xg_v2_match_features is not None
    if (
        selected_league == "Ligue 1"
        and experimental_available
        and ligue1_api_xg_v2_mapping_debug is not None
        and ligue1_api_xg_v2_mapping_debug.get("loaded_file") == str(LIGUE1_API_XG_INJURY_FEATURES_PATH)
    ):
        st.info("Features utilisees : xG, tirs, stabilite de formation, blessures ponderees.")

    if selected_league == "Ligue 1":
        engine_labels_by_code = {
            "ligue1_api_xg": "Ligue 1 API xG — recommandé",
            "main_no_odds": "Moteur principal sans cotes",
            "main_with_odds": "Moteur principal avec cotes réelles",
            "compare_engines": "Comparaison des moteurs",
        }
        engine_options = list(engine_labels_by_code.values())
        default_selected_engine = "ligue1_api_xg" if experimental_available else "main_no_odds"
    else:
        engine_labels_by_code = {
            "main_no_odds": "Moteur principal sans cotes",
            "main_with_odds": "Moteur principal avec cotes réelles",
            "compare_engines": "Comparaison des moteurs",
        }
        engine_options = list(engine_labels_by_code.values())
        default_selected_engine = "main_no_odds"

    selected_engine_label = st.selectbox(
        "Moteur d'analyse",
        engine_options,
        index=engine_options.index(engine_labels_by_code[default_selected_engine]),
    )
    selected_engine = next(
        (code for code, label in engine_labels_by_code.items() if label == selected_engine_label),
        None,
    )
    engine_captions_by_code = {
        "ligue1_api_xg": (
            "Moteur spécialisé Ligue 1 basé sur xG récents, tirs, tirs cadrés, "
            "stabilité de formation et blessures pondérées."
        ),
        "main_no_odds": "Moteur principal basé sur l'Elo et la forme récente, sans utiliser les cotes du marché.",
        "main_with_odds": "Moteur principal combinant Elo, forme récente et cotes réelles du marché.",
        "compare_engines": "Compare les résultats de tous les moteurs disponibles pour ce match.",
    }
    engine_caption = engine_captions_by_code.get(selected_engine)
    if engine_caption:
        st.caption(engine_caption)
    if selected_league == "Ligue 1" and not experimental_available:
        missing_features = []
        loaded_file = None
        if ligue1_api_xg_v2_mapping_debug is not None:
            missing_features = ligue1_api_xg_v2_mapping_debug.get("missing_features", []) or []
            loaded_file = ligue1_api_xg_v2_mapping_debug.get("loaded_file")
        injury_features = {
            "home_injury_impact_score",
            "away_injury_impact_score",
            "injury_impact_diff",
            "home_likely_starter_injuries_count",
            "away_likely_starter_injuries_count",
            "likely_starter_injuries_diff",
        }
        if loaded_file == str(LIGUE1_API_XG_INJURY_FEATURES_PATH) and not any(feature in missing_features for feature in injury_features):
            st.info("Features utilisees : xG, tirs, stabilite de formation, blessures ponderees.")
        elif loaded_file in {
            str(LIGUE1_API_XG_FORMATION_FEATURES_PATH),
            str(LIGUE1_API_XG_FEATURES_PATH),
        } and any(feature in missing_features for feature in injury_features):
            st.warning("Les blessures ponderees ne sont pas disponibles. Relancez le script d’impact des blessures.")
        else:
            st.warning(
                "Les donnees enrichies API-Football ne sont pas disponibles pour ce match. "
                "Essayez le moteur principal sans cotes."
            )
        same_date = ligue1_api_xg_v2_mapping_debug.get("available_matches_same_date") if ligue1_api_xg_v2_mapping_debug else None
        if isinstance(same_date, pd.DataFrame) and not same_date.empty:
            available_teams: list[str] = []
            if "home_team_name" in same_date.columns:
                available_teams.extend(same_date["home_team_name"].dropna().astype(str).tolist())
            if "away_team_name" in same_date.columns:
                available_teams.extend(same_date["away_team_name"].dropna().astype(str).tolist())
            available_teams = sorted({team.strip() for team in available_teams if team and team.strip()})
            if available_teams:
                st.info("Equipes disponibles a cette date : " + ", ".join(available_teams))
        if ligue1_api_xg_v2_mapping_debug is not None:
            render_ligue1_api_xg_v2_mapping_debug(ligue1_api_xg_v2_mapping_debug)
    elif selected_league == "Ligue 1" and ligue1_api_xg_v2_mapping_debug is not None:
        render_ligue1_api_xg_v2_mapping_debug(ligue1_api_xg_v2_mapping_debug)

    allowed_engines = set(engine_labels_by_code)
    if selected_engine not in allowed_engines:
        st.error("Le moteur sélectionné n'est pas reconnu. Merci de choisir un moteur dans la liste.")
        return

    comparison_mode = selected_engine == "compare_engines"
    use_experimental_engine = selected_engine == "ligue1_api_xg"
    use_with_odds_engine = selected_engine == "main_with_odds"

    odds_home = odds_draw = odds_away = None
    if use_with_odds_engine or comparison_mode:
        odds_cols = st.columns(3)
        odds_home = parse_optional_odd(
            odds_cols[0].text_input("Cote victoire domicile", value=odds_defaults["home"])
        )
        odds_draw = parse_optional_odd(odds_cols[1].text_input("Cote nul", value=odds_defaults["draw"]))
        odds_away = parse_optional_odd(
            odds_cols[2].text_input("Cote victoire exterieure", value=odds_defaults["away"])
        )

    main_mode = "with-odds" if use_with_odds_engine else "no-odds"

    current_prediction_key = (
        selected_league,
        home_team,
        away_team,
        match_date.isoformat(),
        selected_engine,
        main_mode,
        comparison_mode,
        odds_home,
        odds_draw,
        odds_away,
    )

    if st.button("Analyser le match", type="primary"):
        if home_team == away_team:
            st.error("Selection impossible : l'equipe domicile et l'equipe exterieure doivent etre differentes.")
            return

        if use_with_odds_engine:
            odds = {
                "victoire domicile": odds_home,
                "nul": odds_draw,
                "victoire exterieure": odds_away,
            }
            missing_or_invalid = [label for label, value in odds.items() if value is None or value <= 1.0]
            if missing_or_invalid:
                st.error(
                    "Le mode with-odds necessite trois cotes renseignees et strictement superieures a 1 : "
                    + ", ".join(missing_or_invalid)
                    + "."
                )
                return

        if use_experimental_engine:
            if ligue1_api_xg_v2_match_features is None:
                st.error(
                    "Les données enrichies API-Football ne sont pas disponibles pour ce match. "
                    "Essayez le moteur principal sans cotes."
                )
                return
            try:
                prediction = predict_ligue1_api_xg_v2(ligue1_api_xg_v2_match_features)
            except Exception as exc:
                st.error(str(exc))
                return

            home_prob = float(prediction["home_win_probability"])
            draw_prob = float(prediction["draw_probability"])
            away_prob = float(prediction["away_win_probability"])
            favorite_probability = max(home_prob, draw_prob, away_prob)
            if favorite_probability == home_prob:
                favorite_team: str | None = home_team
            elif favorite_probability == away_prob:
                favorite_team = away_team
            else:
                favorite_team = "draw"

            ligue1_prediction_result = {
                "match": {
                    "home_team": home_team,
                    "away_team": away_team,
                    "mode": "experimental",
                    "league": selected_league,
                    "warning": None,
                },
                "league": selected_league,
                "odds_source": "api_football",
                "prediction": {
                    "probabilities": {
                        "match": {
                            "home_win": home_prob,
                            "draw": draw_prob,
                            "away_win": away_prob,
                        },
                        "home": {
                            "win": home_prob,
                            "draw": draw_prob,
                            "loss": away_prob,
                            "no_loss": home_prob + draw_prob,
                        },
                        "away": {
                            "win": away_prob,
                            "draw": draw_prob,
                            "loss": home_prob,
                            "no_loss": away_prob + draw_prob,
                        },
                    },
                    "analysis": {
                        "predicted_class_argmax": prediction["predicted_class_argmax"],
                        "predicted_class_adjusted": prediction["predicted_class_argmax"],
                        "recommended_prediction_class": prediction["predicted_class_argmax"],
                        "draw_warning": False,
                        "match_profile": "experimental_api_xg_v2",
                        "confidence_score": prediction["confidence_score"],
                        "favorite_team": favorite_team,
                        "favorite_probability": favorite_probability,
                        "uncertainty_score": 1 - favorite_probability,
                        "top_two_margin": favorite_probability - sorted([home_prob, draw_prob, away_prob], reverse=True)[1],
                        "model_version": prediction["model_version"],
                        "is_draw_plausible": draw_prob >= 0.27,
                        "calibrated_draw_signal": None,
                    },
                    "match": {
                        "home_team": home_team,
                        "away_team": away_team,
                        "mode": "experimental",
                        "league": selected_league,
                        "warning": None,
                    },
                    "league": selected_league,
                },
            }
            readability = compute_match_readability(ligue1_prediction_result, ligue1_api_xg_v2_match_features)
            reliability_score = round(
                (float(readability.get("top_probability", favorite_probability)) * 100)
                + (float(readability.get("top_two_margin", 0.0)) * 80)
                + (float(readability.get("confidence_score", 0.0)) * 0.3)
                + _readability_bonus(str(readability.get("readability_level", "")))
                - (10 if float(readability.get("draw_probability", 0.0)) >= 0.32 else 0)
                - (10 if float(readability.get("top_two_margin", 0.0)) < 0.05 else 0),
                2,
            )
            st.session_state["upcoming_prediction_state"] = {
                "prediction_key": current_prediction_key,
                "selected_engine": selected_engine,
                "selected_league": selected_league,
                "match_date": match_date.isoformat(),
                "prediction_result": ligue1_prediction_result,
                "match_features": ligue1_api_xg_v2_match_features,
                "odds_home": None,
                "odds_draw": None,
                "odds_away": None,
                "readability": readability,
                "reliability_score": reliability_score,
            }
        elif comparison_mode:
            comparison_df, warnings = build_upcoming_comparison_table(
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                selected_league=selected_league,
                odds_home=odds_home,
                odds_draw=odds_draw,
                odds_away=odds_away,
                fixture_id=selected_fixture_id,
            )
            st.session_state["upcoming_prediction_state"] = {
                "prediction_key": current_prediction_key,
                "selected_engine": selected_engine,
                "selected_league": selected_league,
                "match_date": match_date.isoformat(),
                "comparison_mode": True,
                "comparison_df": comparison_df,
                "warnings": warnings,
            }
        else:
            try:
                prediction_result = predict_upcoming_match(
                    home_team,
                    away_team,
                    match_date.isoformat(),
                    mode=main_mode,
                    odds_home=odds_home if use_with_odds_engine else None,
                    odds_draw=odds_draw if use_with_odds_engine else None,
                    odds_away=odds_away if use_with_odds_engine else None,
                    league=selected_league,
                )
            except Exception as exc:
                st.error(str(exc))
                return

            st.session_state["upcoming_prediction_state"] = {
                "prediction_key": current_prediction_key,
                "selected_engine": selected_engine,
                "selected_league": selected_league,
                "match_date": match_date.isoformat(),
                "main_mode": main_mode,
                "prediction_result": prediction_result,
                "odds_home": odds_home,
                "odds_draw": odds_draw,
                "odds_away": odds_away,
            }

    prediction_state = st.session_state.get("upcoming_prediction_state")
    if prediction_state and prediction_state.get("prediction_key") == current_prediction_key:
        if prediction_state.get("comparison_mode"):
            render_engine_comparison_result(prediction_state, profiles_analysis)
        elif prediction_state.get("selected_engine") == "ligue1_api_xg":
            render_upcoming_prediction_result(prediction_state, profiles_analysis, show_save_button=False)
            tracking_row = build_ligue1_api_xg_live_tracking_row(prediction_state)
            if tracking_row is not None:
                if st.button("Enregistrer cette prédiction"):
                    saved = append_ligue1_api_xg_live_tracking(tracking_row)
                    if saved:
                        st.success(f"Prediction enregistree dans {LIGUE1_API_XG_LIVE_TRACKING_PATH}")
                    else:
                        st.info("Cette prédiction est deja enregistree.")
        else:
            render_upcoming_prediction_result(prediction_state, profiles_analysis)

        if selected_league == "Ligue 1":
            render_ligue1_match_context(
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                fixture_id=selected_fixture_id,
            )
            render_ligue1_composition_rotation_context(
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                fixture_id=selected_fixture_id,
            )


def render_predictions_evaluation_tab() -> None:
    """Render evaluated upcoming predictions."""
    if st.button("Rafraichir les evaluations"):
        st.info(
            "Pour rafraichir les evaluations, lancez dans le terminal : "
            "python app/backtesting/evaluate_upcoming_predictions.py"
        )

    evaluated_predictions = load_upcoming_predictions_evaluated()
    if evaluated_predictions.empty and not UPCOMING_PREDICTIONS_EVALUATED_PATH.exists():
        st.info(
            "Aucune evaluation disponible. Lancez python app/backtesting/evaluate_upcoming_predictions.py "
            "apres avoir enregistre des predictions."
        )
        return

    if evaluated_predictions.empty:
        st.info("Le fichier d'evaluation existe, mais il ne contient aucune prediction.")

    st.subheader("Performance réelle Ligue 1 API xG")
    live_tracking = load_ligue1_api_xg_live_tracking()
    if live_tracking.empty and not LIGUE1_API_XG_LIVE_TRACKING_PATH.exists():
        st.info("Aucune prediction API xG en suivi n'est disponible pour l'instant.")
    else:
        if live_tracking.empty:
            st.info("Le fichier de suivi existe, mais il ne contient aucune prediction.")

        live_tracking = _evaluate_ligue1_api_xg_live_tracking(live_tracking)
        tracked_count = len(live_tracking)
        evaluated_live = live_tracking[
            live_tracking["actual_result"].fillna("").astype(str).str.strip() != ""
        ].copy()
        evaluated_count = len(evaluated_live)

        metric_cols = st.columns(4)
        metric_cols[0].metric("Predictions enregistrées", tracked_count)
        metric_cols[1].metric("Predictions évaluées", evaluated_count)
        if evaluated_count:
            metric_cols[2].metric("Accuracy globale", percent(float(evaluated_live["is_correct"].mean())))
        else:
            metric_cols[2].metric("Accuracy globale", "not_available")
        metric_cols[3].metric("Source", "Ligue 1 API xG")

        if evaluated_count:
            evaluated_live["is_correct"] = evaluated_live["is_correct"].astype(str).str.lower() == "true"
            readability_accuracy = evaluated_live.groupby("readability_level")["is_correct"].mean()
            recommendation_accuracy = evaluated_live.groupby("recommendation_status")["is_correct"].mean()
            class_accuracy = evaluated_live.groupby("predicted_class")["is_correct"].mean().reindex(["H", "D", "A"])
            evaluated_live["top_probability"] = pd.to_numeric(evaluated_live["top_probability"], errors="coerce")
            evaluated_live["top_probability_bucket"] = evaluated_live["top_probability"].apply(_top_probability_bucket)
            top_probability_accuracy = evaluated_live.groupby("top_probability_bucket")["is_correct"].mean()

            summary_cols = st.columns(2)
            with summary_cols[0]:
                render_html_card(
                    "Lisibilité",
                    "<div class='metric-label'>Accuracy par lisibilité</div>"
                    + "<div class='metric-value'>"
                    + ", ".join(
                        f"{label}: {percent(float(value))}"
                        for label, value in readability_accuracy.items()
                        if pd.notna(value)
                    )
                    + "</div>",
                    css_class="section-card",
                )
                render_html_card(
                    "Recommandation",
                    "<div class='metric-label'>Accuracy par statut</div>"
                    + "<div class='metric-value'>"
                    + ", ".join(
                        f"{label}: {percent(float(value))}"
                        for label, value in recommendation_accuracy.items()
                        if pd.notna(value)
                    )
                    + "</div>",
                    css_class="section-card",
                )
            with summary_cols[1]:
                render_html_card(
                    "Pronostic",
                    "<div class='metric-label'>Accuracy par classe</div>"
                    + "<div class='metric-value'>"
                    + ", ".join(
                        f"{label}: {percent(float(value))}"
                        for label, value in class_accuracy.items()
                        if pd.notna(value)
                    )
                    + "</div>",
                    css_class="section-card",
                )
                render_html_card(
                    "Probabilité dominante",
                    "<div class='metric-label'>Accuracy par bucket</div>"
                    + "<div class='metric-value'>"
                    + ", ".join(
                        f"{label}: {percent(float(value))}"
                        for label, value in top_probability_accuracy.items()
                        if pd.notna(value)
                    )
                    + "</div>",
                    css_class="section-card",
                )

            st.caption("Buckets de probabilité dominante: < 0.43, 0.43-0.50, 0.50-0.60, > 0.60.")
        else:
            st.info("Aucune prediction n'a encore de resultat reel renseigne.")

        display_columns = [column for column in LIGUE1_API_XG_LIVE_TRACKING_COLUMNS if column in live_tracking.columns]
        if display_columns:
            st.caption("Renseignez actual_result dans le CSV avec H, D ou A, puis relancez l'app.")
            edited_tracking = st.data_editor(
                live_tracking[display_columns].copy(),
                use_container_width=True,
                num_rows="fixed",
                column_config={
                    "actual_result": st.column_config.SelectboxColumn(
                        "actual_result",
                        options=["", "H", "D", "A"],
                        help="Mettre a jour le resultat reel du match.",
                    ),
                },
                disabled=[column for column in display_columns if column != "actual_result"],
                key="ligue1_api_xg_live_tracking_editor",
            )
            if st.button("Enregistrer les resultats saisis"):
                updated = edited_tracking.copy()
                updated = _evaluate_ligue1_api_xg_live_tracking(updated)
                updated = updated[LIGUE1_API_XG_LIVE_TRACKING_COLUMNS]
                updated.to_csv(LIGUE1_API_XG_LIVE_TRACKING_PATH, index=False)
                load_ligue1_api_xg_live_tracking.clear()
                st.success(f"Suivi mis a jour dans {LIGUE1_API_XG_LIVE_TRACKING_PATH}")

    filtered = evaluated_predictions.copy()
    status_options = ["all"]
    if "evaluation_status" in filtered.columns:
        status_options += sorted(filtered["evaluation_status"].dropna().astype(str).unique().tolist())
    selected_status = st.selectbox(
        "Statut", status_options, format_func=lambda value: "Tous les statuts" if value == "all" else value
    )
    if selected_status != "all" and "evaluation_status" in filtered.columns:
        filtered = filtered[filtered["evaluation_status"].astype(str) == selected_status]

    mode_options = ["all"]
    if "mode" in filtered.columns:
        available_modes = set(filtered["mode"].dropna().astype(str))
        mode_options += [mode for mode in ["no-odds", "with-odds"] if mode in available_modes]
    selected_mode = st.selectbox(
        "Mode", mode_options, format_func=lambda value: "Tous les modes" if value == "all" else value
    )
    if selected_mode != "all" and "mode" in filtered.columns:
        filtered = filtered[filtered["mode"].astype(str) == selected_mode]

    if "match_profile" in filtered.columns:
        profile_options = ["all"] + sorted(filtered["match_profile"].dropna().astype(str).unique().tolist())
        selected_profile = st.selectbox(
            "Profil de match", profile_options, format_func=lambda value: "Tous les profils" if value == "all" else value
        )
        if selected_profile != "all":
            filtered = filtered[filtered["match_profile"].astype(str) == selected_profile]

    total_predictions = len(filtered)
    status_series = filtered.get("evaluation_status", pd.Series(dtype=str)).astype(str)
    evaluated = filtered[status_series == "evaluated"].copy()
    pending_count = int((status_series == "pending").sum())

    cols = st.columns(6)
    cols[0].metric("Total predictions", total_predictions)
    cols[1].metric("Evaluees", len(evaluated))
    cols[2].metric("Pending", pending_count)
    if evaluated.empty:
        cols[3].metric("Accuracy evaluee", "not_available")
        cols[4].metric("Brier moyen", "not_available")
        cols[5].metric("Log loss moyen", "not_available")
    else:
        accuracy = evaluated["is_correct"].astype(str).str.lower().eq("true").mean()
        mean_brier = pd.to_numeric(evaluated["brier_score_1N2"], errors="coerce").mean()
        mean_log_loss = pd.to_numeric(evaluated["log_loss_match"], errors="coerce").mean()
        cols[3].metric("Accuracy evaluee", percent(float(accuracy)))
        cols[4].metric("Brier moyen", f"{mean_brier:.4f}" if pd.notna(mean_brier) else "not_available")
        cols[5].metric("Log loss moyen", f"{mean_log_loss:.4f}" if pd.notna(mean_log_loss) else "not_available")

    display_columns = [column for column in EVALUATION_TABLE_COLUMNS if column in filtered.columns]
    if not display_columns:
        st.warning("Aucune colonne attendue n'est disponible dans le fichier d'evaluation.")
        return

    st.dataframe(filtered[display_columns], use_container_width=True)


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(page_title="Football Probability Engine", layout="wide")
    inject_ui_styles()
    st.markdown(
        """
        <div class="app-header">
            <h1>Analyse probabiliste football</h1>
            <p>Comparez les moteurs, évaluez la fiabilité et identifiez les matchs les plus lisibles.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        df_features = load_features()
        profiles_analysis = load_match_profiles_analysis()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    historical_tab, upcoming_tab, evaluation_tab = st.tabs(
        ["Explorer les matchs historiques", "Analyser un match à venir", "Evaluation des predictions"]
    )
    with historical_tab:
        render_historical_tab(df_features, profiles_analysis)
    with upcoming_tab:
        render_upcoming_tab(df_features.copy(), profiles_analysis)
    with evaluation_tab:
        render_predictions_evaluation_tab()


if __name__ == "__main__":
    main()
