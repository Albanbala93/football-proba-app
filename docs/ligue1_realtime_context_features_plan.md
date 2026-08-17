# Ligue 1 Realtime Context Features Plan

## Objectif général

Enrichir au maximum les probabilités Ligue 1 avec des données pre-match actualisees, sans introduire de fuite de donnees.

Le but est de completer le moteur Ligue 1 API xG v2 par une couche de contexte pre-match plus riche, afin d'ameliorer la calibration, la lecture du match et la qualite des recommandations.

## Sources utilisees

### API-Football
- fixtures
- lineups
- injuries
- statistics
- standings si disponible

### Fichiers manuels
- rivalites
- contexte qualitatif
- notes de motivation

### Donnees internes
- xG rolling
- tirs rolling
- stabilite de formation
- historique des resultats

## Features a creer

### Blessures ponderees
- `home_injury_impact_score`
- `away_injury_impact_score`
- `injury_impact_diff`
- `home_missing_starters_count`
- `away_missing_starters_count`

### Enjeu / motivation objective
- `home_rank_before`
- `away_rank_before`
- `home_points_before`
- `away_points_before`
- `home_title_pressure`
- `away_title_pressure`
- `home_europe_pressure`
- `away_europe_pressure`
- `home_relegation_pressure`
- `away_relegation_pressure`
- `motivation_diff`
- `match_context_label`

### Fatigue calendrier
- `home_days_since_last_match`
- `away_days_since_last_match`
- `rest_days_diff`
- `home_matches_last_14_days`
- `away_matches_last_14_days`
- `home_played_europe_last_7_days`
- `away_played_europe_last_7_days`

### Rivalites
- `rivalry_score`
- `is_derby`
- `rivalry_label`

### Contexte qualitatif
- `home_context_score`
- `away_context_score`
- `context_diff`
- `context_label`
- `manual_context_notes`

### Compositions probables / officielles
- `home_expected_formation`
- `away_expected_formation`
- `home_formation_confidence`
- `away_formation_confidence`
- `home_lineup_available`
- `away_lineup_available`
- `home_rotation_score`
- `away_rotation_score`
- `lineup_surprise_score`

## Regle anti-data leakage

Toutes les features doivent etre calculees uniquement avec les informations disponibles avant le match.

Concretement :
- ne jamais utiliser les statistiques du match courant
- ne jamais utiliser le classement calcule avec le match courant
- ne jamais utiliser une lineup officielle si elle n'etait pas disponible avant le coup d'envoi
- ne jamais utiliser une blessure ou une information contextuelle publiee apres la date de prediction

## Logique de mise a jour controlee avant match

La mise a jour doit etre controlee, pas declenchee a chaque clic.

Principes recommandes :
- mise a jour avant journee
- refresh manuel d'un match specifique
- cache local des reponses API
- protection anti rate-limit
- reprise possible sur les fichiers deja telecharges

## Strategie d'ablation avant integration modele

Ne jamais integrer une famille de features sans ablation.

Ordre de test recommande :
1. base API xG v2
2. base + blessures ponderees
3. base + enjeu / motivation
4. base + fatigue calendrier
5. base + rivalites
6. base + contexte qualitatif
7. base + compositions probables / officielles
8. base + all context

Chaque famille doit etre mesuree sur split temporel avec au minimum :
- log_loss
- Brier score multi-classe
- accuracy
- calibration par bucket

## Decision actuelle

Le moteur recommande reste Ligue 1 API xG v2 tant qu'une v3 contextuelle n'a pas prouve un gain net.

Regle de decision :
- conserver la version actuelle si le contexte degrade la calibration
- n'adopter une version contextuelle que si elle ameliore de facon stable le log_loss et le Brier score
- privilegier la robustesse et la lisibilite du signal avant la complexite

