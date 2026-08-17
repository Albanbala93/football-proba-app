# Prototype d'agregation Understat player-level vers match-level

## 1. Contexte

Understat ne fournit pas toujours une table match-level unique dans les exports disponibles. Les exports disponibles peuvent etre structures comme un fichier par equipe et par match.

Pour utiliser le xG dans le modele, il faut donc agreger ces fichiers player-level afin d'obtenir une ligne par match, avec les valeurs domicile et exterieur normalisees.

## 2. Fichiers utilises

- `data/external/understat/player_match_stats/home.xlsx`
- `data/external/understat/player_match_stats/away.xlsx`
- `data/external/understat/match_player_tables_index.csv`

## 3. Script cree

- `scripts/aggregate_understat_player_match_tables.py`

Le script lit les fichiers listes dans l'index, supporte les formats `.csv` et `.xlsx`, agrege les statistiques par equipe, puis pivote les donnees en une ligne par match.

## 4. Sorties generees

- `data/processed/understat_team_match_aggregates.csv`
- `data/processed/understat_matches_normalized.csv`

## 5. Resultat du test

Match : Brest vs Rennes

| Champ | Valeur |
| --- | --- |
| Date | 2025-08-16 |
| Ligue | Ligue 1 |
| Saison | 2025-2026 |
| Score | Brest 3 - 4 Rennes |
| home_xg | 1.96 |
| away_xg | 2.80 |
| home_shots | 10 |
| away_shots | 15 |
| home_xa | 1.32 |
| away_xa | 0.60 |
| home_players_count | 13 |
| away_players_count | 14 |

## 6. Conclusion

Le pipeline d'agregation fonctionne. Il permet de convertir des exports Understat player-level en donnees match-level exploitables.

Ce prototype n'est pas encore integre au modele principal. L'integration du xG necessitera davantage de matchs pour calculer des rolling features fiables.

## 7. Prochaine etape

- Collecter un echantillon plus large.
- Automatiser ou semi-automatiser l'index.
- Generer les features xG rolling.
- Backtester avec et sans xG.
