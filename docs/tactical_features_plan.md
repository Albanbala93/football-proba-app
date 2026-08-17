# Tactical Features Plan

## Objectif

Documenter la future couche tactique du modèle `football-proba-app`.

Cette couche doit enrichir les probabilités avec des informations sur les formations, les compositions, les absences et les rapports de force structurels entre équipes.

## 1. Pourquoi Ajouter Les Schémas Tactiques

Les schémas tactiques peuvent apporter un signal complémentaire au modèle actuel.

Objectifs principaux :

- mieux comprendre le style réel des équipes ;
- enrichir les probabilités au-delà des résultats, Elo, tirs et cotes ;
- identifier des rapports de force structurels entre équipes.

Une équipe qui joue régulièrement avec trois défenseurs centraux, deux pistons et deux attaquants ne produit pas le même profil qu'une équipe en `4-3-3` ou en `4-2-3-1`. Ces informations peuvent aider le modèle à mieux représenter le contexte du match, surtout lorsqu'elles sont combinées avec les signaux déjà présents.

## 2. Limite Importante

Une formation ne gagne pas mécaniquement contre une autre.

Points à garder en tête :

- un `4-3-3` peut être offensif ou prudent selon les joueurs et les consignes ;
- un `3-5-2` peut défendre bas ou presser haut ;
- un `5-4-1` peut être une vraie ligne de cinq ou une structure asymétrique selon les phases de jeu ;
- la formation doit être combinée avec stabilité, joueurs, contexte et adversaire.

La formation brute ne doit donc jamais être interprétée comme une règle déterministe. Elle doit être traitée comme une variable contextuelle, utile seulement si elle améliore les métriques en backtest.

## 3. Données Nécessaires

La couche tactique nécessite des données pré-match ou historiques structurées :

- formations de départ ;
- lineups ;
- positions des joueurs ;
- titulaires/remplaçants ;
- absents/blessés/suspendus ;
- source de la donnée : officielle / probable / estimée ;
- horodatage de disponibilité de la donnée.

L'horodatage est important pour éviter d'utiliser une information connue uniquement après coup dans un modèle censé prédire avant le match.

## 4. Sources Possibles

Sources candidates :

- `Sportmonks` ;
- `API-Football` ;
- `StatsBomb Open Data` pour prototypage ;
- éventuelle saisie manuelle temporaire pour tester quelques matchs.

La saisie manuelle peut être utile pour valider le schéma de données et les premières transformations avant de payer ou d'intégrer une API complète.

## 5. Tables À Créer

Tables prévues :

- `data/external/lineups/lineups.csv`
- `data/processed/match_formations.csv`
- `data/processed/tactical_features.csv`

Rôle des tables :

- `lineups.csv` stocke les données source de composition et de joueurs.
- `match_formations.csv` agrège les formations par match et par équipe.
- `tactical_features.csv` contient les variables prêtes à joindre dans `build_features.py`.

## 6. Schéma Proposé Pour lineups.csv

Colonnes :

- `match_id`
- `date`
- `league`
- `season`
- `team`
- `opponent`
- `is_home`
- `formation`
- `player_id`
- `player_name`
- `position`
- `is_starter`
- `lineup_source`

Valeurs possibles pour `lineup_source` :

- `official_lineup`
- `expected_lineup`
- `estimated_lineup`

Une extension future devrait ajouter `lineup_available_at` pour tracer précisément quand l'information est devenue disponible.

## 7. Features V1

Première version, volontairement simple et mesurable :

- `home_formation`
- `away_formation`
- `formation_matchup`
- `home_defenders_count`
- `away_defenders_count`
- `home_midfielders_count`
- `away_midfielders_count`
- `home_forwards_count`
- `away_forwards_count`
- `midfield_density_diff`
- `attacking_line_diff`
- `defensive_line_diff`
- `home_formation_stability_5`
- `away_formation_stability_5`

Définitions candidates :

- `formation_matchup` : combinaison catégorielle, par exemple `4-3-3_vs_3-5-2`.
- `midfield_density_diff` : différence entre milieux domicile et milieux extérieur.
- `attacking_line_diff` : différence entre attaquants domicile et attaquants extérieur.
- `defensive_line_diff` : différence entre défenseurs domicile et défenseurs extérieur.
- `formation_stability_5` : fréquence de la formation actuelle sur les 5 derniers matchs disponibles de l'équipe.

## 8. Features V2

Deuxième version, à envisager seulement après validation de l'impact V1 :

- `width_advantage`
- `wingback_presence`
- `double_pivot`
- `back_three`
- `back_five`
- `two_strikers`
- `matchup_style_score`
- `tactical_mismatch_score`

Ces variables demandent plus d'interprétation que les counts simples. Elles doivent être définies précisément avant entraînement pour éviter des signaux arbitraires ou difficiles à reproduire.

## 9. Règle Anti-Data Leakage

Pour entraîner un modèle de prédiction pré-match, ne jamais utiliser une information qui n'était pas disponible avant le match.

Règles pratiques :

- distinguer `official_lineup`, `expected_lineup` et `estimated_lineup` ;
- stocker `lineup_available_at` si possible ;
- ne pas utiliser une composition officielle publiée après l'heure supposée de prédiction ;
- ne pas utiliser la formation réellement observée après coup pour simuler une prédiction pré-match ;
- backtester avec le même niveau d'information que celui disponible en production.

Cette règle est critique. Sans elle, le modèle peut apprendre une information postérieure au coup d'envoi et produire des métriques trop optimistes.

## 10. Roadmap

Étapes proposées :

1. Choisir la source API.
2. Récupérer les lineups historiques.
3. Générer `match_formations.csv`.
4. Générer `tactical_features.csv`.
5. Intégrer les features tactiques dans `build_features.py`.
6. Entraîner modèle avec/sans tactique.
7. Comparer les métriques.
8. Intégrer la lecture tactique dans Streamlit.

## 11. Recommandation MVP

Pour un MVP pragmatique :

- commencer par formation estimée ou officielle historique ;
- tester d'abord sur un seul championnat ;
- comparer l'impact réel avant généralisation Big 5.

Le critère de décision doit rester mesurable : amélioration ou non des métriques de backtest, stabilité par championnat, et absence de data leakage.
