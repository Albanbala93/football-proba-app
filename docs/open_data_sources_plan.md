# Open Data Sources Plan

## Objectif

Documenter la stratégie open source / quasi-open data pour enrichir `football-proba-app` sans dépendre immédiatement d'un abonnement premium.

L'approche recommandée est pragmatique : conserver les CSV actuels comme socle, ajouter en priorité les sources qui améliorent le signal Big 5, et réserver les API premium aux données difficiles à obtenir autrement.

## 1. Football-Data.co.uk / CSV Actuels

Usage :

- résultats historiques ;
- cotes ;
- statistiques match ;
- base principale du modèle actuel.

Rôle dans le projet :

Football-Data.co.uk reste la source structurante actuelle. Les fichiers CSV alimentent le nettoyage, la construction des features, les backtests et les modèles principaux.

Points forts :

- couverture Big 5 déjà en place ;
- historique exploitable ;
- cotes disponibles selon les saisons/fichiers ;
- intégration déjà fonctionnelle dans le projet.

Limites :

- pas de xG natif ;
- pas de lineups détaillés ;
- pas de formations tactiques ;
- granularité limitée au niveau match.

## 2. football-data.org

Usage :

- fixtures ;
- calendriers ;
- compétitions ;
- équipes ;
- vérification saison active.

Limite :

- dépend du plan API ;
- moins prioritaire car les CSV actuels couvrent déjà beaucoup.

Rôle potentiel :

football-data.org peut servir de source complémentaire pour vérifier les calendriers, les équipes actives et les compétitions. Son intérêt principal serait opérationnel, notamment pour fiabiliser les matchs à venir ou la saison active.

## 3. OpenFootball

Usage :

- fallback open source résultats/calendriers ;
- validation de calendriers et noms d'équipes.

Limite :

- pas de xG ;
- pas de tactique ;
- pas de lineups détaillés.

Rôle potentiel :

OpenFootball peut être utile comme source de contrôle ou de secours, notamment pour vérifier les noms d'équipes et certains calendriers historiques. Ce n'est pas une source prioritaire pour améliorer la performance modèle.

## 4. StatsBomb Open Data

Usage :

- laboratoire de données événementielles ;
- lineups ;
- events ;
- 360 data sur certains matchs ;
- prototypage features tactiques fines.

Limite :

- couverture partielle ;
- pas Big 5 complet récent.

Rôle potentiel :

StatsBomb Open Data est très utile pour prototyper des features avancées : occupation des zones, séquences, pression, tirs, positions, structures avec et sans ballon, et données événementielles. En revanche, sa couverture ne permet pas de l'utiliser directement comme socle Big 5 récent.

## 5. Understat

Usage :

- xG Big 5 ;
- xGA ;
- xG rolling ;
- xG diff ;
- over/underperformance ;
- enrichissement prioritaire du modèle Big 5.

Limite :

- vérifier méthode d'accès propre ;
- mapping noms équipes nécessaire.

Rôle recommandé :

Understat est la priorité open/quasi-open la plus intéressante pour améliorer le modèle Big 5. Les features xG peuvent apporter un signal plus fin que les résultats bruts, les tirs simples ou l'Elo.

Features candidates :

- `home_xg_for_5`
- `away_xg_for_5`
- `home_xg_against_5`
- `away_xg_against_5`
- `xg_diff_between_teams`
- `home_xg_overperformance_5`
- `away_xg_overperformance_5`
- `home_xga_trend`
- `away_xga_trend`

## Roadmap Recommandée

1. Priorité 1 : Understat xG Big 5.
2. Priorité 2 : football-data.org fixtures si besoin.
3. Priorité 3 : StatsBomb Open Data pour prototypage event/tactique.

## Synthèse

La meilleure trajectoire court terme est d'enrichir le modèle Big 5 avec des données xG Understat, car elles ciblent directement les championnats déjà modélisés.

StatsBomb Open Data doit être utilisé comme environnement de recherche, pas comme source de production Big 5.

Le projet s'appuie par ailleurs sur API-Football pour l'enrichissement Ligue 1 (xG, compositions, contexte de match). Voir `docs/ligue1_realtime_context_features_plan.md`.
