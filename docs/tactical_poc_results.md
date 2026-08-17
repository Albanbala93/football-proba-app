# Tactical POC Results

## 1. Contexte

Objectif : tester si les features tactiques ajoutent un signal utile au modèle de probabilités football.

Source : Sportmonks.

Ligue test : Scottish Premiership.

Données récupérées :

- formations ;
- lineups ;
- sidelined ;
- statistics.

Ce POC est volontairement isolé du moteur principal Big 5. Il sert à valider si les données tactiques peuvent apporter de l'information mesurable avant une intégration plus large.

## 2. Volume

- Fixtures récupérées : 228
- Matchs avec résultats : 222
- Split temporel : 177 train / 45 test

## 3. Features Tactiques Générées

Features principales :

- `home_formation`
- `away_formation`
- `formation_matchup`
- `defenders_count`
- `midfielders_count`
- `forwards_count`
- `midfield_density_diff`
- `attacking_line_diff`
- `defensive_line_diff`
- `back_three` / `back_four` / `back_five`
- `two_strikers`
- `formation_stability_5`

Les features de stabilité sont calculées sans data leakage : uniquement sur les matchs strictement antérieurs de chaque équipe.

## 4. Analyse Descriptive

- Top formation globale : `4-2-3-1`, 148 occurrences, 32.46 %
- Équipe la plus stable : Falkirk, stabilité moyenne 0.9135
- Équipe la moins stable : Kilmarnock, stabilité moyenne 0.1743
- Meilleure formation domicile min 5 : `4-3-3`, `home_win_rate` 0.6875
- Meilleure formation extérieur min 5 : `3-1-4-2`, `away_win_rate` 0.6667

## 5. Résultats Modèle

`baseline_simple` :

- accuracy : 0.2667
- log_loss : 1.2333
- brier : 0.2479

`tactical_model` :

- accuracy : 0.3556
- log_loss : 1.2137
- brier : 0.2418

## 6. Conclusion

La couche tactique améliore accuracy, log loss et Brier sur cette ligue test.

Le signal semble donc exploitable.

L'échantillon reste limité, avec une seule ligue et une seule saison test. Les résultats doivent être confirmés sur plusieurs ligues et plusieurs saisons avant toute intégration au moteur principal.

Pour le Big 5, il faut un accès Sportmonks adapté ou une autre source équivalente.

## 7. Décision Recommandée

- Ne pas intégrer encore au moteur principal Big 5.
- Conserver le POC tactique.
- Si accès Big 5 disponible, prioriser l'ingestion tactique Ligue 1 / Premier League.
