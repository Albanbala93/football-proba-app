# football-proba-app

Prototype personnel de calcul de probabilites de matchs de football.

Le but du projet est d estimer des probabilites, pas de produire une prediction certaine. Les sorties doivent etre lues comme des signaux probabilistes avec un niveau de confiance variable selon les donnees disponibles.

## Objectif du projet

- Estimation probabiliste de resultats de matchs
- Comparaison de plusieurs moteurs de probabilites
- Analyse du comportement par championnat et par profil de match
- Interface de consultation et de suivi des predictions via Streamlit

## Donnees utilisees

Le projet travaille actuellement sur le Big 5 europeen :

- Premier League
- Ligue 1
- La Liga
- Serie A
- Bundesliga

La source principale est composee de CSV Football-Data places dans `data/raw/`.

La periode effective d analyse depend du contenu de `data/processed/matches_features.csv`, qui est alimente par les fichiers disponibles dans `data/raw/`.

## Architecture principale

```text
app/
  data_pipeline/
    clean_matches.py
    build_features.py
  core/
    probability_engine.py
    probability_engine_no_odds.py
    probability_engine_gb.py
    upcoming_match_predictor.py
  backtesting/
    run_probability_engine_backtest.py
    run_probability_engine_no_odds_backtest.py
    run_probability_engine_gb_backtest.py
    analyze_match_profiles.py
    analyze_league_performance.py
    compare_engines.py
frontend/
  streamlit_app.py
```

## Moteurs disponibles

### logistic_v3

Moteur principal base sur des modeles logistiques.

- meilleur equilibre global
- signal nul plus propre
- interpretable

### no_odds_elo_v1

Moteur sans cotes.

- utilise quand aucune cote reelle n est disponible
- repose davantage sur Elo et les features de forme

### gradient_boosting_v1

Moteur experimental base sur `HistGradientBoostingClassifier`.

- accuracy legerement competitive sur le backtest
- signal nul moins selectif
- conserve pour comparaison avec les autres approches

## Modes de prediction a venir

- `no-odds`
- `with-odds`
- `pseudo-odds`

## Commandes utiles

Rebuild complet :

```powershell
python scripts/rebuild_all.py
```

Health check :

```powershell
python scripts/health_check.py
```

Lancement de l interface :

```powershell
python -m streamlit run frontend/streamlit_app.py
```

Telechargement Big 5 :

```powershell
python scripts/download_big5_data.py --force
```

Comparaison des moteurs :

```powershell
python app/backtesting/compare_engines.py
```

## Regles de prudence

- Ne jamais utiliser de donnees posterieures au match analyse.
- Les probabilites ne sont pas des certitudes.
- Les cotes reelles ameliorent generalement la performance.
- Les pseudo-cotes sont experimentales et doivent etre lues avec prudence.

## Prochaines pistes

- xG via une source externe
- donnees joueurs
- compositions probables
- blessures et suspensions
- calibration plus fine par championnat

## Notes

- Les donnees brutes ne sont pas incluses dans le depot.
- Les sorties de backtest et de comparaison sont ecrites dans `data/predictions/`.
- L interface Streamlit permet de consulter l historique, d analyser un match a venir et de suivre les evaluations des predictions enregistrees.

## Migration

Ce depot a ete migre depuis un backup Google Drive le 2026-08-17. Deux fichiers d audit JSON volumineux (`api_football_coverage_audit_61_2025.json`, `api_football_coverage_audit_61_2024.json`) ont ete exclus de la migration en raison de leur taille. Les variables d environnement requises sont listees dans `.env.example` (a copier vers `.env`, jamais commite).
