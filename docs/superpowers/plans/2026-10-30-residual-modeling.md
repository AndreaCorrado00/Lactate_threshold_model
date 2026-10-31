# Piano: modello proporzionale con correzione degli scostamenti

Disegno approvato nella conversazione: stima OLS senza intercetta di k sul solo training, delta = target − k × massimo, regressore multivariato regolarizzato/ML su delta, confronto con correzione nulla. La stima di k si ripete anche nei fold interni; nessuna trasformazione globale dei target. Stesso k per baseline e correzione di ogni fold. Ridge iniziale, supporto RF/XGBoost/Bagging; nessun training reale o test holdout in questa implementazione.

1. Branch `feature/residual-modeling`, preservando la cancellazione preesistente del brief.
2. Spostare implementazioni della prima fase in `src/first_model/`; mantenere alias compatibili `src.train`, `src.preprocessing`, ecc. per CLI, import e joblib esistenti. ETL e split rimangono condivisi in `src/`.
3. Implementare `src/residuals/model.py`: modello composto che salva k, preprocessing e regressore; opzioni residual e direct per controllo, k stimato per fit o fisso esplicito esplorativo.
4. Implementare `src/residuals/train.py`: CV esterna persistente, tuning interno per MAE ricostruito, confronto appaiato con proporzionale; default no refit, `--refit` esplicito. Output in W e delta, metriche fold/OOF, Q_delta, audit k/fit, importanza della correzione su covariate originali, grafici e report.
5. Configurazione e documentazione separate per fasi; mantenere i risultati del primo modello invariati e recuperabili.
6. Test prima dell'implementazione: k/preprocessing non dipendono dalla validation, baseline condivisa, ricostruzione corretta, nessun accesso al test, tuning annidato, modelli alternativi, serialization/refit e retrocompatibilità.
7. Suite completa, smoke CLI e caricamento del vecchio Ridge su train; nessuna lettura del test reale.

Metriche: MAE/RMSE/R² sul target originale per entrambe le procedure, MAE/RMSE/R² di delta per correzione e zero, Q_delta = 1 − SSE_corretto / somma(delta²), correlazione delta osservato/predetto (null se costante), bias e miglioramenti per fold. Q_delta distinto dal normale R²; somme nulle riportate come null. Con k diverso fra fold la delta OOF è definita rispetto al riferimento del suo fold.

Non sono previsti PCA, forward selection o scelta automatica del modello finale in questa fase. Output separati dalle run Ridge; si conserva il carattere esplorativo del train e del test già osservato.
