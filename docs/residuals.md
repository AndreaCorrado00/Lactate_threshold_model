# Seconda fase: modellazione degli scostamenti

Framework sviluppato su `feature/residual-modeling`, ora presente nel working tree di `main`. Questa fase confronta, nelle stesse validation, il proporzionale senza intercetta con una correzione multivariata. Il primo modello Ridge, i suoi risultati e la sua configurazione restano separati e invariati. La prima run Ridge è stata successivamente eseguita in sola validation: [risultati e conclusioni](../reports/archive/residuals/results.md).

## Formulazione

In ogni fit locale: `k = somma(P_max * P_TLac) / somma(P_max²)`, poi `delta = P_TLac - k * P_max`. Il regressore viene addestrato su delta; `P_pred = k * P_max + delta_pred`. Il riferimento usa lo stesso k, con correzione nulla. `P_max` deve essere osservata, positiva e finita, anche in inferenza; non viene imputata o inclusa nelle feature della correzione.

Questa è una regressione con offset stimato in un primo stadio, non una regressione parziale FWL o una stima congiunta di tutti i coefficienti. La Ridge della correzione mantiene la propria intercetta: un eventuale miglioramento può comprendere ricalibrazione media, non necessariamente differenze individuali. [Documentazione glmnet sugli offset](https://glmnet.stanford.edu/articles/glmnet.html).

## Configurazione

`configs/residuals/training.json`: train QC 222 record; stesse covariate ammesse della prima Ridge, senza potenza massima, step o lattato. Supporta `ridge`, `random_forest`, `xgboost`, `bagging` mediante gli stessi costruttori controllati della prima fase.

- `target_mode: residual`: stima gli scostamenti (default).
- `target_mode: direct`: controllo sulla potenza originale, con lo stesso confronto proporzionale. La correzione esportata è allora predetto meno baseline; non è un fit su delta.
- `proportional: {"mode": "fit"}`: stima k in ogni training locale, anche nei fold interni (default robusto rispetto alla contaminazione fra partizioni).
- `proportional: {"mode": "fixed", "k": 0.7986}`: variante esplicita esplorativa. Se k proviene dal train completo, tale dipendenza va dichiarata; non è il default.
- `scaling`: se omesso, true per Ridge e false per gli alberi; booleano esplicito per override. Mediane per sesso e scaling sono stimati in ogni fit locale.
- `tuning`: stessa grammatica della prima fase; quattro fold interni e fino a dieci configurazioni, selezione per MAE medio della potenza ricostruita. Con lo stesso offset, il MAE coincide con quello di delta.
- `importance_repeats`: permutazioni delle feature originali della validation, preprocessing e baseline congelati. La permutazione del sesso può cambiare le mediane applicate ai mancanti; misura il contributo alla pipeline completa.

Nessuna PCA o forward selection: chiavi relative rifiutate per non mescolare procedure. `tuning: null` usa i parametri di partenza senza ricerca. Seed e numero di thread controllati; output preesistente rifiutato.

## Isolamento

Cinque fold esterni persistenti, tutti i 222 record in OOF. Per ogni candidato nei quattro fold interni si stimano k, mediane, scaler e regressore sui soli dati di training interno. Poi la scelta viene rifittata sul training esterno ed applicata alla validation esterna. Non si salva o riutilizza un unico target delta costruito sul train completo prima della CV.

Il k della validation è quello del suo training, mai stimato sui target della validation. I fold hanno perciò riferimenti leggermente diversi: la delta OOF è definita rispetto al riferimento del proprio fold. L'audit registra ID train/validation e k anche internamente.

## Comandi dalla root

Solo validation, Ridge:

```powershell
.\.venv\Scripts\python.exe -m src.residuals.train --config configs/residuals/training.json --model ridge --output results/residuals/runs/ridge_residual_01
```

Controllo diretto, senza cambiare il file:

```powershell
.\.venv\Scripts\python.exe -m src.residuals.train --config configs/residuals/training.json --model ridge --target-mode direct --output results/residuals/runs/ridge_direct_01
```

Gli altri modelli usano lo stesso comando con `--model random_forest`, `xgboost` o `bagging` e nuove cartelle di output. Nessun modello viene lanciato automaticamente. La scelta tra modelli va trattata come nuova esplorazione; non ampliare la ricerca in base al test già osservato.

Con `--refit` si ripete il tuning sul train completo e si salva il predittore composto in `model.joblib`; senza flag non viene salvato alcun modello completo. Questa nuova fase non offre un comando di valutazione sul holdout: non usare `src.first_model.evaluate` con questi bundle.

Inferenza dopo un eventuale refit:

```python
import joblib
bundle = joblib.load('results/residuals/runs/ridge_residual_refit/model.joblib')
# rows richiede gender, covariate originali e max_power_W; target non richiesto.
baseline_W, delta_W, predicted_W = bundle['predictor'].predict_parts(rows)
```

Caricare soltanto file joblib fidati. Nessuna trasformazione da rifittare in inferenza.

## Risultati e criteri di lettura

- `predictions_oof.csv`: una riga per ID, fold, k, potenza massima, osservato, baseline, delta osservata/predetta, potenza ricostruita ed errore con segno predetto meno osservato.
- `metrics.csv`, `metrics_folds.csv`: MAE e RMSE in W, R² adimensionale, sul target originale e su delta per correzione nulla e modello.
- `gains_folds.csv`: miglioramenti appaiati per fold, Q_delta e correlazione. Guadagno positivo = errore minore del proporzionale.
- `summary.json`: metriche OOF, provenienza, configurazione, versioni e stato del refit.
- `folds.json`: ID delle partizioni, k, preprocessing e audit tuning. `tuning_fold_*/` conserva i trial progressivamente.
- `importance.csv`, `importance_folds.csv`: permutation importance delle covariate originali; non include il contributo di P_max, che rimane fisso.
- `validation.html`, `report.md`: confronto potenza e scostamenti. `tuning_refit.json` e `model.joblib` solo su richiesta di refit.

`Q_delta = 1 - somma((delta - delta_pred)²)/somma(delta²)`: positivo indica un miglioramento quadratico rispetto alla correzione nulla. È distinto da R², che usa la media di delta. Per denominatore nullo Q_delta è null; per serie costanti la correlazione è null. Non limitare o nascondere valori negativi.

La potenza ricostruita può mantenere un buon R² anche se gli scostamenti non sono prevedibili: leggere insieme guadagni, Q_delta, correlazione e dispersione delle correzioni. Coerenza fra fold più informativa di un piccolo guadagno aggregato isolato. Non è una stima della varianza intra-individuale; ID non garantiscono atleti distinti. Non attribuire causalità o validità clinica.

## Prima fase e compatibilità

Implementazioni storiche in `src/first_model/`; ETL, audit e split restano in `src/` come strumenti condivisi. I vecchi moduli `src.train`, `src.preprocessing`, ecc. sono piccoli alias: preservano CLI, import e caricamento dei joblib già prodotti, senza duplicare le implementazioni. Configurazione e procedura diretta in `configs/first_model/` e `docs/first_model/`.

Le run Ridge e i report della prima fase non sono stati spostati o riscritti. Le nuove run stanno esclusivamente in `results/residuals/runs/`, ignorate da Git. Il test già osservato resta fuori da questa implementazione; una conferma finale richiederà nuovi dati indipendenti.
