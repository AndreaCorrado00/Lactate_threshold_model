# Uso e riproduzione di Ridge

Configurazione: `configs/first_model/training_ridge.json`. Train QC da 222 record, 20 covariate; potenza massima, step, cadenza e variabili derivate dal lattato esclusi. Sesso codificato F=0/M=1 e standardizzato insieme alle altre covariate.

Mediane per sesso → StandardScaler → Ridge. Target in W. Ogni fold interno ed esterno stima il preprocessing solo sul proprio training. Cinque fold esterni, quattro interni. Alpha candidati: 0,0001; 0,001; 0,01; 0,1; 1; 10; 100; 1000; 10000; 100000. Dieci configurazioni coprono tutti i candidati; scelta per MAE interno. Nessuna PCA o forward selection. Ricerca ripetuta sul train completo per il refit; alpha finale 0,1.

## Inferenza

Caricare soltanto il file joblib fidato prodotto localmente. Le righe devono avere le covariate elencate nel JSON e `gender` F/M; mancanti numerici rappresentati da `None`, non chiavi assenti.

```python
import joblib

bundle = joblib.load('results/runs/ridge_final/model.joblib')
# rows: lista di dizionari con le 20 covariate richieste.
predicted_W = bundle['model'].predict(bundle['preprocessor'].transform(rows))
```

Non rifittare imputer/scaler sui nuovi record o standardizzare manualmente una seconda volta. Il target non serve per inferenza. Usare le unità originali: altezza m, massa kg, potenze W, frequenze cardiache bpm. Le unità ematologiche non sono dichiarate dalla fonte.

## Coefficienti e diagnostica

`reports/ridge/coefficients.csv` contiene coefficienti standardizzati e nelle unità originali, medie e scale del train. Dopo imputazione:

`P_pred = intercetta_standardizzata + somma(coef_standardizzato[j] * (x[j] - media[j]) / scala[j])`.

Intercetta standardizzata: 272,63964 W. Intercetta della formula equivalente nelle unità originali: −433,12935 W. L'equivalenza è verificata numericamente sulle predizioni test. L'intercetta non ha significato fisiologico fuori dal dominio osservato. I coefficienti sono condizionati alle altre covariate, non effetti causali; collinearità può produrre segni opposti. Per il sesso il coefficiente standardizzato non coincide con la differenza grezza F/M. Importanze di permutazione OOF nella run.

Il report usa errore = predetto − osservato. I CSV originali train/evaluate mantengono `residual_W = osservato − predetto`. I nomi delle colonne e le etichette distinguono i segni.

## Valutazione

Comandi nel notebook `README.ipynb`. Richiede refit completo, controlla hash, provenienza e disgiunzione degli ID anche rispetto al train originale da 226 record. Solo transform/predict, nessuna esclusione automatica o correzione sul test. Bootstrap percentili al 95%, 2000 campioni, seed 42: condizionato al modello e per riga, non intervalli individuali o incertezza della selezione.

Il test era già stato osservato durante lo sviluppo. Ogni ulteriore scelta guidata dai suoi risultati richiederà nuovi dati per una conferma indipendente.

## Archivio

Tentativi precedenti e PCA: `archive/experiments_20261030_165918/`, inventario `inventory.json`, interamente ignorati da Git. Alcuni percorsi nelle configurazioni storiche si riferiscono alla disposizione precedente e vanno adattati prima del recupero. La prima fase mantiene soltanto i risultati Ridge; il codice riutilizzabile e i suoi test sintetici rimangono disponibili. La fase separata degli scostamenti è descritta in [docs/residuals.md](../residuals.md), con [risultati OOF](../../reports/archive/residuals/results.md).
