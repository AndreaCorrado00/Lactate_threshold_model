# Predizione di P_TLac con Ridge

Modello corrente: **Ridge con scaling**, 20 covariate originali, senza PCA o forward selection. Target `P_TLac_W`, in watt. Studio retrospettivo/esplorativo, senza validita clinica attribuita o equivalenza con IAT/LT1/LT2/MLSS.

| Partizione | n | MAE (W) | RMSE (W) | R2 |
|---|---:|---:|---:|---:|
| Validation OOF | 222 | 15,93 | 19,89 | 0,810 |
| Test | 57 | 12,69 | 16,57 | 0,867 |

Il test era gia stato osservato durante lo sviluppo: questa valutazione non costituisce una nuova conferma indipendente della selezione di Ridge. [Report completo](reports/results.md).

## File attivi

- `configs/training_ridge.json`: unica configurazione di training attiva.
- `results/runs/ridge_final/`: unica run attiva, CV, tuning, refit e `model.joblib`.
- `results/test_ridge/`: unica valutazione test attiva, metriche, predizioni e grafici interattivi.
- `results/metrics.csv`: riepilogo CV/test.
- `reports/results.md` e `reports/ridge/`: analisi, coefficienti, sottogruppi e rapporti potenza/massima.
- `reports/data_audit.md`: coorte ed esclusioni.
- `docs/ridge.md`: riproduzione e interpretazione.
- `archive/`: tentativi precedenti, PCA inclusa, ignorati da Git.

Il codice generico e i test sintetici sono conservati per evitare riscritture inutili; non rappresentano altre run attive. La modellizzazione degli scostamenti individuali non e stata avviata.

## Ambiente

Windows, Python 3.14; dipendenze fissate in `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

## Training e refit

Dalla root, sul train QC gia predisposto:

```powershell
.\.venv\Scripts\python.exe -m src.train --config configs/training_ridge.json --model ridge --output results/runs/ridge_final --refit
```

Run gia completata: il comando rifiuta la sovrascrittura. Per ripetizioni deliberate usare una nuova cartella. Senza `--refit` si esegue solo CV. Cinque fold esterni persistenti, quattro interni, seed 42, dieci alpha; imputazione per sesso e scaling sempre stimati sui dati di fit. Alpha finale: **0,1**.

## Valutazione congelata

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --run results/runs/ridge_final --holdout-dir data/splits/v1 --output results/test_ridge --allow-test-access
```

Valutazione gia completata; non rieseguirla automaticamente o usare il test per ulteriore tuning. Il comando verifica hash e ID disgiunti, applica solo transform/predict e rifiuta output preesistente. Nuove scelte guidate dal test richiederanno nuovi dati indipendenti per una conferma conclusiva.

## Dati e archivio

Originali in `data/raw/`, manifest di estrazione `data/processed/cycling.json`, train corrente `data/splits/v2_train_qc/`, test originale `data/splits/v1/`. Non rigenerare lo split. Le righe sono test, non atleti univocamente identificati.

`data/`, `.venv/`, `archive/` e `results/runs/` sono ignorati da Git. Il modello binario resta locale; configurazione, codice, documentazione e risultati finali esportati sono disponibili per il versionamento. Nessun commit o staging eseguito durante la riorganizzazione.
