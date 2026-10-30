# Linee guida del repository

## Lingua e ambito

Usare l’italiano nelle comunicazioni e nella documentazione; mantenere gli identificatori tecnici coerenti con il codice. `brief_ml_soglia_lattacida.md` definisce requisiti e criteri di accettazione. Lo studio è retrospettivo/esplorativo: non attribuire validità clinica alle predizioni né equiparare `T_Lac` a IAT, LT1, LT2 o MLSS.

## Struttura del progetto

Il repository contiene attualmente il brief, senza codice applicativo o test. Organizzare l’implementazione prevista così:

- `src/`: download, ETL, costruzione del dataset e modellazione.
- `data/raw/`: dati originali, mai da versionare.
- `reports/`: `data_audit.md`, `results.md` e grafici.
- `results/metrics.csv`: metriche esportate.
- `tests/`: verifiche automatiche.

Aggiungere `README.md` con istruzioni riproducibili e `requirements.txt` oppure `pyproject.toml` con versioni delle dipendenze. Escludere dati grezzi e `.venv/` in `.gitignore` prima dello staging.

## Comandi di sviluppo

Non esistono ancora comandi configurati per addestramento o test. Per preparare l’ambiente in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Dopo aver creato `requirements.txt`, usare `python -m pip install -r requirements.txt`. Documentare nel README i comandi effettivi per audit, addestramento e valutazione.

## Stile e test

Usare quattro spazi in Python, `snake_case` per funzioni/moduli e `PascalCase` per classi. Esplicitare le unità nelle colonne, ad esempio `P_TLac_W` e `mass_kg`. Separare preparazione dati e valutazione.

Non sono configurati formatter, linter, framework di test o soglie di copertura. Preferire pytest, file `tests/test_*.py` e funzioni `test_*`; una volta configurato, eseguire `python -m pytest`. Verificare estrazione, unità, validità del target e isolamento delle partizioni con dati sintetici.

## Metodo e riproducibilità

Verificare l’integrità di `data_v2.zip` e produrre l’audit prima della modellazione. Costruire una riga per test ciclistico; documentare target, esclusioni, colonne disponibili e definizione dei set A/B, non dettagliati nel brief aggiornato. Usare solo covariate non derivate dal lattato.

Applicare split 80/20, separando per atleta quando possibile; altrimenti dichiarare il rischio di test ripetuti. Fissare i seed; adattare imputazione, scaling e tuning esclusivamente sul train. Confrontare DummyRegressor, Ridge e RandomForestRegressor; XGBoost è opzionale. Mantenere identiche feature e partizioni nel confronto tra modelli.

Riportare MAE, RMSE, R², intervalli bootstrap e grafici richiesti dal brief; confrontare A/B. Usare W/kg solo con massa disponibile. Se target, covariate o numerosità sono inadeguati, fermarsi all’audit senza inventare risultati.

## Commit e pull request

Non esiste ancora una convenzione storica. Usare oggetti brevi all’imperativo, ad esempio `Aggiungi audit dei dati ciclistici`. Descrivere nelle PR scopo, verifiche eseguite, eventuali issue collegate e modifiche metodologiche; allegare metriche o grafici quando cambia la valutazione.

## Regolo di implementazione
1. non implementare codice se non richiesto
