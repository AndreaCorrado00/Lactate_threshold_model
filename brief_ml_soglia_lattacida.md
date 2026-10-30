# Brief di progetto — Predizione della potenza alla soglia lattacida senza misurare il lattato

## Obiettivo
Sviluppare e valutare una pipeline riproducibile che predica la **potenza alla soglia lattacida** (`P_TLac`, W; secondariamente W/kg) in test incrementali di ciclismo, utilizzando **solo covariate non derivate dal lattato**. Confrontare una regressione lineare regolarizzata con Random Forest e XGBoost, senza ricercare architetture complesse.

**Ambito:** studio retrospettivo/esplorativo, non uno strumento clinico né una validazione di LT1/LT2 universali.

## Riferimenti
1. **Römer C., Wolfarth B. (2023)**, *Prediction of Relevant Training Control Parameters at Individual Anaerobic Threshold without Blood Lactate Measurement*, *International Journal of Environmental Research and Public Health*, 20(5), 4641. https://doi.org/10.3390/ijerph20054641 — regressione multipla per stimare FC e W/kg alla **IAT secondo Dickhuth** su 1.234 test; riferimento metodologico, **non replica diretta** del target.
2. **Donne B. et al. (2024)**, *Graded Incremental Test Data (Cycling, Running, Kayaking, Rowing): an open access dataset*, Zenodo v2. https://doi.org/10.5281/zenodo.10841412 — dataset e definizione operativa del target.

## Dataset
- Fonte: https://zenodo.org/records/10841412
- File: `data_v2.zip` (~6,7 MB), versione v2 del 19/03/2024.
- Totale dichiarato: **835 file di test**, **285 di ciclismo**, 266 rowing/kayaking, 284 running. **Un test non implica un atleta unico**.
- Nei test ciclistici: protocollo incrementale, step di 3 minuti (20/30/40 W), FC, VO2 e lattato per step; covariate antropometriche, altri parametri cardiorespiratori e indici derivati.
- La fonte definisce `T_Lac` come **punto d'intersezione di due rette** adattate con regressione segmentata alla curva lattato–carico. Non assumere che `T_Lac` coincida con IAT di Dickhuth, LT1, LT2 o MLSS.

## Domanda e ipotesi
**RQ:** si può stimare `P_TLac` senza usare le concentrazioni di lattato, con errore inferiore a una baseline semplice?

- **H0:** i modelli ML non riducono l'errore predittivo rispetto alla regressione regolarizzata.
- **H1:** almeno un modello non lineare riduce MAE/RMSE su test non visti, a parità di covariate e partizioni.
- **Analisi secondaria:** la disponibilità delle misure di VO2 migliora la predizione rispetto a variabili antropometriche e FC?

## Unità di analisi, feature e target
- **Una riga per test ciclistico.**
- **Target primario:** `P_TLac_W` riportato dagli autori oppure ricalcolato dalla curva lattato-potenza usando il metodo pubblicato, se necessario e riproducibile.
- **Target secondario:** `P_TLac_W_kg = P_TLac_W / mass_kg` (solo se la massa è disponibile).

## Procedura operativa ad alto livello
1. Scaricare `data_v2.zip` da Zenodo, verificare integrità e decomprimere in `data/raw/` (mai versionare i dati grezzi).
2. Produrre `reports/data_audit.md`: struttura file/fogli, dizionario delle colonne, conteggio test ciclistici, identificativi atleta disponibili, missingness, range e unità, metodi usati per determinare la soglia.
3. Estrarre solo `cycling`, costruire tabella `one-row-per-test`. Se il target non è disponibile, ricalcolarlo dalla curva lattato-carico mediante regressione segmentata e documentare criteri di qualità; scartare test per cui la soglia non sia identificabile.
4. Costruire feature set A e B **solo con variabili realmente disponibili**; definire criteri di esclusione prima di addestrare. Non interpolare o imputare utilizzando il test set.
5. Suddividere train/test (80/20); se esiste un identificativo stabile di atleta, separare **per atleta** con `GroupShuffleSplit`/`GroupKFold`; altrimenti dichiarare il rischio di campioni ripetuti e limitarne l'interpretazione.
6. Pipeline scikit-learn con imputazione e scaling fit sul solo train, modelli: DummyRegressor (media), Ridge e RandomForestRegressor; XGBoost opzionale. Stesso split e stesse feature per tutti; tuning minimo esclusivamente sul train.
7. Riportare MAE (W), RMSE (W), R² e grafico osservato–predetto; per il target normalizzato MAE/RMSE in W/kg. Intervalli di confidenza via bootstrap sul test, raggruppato per atleta se possibile.
8. Confrontare A vs B e modelli, commentando guadagno rispetto a Dummy/Ridge, dimensione campione e validità esterna. Non confrontare direttamente i valori di R² dello studio Römer come se il target fosse identico.

## Deliverable minimi
- `README.md` con istruzioni di riproduzione (`python -m ...` o notebook).
- `src/` (download/ETL, dataset, modeling), `requirements.txt` o `pyproject.toml` con versioni.
- `reports/data_audit.md`, `reports/results.md`, `results/metrics.csv`.
- Grafici: target distribution, observed-vs-predicted, residuals, confronto modelli e confronto feature set.
- Seed fissato, niente dati grezzi nel repository.

## Criteri di accettazione e stop
- Almeno una baseline e un modello ML addestrati senza leakage, su un sottoinsieme con target affidabile.
- Risultati riproducibili con uno script e documentazione dei record esclusi.
- **Stop/ridefinizione** se non esiste target affidabile, le covariate consentite sono troppo incomplete o il numero di test utilizzabili è insufficiente: produrre solo audit, senza inventare risultati.

**Interpretazione:** una buona predizione statistica di `P_TLac` non dimostra che il modello ricostruisca la concentrazione di lattato, le soglie ventilatorie o la dinamica fisiologica dell'atleta.
