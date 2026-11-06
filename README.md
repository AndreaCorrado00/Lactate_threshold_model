# Predizione di P_TLac

Stima della potenza alla soglia lattacida (`P_TLac_W`) su cicloergometro senza dati sul lattato, confrontando una regressione Ridge su 20 descrittori con un modello proporzionale alla potenza massima del test. Studio retrospettivo ed esplorativo, senza validità clinica dimostrata. La relazione completa è in [predizione della potenza alla soglia.pdf](predizione%20della%20potenza%20alla%20soglia.pdf).

## Riprodurre l'analisi

**Seguire il notebook [README.ipynb](README.ipynb)**, che contiene in ordine i comandi da terminale (PowerShell, dalla radice del repository) per:

1. preparare l'ambiente (`requirements.txt`) ed eseguire i test;
2. scaricare `data_v2.zip` da Zenodo e verificarne l'integrità;
3. costruire il dataset (una riga per test);
4. creare la partizione train/test e i fold di validazione;
5. definire le covariate;
6. addestrare la Ridge;
7. valutarla sul test.

Il notebook si legge in VS Code o Jupyter senza eseguire celle Python. Dati e modelli binari sono locali e non versionati.

## Dati

279 test incrementali (step di 3 min, incrementi di 20–40 W) dall'archivio pubblico di Donne et al. (2024, [doi:10.5281/zenodo.10841412](https://doi.org/10.5281/zenodo.10841412)); atleti allenati, ~80% uomini, età media ~30 anni. Target: `Load @ Tlac` riportato dal laboratorio (regressione segmentale a due rette sulla curva lattato-carico), non ricalcolato; mediana 280 W (144–380 W). Split casuale stratificato per sesso ed età: 222 test di stima, 57 di valutazione. [Audit dati](reports/data_audit.md).

## Modelli

- **Ridge**: combinazione lineare di 20 descrittori standardizzati (sesso, età, 6 antropometrici, 4 ematologici, 4 massimali, 4 sub-massimali). Esclusi lattato e potenza massima. Imputazione per mediana stratificata per sesso e scaling stimati nei soli fold di training; α = 0,1 scelto con CV annidata sul train.
- **Proporzionale**: `P_TLac = k · P_max`, con k stimato ai minimi quadrati senza intercetta sul train (k = 0,799). Assegna a tutti la stessa frazione.

Entrambi applicati una sola volta ai 57 test; IC 95% con bootstrap (2000 ricampionamenti, seed 42), differenza fra MAE con bootstrap appaiato.

## Risultati

| Indice (57 test) | Ridge | Proporzionale |
|---|---:|---:|
| MAE (W) | 12,7 (10,1–15,6) | 10,3 (8,2–12,5) |
| RMSE (W) | 16,6 (12,7–20,7) | 13,3 (10,8–15,7) |
| R² | 0,87 (0,76–0,93) | 0,91 (0,86–0,95) |
| Bias (W) | −3,8 | −1,8 |

Differenza fra MAE: 2,4 W (IC 95% 0,3–4,6) a favore del proporzionale. Pesi Ridge maggiori: potenza a 160 bpm (+47 W/SD), massa (+37), FC mediana (+22), VO₂max relativo (+19); ematologici < 5 W. [Metriche Ridge](results/metrics.csv) · [Risultati dettagliati](reports/results.md).

## Discussione

- La stima Ridge è correlata con la potenza massima a r = 0,97 pur non avendola fra gli input: ricostruisce indirettamente la capacità aerobica complessiva, aggiungendo errore rispetto all'uso diretto di `P_max`.
- Sul piano pratico il proporzionale richiede solo un test incrementale; la Ridge richiede metabolimetro, esame del sangue, composizione corporea e comunque un test massimale.
- k = 0,80 è specifico del protocollo: va ricalcolato per durate degli step o incrementi diversi.

## Limiti

Il limite principale è comune ai due modelli: **nessuno dei due prevede la frazione di utilizzo individuale**, cioè la percentuale della potenza massima a cui si colloca la soglia (`P_TLac / P_max`). Nei test di valutazione questa frazione ha media 80%, SD ~4 punti e varia fra 68% e 88%.

- Il proporzionale assegna per costruzione a tutti lo stesso valore (79,9%): il suo errore coincide con `(frazione dell'atleta − 0,80) · P_max`.
- La Ridge, che in principio potrebbe distinguere gli atleti, produce frazioni meno disperse (SD 2,8 punti) e correlate con quelle misurate solo a r = 0,11.

Entrambi i modelli colgono quindi la capacità aerobica complessiva, non la sua frazione di utilizzo. Questa dipende probabilmente da caratteristiche periferiche del muscolo (densità mitocondriale, capillarizzazione, tipologia di fibre, trasportatori del lattato) assenti fra i descrittori. Per gli atleti con frazione molto alta o molto bassa l'errore è sistematico, proprio sulla componente che più ne caratterizza il profilo.

Ci sono poi limiti dei dati e della procedura. L'archivio non identifica gli atleti, quindi test ripetuti dello stesso atleta possono finire sia nel train sia nel test. Il test era già stato consultato durante lo sviluppo della Ridge. Il campione proviene da un solo laboratorio, con un solo protocollo. `T_Lac` non va equiparata a LT1, LT2, OBLA o MLSS.
