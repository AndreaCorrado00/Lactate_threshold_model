# Coorte del modello Ridge

Originali Donne et al., Zenodo v2, `data_v2.zip`, già verificato nella fase di audit (MD5 atteso `f64cb1bf4d66129daed6c1954ea6ca9f`). Mai modificati; fuori Git. Audit esteso precedente conservato nell'archivio locale.

285 test ciclistici; A229 e A519 esclusi per target mancante/non finito. Split originale: 226 train, 57 test, stratificato per sesso ed età, seed 42. ID per test, non per atleta: l'indipendenza fra atleti nelle partizioni non è garantita.

Train corrente: 222 record dopo le esclusioni richieste:

| ID | Motivazione |
|---|---|
| A539 | VO₂max assoluto e relativo incoerenti rispetto alla massa nella fonte |
| A1114 | VO₂ a 160 bpm sospetto, errore non definitivamente confermato |
| A4000 | Target 266 W oltre la massima potenza osservata 240 W |
| A4001 | Target 266 W oltre la massima potenza osservata 250 W |

Esclusioni successive all'esplorazione, non tutte errori accertati. Nessuna esclusione nel test. Fold esterni originali conservati per i record mantenuti: 45, 45, 43, 44, 45 test. Manifest e motivi in `data/splits/v2_train_qc/`; test originale in `data/splits/v1/`.

Target non imputato. Covariate conservate con mancanti; imputazione per sesso e scaling soltanto nei fit. Ridge usa sesso, antropometria, ematologia, misure cardiorespiratorie, età e potenza a 160 bpm. Non usa massima potenza, step, cadenza o variabili derivate dal lattato. Nella prima fase la potenza massima serve solo per i rapporti diagnostici; nella fase scostamenti è invece un input necessario della baseline proporzionale, escluso dalle covariate della correzione.

Nel test sette celle mancanti fra gli input: Hct, RBC, WBC, ventilazione massima ed età una ciascuna, mediana VO₂ due. Imputate con mediane del train. Disponibilità dettagliate in `ridge/analysis.json`.

Il holdout è stato osservato prima della selezione finale Ridge. La valutazione autorizzata non costituisce una nuova conferma indipendente del processo di sviluppo. Nessuna validità clinica o equivalenza T_Lac/IAT/LT1/LT2/MLSS attribuita.
