# Scostamenti individuali: risultato della validation

31 ottobre 2026. Run `ridge_residual_01`: 222 record OOF, cinque fold esterni, quattro interni, scaling attivo, nessun refit o accesso al test. Configurazione: `configs/residuals/training.json`.

Confronto appaiato tra `k * P_max` e `k * P_max + Ridge(X)`. k, imputazione, scaling e tuning sono stimati sul solo training di ciascun fold, anche interno. P_max è necessaria in inferenza, esclusa dalle covariate della correzione.

| Procedura | MAE (W) | RMSE (W) | R² OOF |
|---|---:|---:|---:|
| Proporzionale | 10,833 | 14,265 | 0,9024 |
| Proporzionale + Ridge | 10,824 | 14,215 | 0,9031 |

Guadagno MAE 0,009 W, positivo solo in due fold su cinque. Q_delta 0,00697: riduzione dello 0,70% dell'errore quadratico rispetto alla correzione nulla. Correlazione degli scostamenti osservati/predetti 0,0838; deviazione standard 14,30 W osservata e 1,09 W predetta. Alpha selezionato: 1000 nei fold 0, 1, 2, 4; 10000 nel fold 3. k varia da 0,7966 a 0,8000.

La correzione lineare regolarizzata non mostra un contributo utile alla personalizzazione. La forte regolarizzazione e la scarsa dispersione delle correzioni sono coerenti con questo esito; non dimostrano assenza di ogni relazione non lineare. L'intercetta della correzione può inoltre assorbire ricalibrazione media.

## Conclusione del progetto

Plateau operativo delle prestazioni con i dati e le procedure esplorate, non un limite matematico dimostrato. Non proseguire il tuning per inseguire vantaggi minimi sullo stesso campione. Il residuo non è automaticamente rumore: può comprendere errore di misura, variabilità fra giornate e covariate mancanti. Servono misurazioni ripetute per separare queste componenti e nuovi soggetti indipendenti per una conferma.

La Ridge diretta resta il modello selezionato senza P_max. Quando P_max è disponibile, il proporzionale è il riferimento più semplice sostenuto da questa validation. Il confronto numerico fra le due procedure non usa lo stesso insieme informativo. Nessun modello proporzionale finale è stato rifittato o valutato sul test in questa fase.

Esplorazione successiva a numerosi tentativi e a precedenti accessi al test: i risultati non sono una conferma indipendente. Le righe identificano test e non necessariamente atleti distinti. Nessuna validità clinica o equivalenza di T_Lac con IAT, LT1, LT2 o MLSS.

Dati aggregati: [summary.json](summary.json); [guadagni per fold](gains_folds.csv). Run dettagliata locale ignorata da Git: `results/residuals/runs/ridge_residual_01/`.
