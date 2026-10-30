"""Genera audit Markdown e pagina HTML autonoma del sottoinsieme ciclistico."""

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from plotly.offline import get_plotlyjs

from .analysis import analyze, flag_extremes
from .dataset import FIELDS, SIGNALS, load_dataset

EXPECTED_MD5 = 'f64cb1bf4d66129daed6c1954ea6ca9f'
SOURCE = 'https://zenodo.org/records/10841412'


def verify_archive(raw):
    archive = raw / 'data_v2.zip'
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != EXPECTED_MD5:
        raise ValueError('Checksum MD5 diverso da quello pubblicato da Zenodo')
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Archivio ZIP danneggiato')
        for member in z.infolist():
            destination = (raw / member.filename).resolve()
            if not destination.is_relative_to(raw.resolve()):
                raise ValueError('Percorso non sicuro nello ZIP')
            if member.is_dir():
                continue
            content = z.read(member)
            if destination.exists():
                if destination.read_bytes() != content:
                    raise ValueError(f'File estratto modificato: {member.filename}')
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
    return digest


def display(value):
    if value is None:
        return '—'
    if isinstance(value, float):
        return f'{value:.3f}'.rstrip('0').rstrip('.').replace('.', ',')
    return str(value).replace('|', '\\|').replace('\n', ' ')


def markdown_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(display(v) for v in row) + ' |' for row in rows]])


def write_audit(records, index, analysis, destination):
    a = analysis
    sexes = Counter(r['gender'] for r in records)
    statuses = Counter(r['target_status'] for r in records)
    dates = sorted(r['test_date'] for r in records if r['test_date'])
    flags = [(r['id'], issue) for r in records for issue in r['issues']]
    fields = a['fields']
    pairs = []
    corr = a['correlations']['spearman']
    for i, f in enumerate(fields):
        if f['group'] == 'lattato':
            continue
        for j in range(i + 1, len(fields)):
            if fields[j]['group'] == 'lattato' or corr['z'][i][j] is None:
                continue
            pairs.append((f['label'], fields[j]['label'], corr['z'][i][j], corr['n'][i][j]))
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    parts = [
        '# Audit esplorativo dei dati ciclistici',
        'Questa fase descrive il dataset originale. Nessuna imputazione, correzione automatica, selezione di feature predittive o modellazione è stata eseguita. '
        'La pagina [interattiva](index.html) permette di consultare tutti i test ciclistici e una sezione EDA dedicata.',
        '## Fonte, integrità e perimetro',
        f'[Donne et al., Zenodo v2, 19/03/2024]({SOURCE}), DOI 10.5281/zenodo.10841412. Archivio `data_v2.zip`, '
        f'MD5 pubblicato e verificato `{EXPECTED_MD5}`. CRC ZIP verificati e ogni file estratto confrontato byte per byte con l’archivio. '
        'Gli originali sono in `data/raw/`, esclusi da Git; l’HTML incorpora i dati ciclistici per funzionare offline.',
        f'L’indice contiene {len(index)} riferimenti unici a test, tutti presenti. '
        + ', '.join(f'{sport}: {n}' for sport, n in Counter(r['sport'] for r in index).items()) + '. '
        f'L’analisi comprende esclusivamente i {len(records)} test con sport `Cycling`.',
        f'Sesso riportato: {dict(sexes)}. Date interpretabili da {dates[0]} a {dates[-1]}. '
        'Il nome è vuoto nei file ciclistici e l’identificativo Axxx identifica il file/test, non un atleta. '
        'Data di nascita, sesso e antropometria non sono usati per inventare un identificativo personale. '
        'Non è possibile garantire l’indipendenza tra test né ricavare con certezza il numero di atleti.',
        '## Struttura e regole di lettura',
        'Ogni file ciclistico ha `Sheet1` (16 righe di metadati e antropometria), `Sheet2` (misure per step) e '
        '`Sheet3` (14 indici fisiologici). `Data_Summary.xlsx` contiene il foglio `Summary_Updated_v2` con 835 righe e 13 colonne: '
        'indice numerico senza nome, `file`, `name`, `sport`, `event`, `gender`, `DOB`, `test_date`, `lt_load`, `vo2max`, `HRmax`, `height`, `weight`. '
        'La pagina popola tabelle leggibili senza replicare lo stile Excel. Eventuali celle aggiuntive vengono conservate. '
        'Le colonne aggiuntive vuote/formattate non sono interpretate come nuove variabili.',
        'L’EDA usa i singoli file; target, VO₂max, FCmax, altezza e massa sono confrontati con l’indice e le discrepanze vengono segnalate. '
        'Le celle numeriche e le stringhe numeriche non ambigue diventano numeri; celle vuote, `-`, `Missing` e stringhe malformate restano '
        'non disponibili nelle statistiche. Le tabelle conservano i valori originali. Date Excel e date testuali GG/MM/AAAA vengono interpretate '
        'senza deduzioni sulle date mancanti. Età = giorni tra nascita e test / 365,2425.',
        '## Unità e dizionario',
        'La fonte descrive potenza in W, FC in bpm, lattato in mmol/L, VO₂ relativo in mL/kg/min e assoluto in L/min, ventilazione in L/min, '
        'altezza in m, massa in kg e pliche in mm. BMI in kg/m², grasso in % e LBM in kg seguono la convenzione delle rispettive etichette: '
        'le unità non sono esplicitate nelle celle. Le unità ematologiche non sono dichiarate e non vengono inventate. '
        'Cadenza ciclistica interpretata in giri/min dal protocollo. I valori non vengono riscalati.',
        markdown_table(['Colonna analitica', 'Etichetta originale', 'Unità', 'Gruppo'], FIELDS),
        markdown_table(['Colonna del segnale', 'Etichetta originale', 'Unità'], SIGNALS),
        'Metadati non numerici: `Name:`, `Gender:`, `Event:`, `Test:`, `Test Date:`, `DOB:`. '
        'Sintesi aggiuntive: età, numero di step attivi, potenza massima osservata e mediane di FC, VO₂, lattato e cadenza sugli step attivi. '
        'Queste mediane sono calcolate solo sui valori disponibili; nessun riempimento dei mancanti. '
        '`P_TLac_W_kg` è calcolata solo con target positivo e massa positiva disponibile.',
        '## EDA dei segnali',
        f'Sono presenti {a["active_n"]} step con potenza > 0 W e {a["rest_n"]} righe a riposo (0 W). '
        'Il riposo resta nelle tabelle ma è escluso da tutti i grafici, dalle distribuzioni dei segnali e dalle sintesi per test. '
        'Step con potenza mancante o negativa sono segnalati ed esclusi dai grafici. I grafici rispettano i carichi osservati, '
        'non interpolano i mancanti e non adattano curve.',
        markdown_table(['Segnale', 'Unità', 'Step numerici', 'Non disponibili', 'Min', 'Mediana', 'Max'],
                       [(label, unit, a['signal_stats'][key]['n'], a['signal_stats'][key]['missing'],
                         a['signal_stats'][key]['min'], a['signal_stats'][key]['median'], a['signal_stats'][key]['max'])
                        for key, label, unit in SIGNALS]),
        'Le statistiche della tabella precedente sono descrittive per step, non stime su campioni indipendenti. '
        'I box per carico nella pagina combinano test diversi: la composizione cambia con la potenza. '
        'I profili e la distribuzione del numero di step mostrano l’eterogeneità dei protocolli. '
        f'La cadenza ha almeno un valore attivo in {a["stats"]["median_cadence_rpm"]["n"]} test su {len(records)}.',
        '## EDA tabellare: disponibilità e range per test',
        'Una riga per test. “Non disponibili” comprende valori mancanti e non numerici. I valori numerici estremi non sono eliminati. '
        'I quartili usano interpolazione lineare (NumPy). Le distribuzioni per sesso non aggiustano per altre variabili.',
        markdown_table(['Variabile', 'Unità', 'n', 'Non disponibili', 'Min', 'Q1', 'Mediana', 'Q3', 'Max'],
                       [(f['label'], f['unit'], *[a['stats'][f['key']][k] for k in ('n', 'missing', 'min', 'q1', 'median', 'q3', 'max')])
                        for f in fields]),
        '## Target e indici legati al lattato',
        'Il target originale `Load @ Tlac` è disponibile e viene letto come `P_TLac_W`; non è ricalcolato. '
        'Secondo la fonte, T_Lac è l’intersezione di due rette adattate con regressione segmentata alla relazione lattato-carico. '
        'Non è equiparata a IAT, LT1, LT2 o MLSS.',
        markdown_table(['Controllo preliminare del target', 'Test'], statuses.items()),
        '“Entro intervallo osservato” significa soltanto che il target è positivo e compreso tra potenza attiva minima e massima. '
        'Non prova l’affidabilità del breakpoint né valida il calcolo originale. I test senza target o con target fuori intervallo '
        'sono conservati nel catalogo e nell’EDA, con segnalazione. Nessun criterio di inclusione per training è ancora stato applicato.',
        '## Correlazioni tra feature',
        'Pearson e Spearman sono calcolati per test su coppie complete, senza imputazione. Spearman assegna ranghi medi ai pareggi '
        'dopo avere selezionato le coppie complete. Il numero di coppie è mostrato per ogni cella. '
        'Con meno di 3 coppie o variabile costante il coefficiente è non definito. Non si calcolano p-value, '
        'intervalli di confidenza o significatività: l’indipendenza tra test non è dimostrata.',
        'La matrice iniziale esclude target, indici alla soglia, valori a 2 mmol/L e misure di lattato. '
        'Una vista separata include tutte le variabili a fini descrittivi. Sesso, date e identificativi non sono codificati numericamente. '
        'Sono incluse anche quantità matematicamente correlate (BMI, altezza, massa; VO₂ relativo e assoluto) e sintesi dipendenti dal protocollo. '
        'Una correlazione elevata non è una prova di valore predittivo fuori campione.',
        'Le dieci correlazioni Spearman di maggiore valore assoluto tra variabili non derivate dal lattato sono:',
        markdown_table(['Variabile X', 'Variabile Y', 'ρ', 'n'], pairs[:10]),
        '## Anomalie e controlli da approfondire',
        'Oltre a mancanti e stringhe malformate, sono segnalati estremi numerici oltre Q1−3×IQR o Q3+3×IQR, '
        'con almeno quattro valori e IQR > 0. Questa regola descrittiva non certifica errori fisiologici e non elimina osservazioni. '
        'Le correlazioni includono gli estremi numerici. Le unità ematologiche, i valori sospetti e i target fuori intervallo '
        'richiedono revisione prima della modellazione.',
        markdown_table(['Test', 'Segnalazione'], flags),
        '## Implicazioni per la successiva modellazione',
        'Nessun modello, split o tuning in questa fase. Una possibile definizione da confermare dopo la revisione dati: '
        'set A con antropometria e FC, set B con le stesse variabili più VO₂. La scelta delle specifiche sintesi di FC/VO₂ '
        'deve essere fissata prima della valutazione. Ematologia, ventilazione e cadenza non vengono automaticamente aggiunte ai set. '
        'Tutti gli indici derivati dal lattato e i valori di FC/VO₂ selezionati alla soglia devono essere esclusi dalle covariate. '
        'Non confondere una sintesi di un test completo con una misura disponibile prima dello sforzo.',
        'Il target necessita di audit di affidabilità e i dati numerici sospetti di verifica. '
        'In assenza di identificativo atleta, uno split per file non garantirebbe separazione tra persone: '
        'questo rischio dovrà essere dichiarato. Lo studio resta retrospettivo ed esplorativo, senza validità clinica attribuita.',
        '## Riproduzione',
        'Dalla radice del repository: `python -m src.build_report`. Il comando verifica l’archivio, ricostruisce i dati '
        'e rigenera `reports/data/data_audit.md` e `reports/data/index.html`. Vedere il README per download e ambiente. '
        'Gli originali non sono mai risalvati da openpyxl; le estensioni Excel non supportate non influiscono sui file conservati.',
    ]
    destination.write_text('\n\n'.join(parts) + '\n', encoding='utf-8')


def write_html(payload, output):
    """Rigenera solo la presentazione, senza modificare dati o audit."""
    data_json = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    template = Path(__file__).with_name('report.html').read_text(encoding='utf-8')
    html = template.replace('__PLOTLY__', get_plotlyjs()).replace('__DATA__', data_json)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'index.html').write_text(html, encoding='utf-8')
    return len(html.encode('utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('data/raw'))
    parser.add_argument('--output', type=Path, default=Path('reports/data'))
    parser.add_argument('--render-only', action='store_true', help='Aggiorna solo HTML dal manifest EDA esistente')
    args = parser.parse_args()
    if args.render_only:
        payload = json.loads(Path('data/processed/cycling.json').read_text(encoding='utf-8'))
        size = write_html(payload, args.output)
        print(f'HTML aggiornato: {args.output / "index.html"} ({size:,} byte); dati e audit invariati.')
        return
    digest = verify_archive(args.raw)
    records, index = load_dataset(args.raw / 'data')
    flag_extremes(records)
    analysis = analyze(records)
    payload = {'records': records, 'analysis': analysis, 'md5': digest,
               'target_counts': dict(Counter(r['target_status'] for r in records))}
    size = write_html(payload, args.output)
    write_audit(records, index, analysis, args.output / 'data_audit.md')
    processed = Path('data/processed')
    processed.mkdir(parents=True, exist_ok=True)
    (processed / 'cycling.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'{len(records)} test; {analysis["active_n"]} step attivi; {analysis["rest_n"]} righe a riposo escluse dai grafici.')
    print(f'Target: {payload["target_counts"]}')
    print(f'HTML: {args.output / "index.html"} ({size:,} byte)')
    print(f'Audit: {args.output / "data_audit.md"}')


if __name__ == '__main__':
    main()
