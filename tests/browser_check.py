"""Verifica opzionale della pagina locale con Microsoft Edge headless."""

from pathlib import Path
from playwright.sync_api import sync_playwright

out=Path('data/qa');out.mkdir(parents=True,exist_ok=True)
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(channel='msedge',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1080},device_scale_factor=1)
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(Path('reports/data/index.html').resolve().as_uri())
    page.wait_for_function('window.reportReady === true')
    base_layout = page.evaluate('JSON.stringify(base)')
    assert page.locator('#overview-panel').is_visible()
    assert 'P_TLac_W' in page.locator('#primary-target').inner_text()
    assert 'P_TLac_W_kg' in page.locator('#secondary-target').inner_text()
    assert 'mass_kg' in page.locator('#secondary-target').inner_text()
    page.screenshot(path=str(out/'overview_desktop.png'))
    page.locator('#tab-catalog').click()
    assert page.locator('.record-button').count()==285
    assert page.locator('#test-title').inner_text()=='A200'
    assert '164 W' in page.locator('#test-targets').inner_text()
    assert '3,2 W/kg' in page.locator('#test-targets').inner_text()
    assert page.evaluate("document.querySelector('#detail-hr').data[0].x.every(x=>x>0)")
    assert page.locator('#steps-table tbody tr').count()==6
    page.screenshot(path=str(out/'catalog_desktop.png'))
    page.locator('#search').fill('A3002')
    page.locator('.record-button').click()
    assert '1,674' in page.locator('#personal-table').inner_text(), 'precisione originale persa'
    page.locator('#search').fill('A229')
    page.locator('.record-button').click()
    assert page.locator('#test-targets').inner_text().count('Non disponibile') == 2
    page.locator('#issues summary').click()
    assert 'mancante/non numerico' in page.locator('#issues').inner_text()
    page.locator('#search').fill('zzzz')
    assert page.locator('.record-button').count()==0
    page.locator('#search').fill('')
    page.locator('#gender').select_option('F')
    assert page.locator('.record-button').count()==54
    page.locator('#tab-eda').click()
    page.wait_for_function("document.querySelector('#matrix').data !== undefined")
    assert len(page.evaluate("document.querySelector('#matrix').data[0].x")) == 8
    numeric_charts = ['overview-primary', 'overview-secondary', 'detail-hr', 'detail-vo2',
                      'detail-bla', 'detail-cadence', 'distribution', 'signal-chart',
                      'steps-distribution', 'scatter']
    def check_axes(chart_id, x_type='linear', y_type='linear'):
        actual = page.evaluate("id=>{const l=document.getElementById(id)._fullLayout;return [l.xaxis.type,l.yaxis.type]}", chart_id)
        assert actual == [x_type, y_type], f'{chart_id}: assi errati {actual}'
    for chart_id in numeric_charts:
        check_axes(chart_id)
    check_axes('gender-chart', 'category')
    check_axes('signal-missing', 'category')
    check_axes('matrix', 'category', 'category')
    assert page.evaluate('JSON.stringify(base)') == base_layout, 'Plotly ha modificato il layout condiviso'
    page.locator('#scatter-x').select_option('P_TLac_W')
    page.locator('#scatter-y').select_option('P_TLac_W')
    check_axes('scatter')
    assert page.evaluate("document.getElementById('scatter').calcdata[0].every(p=>p.x===p.y)"), 'X=Y non rappresentato sulla diagonale'
    page.locator('#scatter').screenshot(path=str(out/'scatter_identity_fixed.png'))
    page.locator('#matrix').screenshot(path=str(out/'matrix_essential.png'))
    assert page.evaluate("document.querySelector('#signal-chart').data.every(t=>t.x.every(x=>x>0))")
    page.screenshot(path=str(out/'eda_desktop.png'))
    page.locator('#signal-view').select_option('boxes')
    check_axes('signal-chart')
    assert page.evaluate("document.querySelector('#signal-chart').data.every(t=>t.x.every(x=>x>0))")
    page.locator('#signal-feature').select_option('cadence_rpm')
    page.locator('#signal-chart').screenshot(path=str(out/'signals.png'))
    page.locator('#correlation-method').select_option('pearson')
    page.locator('#correlation-group').select_option('all')
    assert page.locator('#correlation-warning').is_visible()
    assert len(page.evaluate("document.querySelector('#matrix').data[0].x"))==32
    page.locator('#matrix').screenshot(path=str(out/'matrix.png'))
    page.locator('#scatter-x').select_option('P_TLac_W')
    page.locator('#scatter-y').select_option('mass_kg')
    assert '283 coppie complete' in page.locator('#scatter-caption').inner_text()
    page.locator('#correlation-group').select_option('antropometria')
    page.evaluate("document.querySelector('#matrix').emit('plotly_click',{points:[{x:'Altezza',y:'Massa corporea'}]})")
    assert page.locator('#scatter-x').input_value()=='height_m'
    assert page.locator('#scatter-y').input_value()=='mass_kg'
    # Le viste categoriche non devono contaminare i grafici numerici successivi.
    keys = page.locator('#distribution-feature option').evaluate_all('(options)=>options.map(o=>o.value)')
    for key in keys:
        page.locator('#distribution-feature').select_option(key)
        check_axes('distribution')
        check_axes('gender-chart', 'category')
        page.locator('#scatter-x').select_option(key)
        page.locator('#scatter-y').select_option(key)
        check_axes('scatter')
        assert page.evaluate("document.getElementById('scatter').calcdata[0].every(p=>p.x===p.y || (!Number.isFinite(p.x)&&!Number.isFinite(p.y)))")
    for key in ['hr_bpm', 'vo2_ml_kg_min', 'bla_mmol_l', 'cadence_rpm']:
        page.locator('#signal-feature').select_option(key)
        for mode in ['profiles', 'boxes']:
            page.locator('#signal-view').select_option(mode)
            check_axes('signal-chart')
    assert page.evaluate('JSON.stringify(base)') == base_layout
    page.set_viewport_size({'width':390,'height':844})
    page.locator('#tab-overview').click()
    page.wait_for_function('document.documentElement.scrollWidth <= window.innerWidth',timeout=5000)
    page.evaluate('window.scrollTo(0,0)')
    page.screenshot(path=str(out/'overview_mobile.png'))
    page.locator('#tab-catalog').click()
    page.locator('#search').fill('A200')
    page.locator('.record-button').click()
    page.wait_for_function('document.documentElement.scrollWidth <= window.innerWidth',timeout=5000)
    page.screenshot(path=str(out/'catalog_mobile.png'))
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'overflow mobile'
    page.locator('#tab-eda').click()
    page.wait_for_function('document.documentElement.scrollWidth <= window.innerWidth',timeout=5000)
    page.screenshot(path=str(out/'eda_mobile.png'))
    # Controlla anche i grafici dei run completati, senza leggere dataset train/test.
    for summary in Path('results/runs').glob('*/summary.json'):
        for filename, axes in [
            ('validation.html', {'xaxis':'linear','yaxis':'linear','xaxis2':'linear','yaxis2':'linear'}),
            ('importance.html', {'xaxis':'linear','yaxis':'category'}),
        ]:
            artifact = summary.parent / filename
            if not artifact.exists():
                continue
            page.goto(artifact.resolve().as_uri())
            page.wait_for_function("document.querySelector('.js-plotly-plot')?._fullLayout !== undefined")
            actual = page.evaluate("keys=>Object.fromEntries(keys.map(k=>[k,document.querySelector('.js-plotly-plot')._fullLayout[k].type]))", list(axes))
            assert actual == axes, f'{artifact}: {actual}'
    assert not errors, errors
    browser.close()
print('Verifica browser completata: assi numerici/categorici, tutte le variabili EDA, identità X=Y, grafici dei run completati, catalogo e layout mobile; nessun errore JavaScript.')
