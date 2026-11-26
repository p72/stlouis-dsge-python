"""
観測データ15系列を FRED API から取得して四半期データを作る（論文 3.2 節に準拠）。

使い方:  FRED_API_KEY=xxxx python data.py   ->  data/observables.csv

TFP と 10年インフレ期待は、論文 3.2 節どおり FRED 以外から作る（paper_sources()）:
- TFP成長率: SF連銀 Fernald の quarterly_tfp.xlsx から、モデル整合的な系列
      dtfp/(1−alpha) − alpha/(1−alpha) × dutil
  を作り、推定期間（1959Q1〜2026Q2）の平均を引く（dtfp: 稼働率調整なしの事業部門TFP, alpha: 資本シェア, dutil: 稼働率）
- 10年インフレ期待: フィラデルフィア連銀 SPF の10年PCEインフレ予測（PCE10, 2007Q1〜）。それより前は10年CPI予測
  （SPF の INFCPI10YR 1991Q4〜、それより前は Livingston と Blue Chip を合わせた系列 1979Q4〜1991Q3）から、
  PCE10 と CPI10 の重なる期間（2007Q1〜）の平均差を引いて接続する

    python data.py --paper-sources   # data/raw/ に原データを取得し、observables_paper.csv を作る（FREDキー不要）

data/observables.csv は旧来の代替データ版（TFP欠損、EXPINF10YR で代用）のまま残してあり、
python data.py（FREDキーが必要）で作り直せる。
"""
import os
import sys
import json
import time
import urllib.request
import numpy as np
import pandas as pd

KEY = os.environ.get('FRED_API_KEY')
URL = 'https://api.stlouisfed.org/fred/series/observations?series_id={}&api_key={}&file_type=json'

SERIES = ['GDP', 'GDPDEF', 'POPTHM', 'PCE', 'GPDI', 'GCE',
          'BOGZ1FA366404005Q', 'BOGZ1FA366403005Q', 'BOGZ1FA366402005Q',
          'COMPNFB', 'AWHNONAG', 'CE16OV', 'CNP16OV', 'FEDFUNDS', 'PCEPILFE',
          'WTISPLC', 'DGOERG3Q086SBEA', 'DGS10', 'AAA', 'DGS20',
          'EXPINF10YR', 'CPIAUCSL', 'PCEPI']


def fred(sid):
    with urllib.request.urlopen(URL.format(sid, KEY), timeout=60) as r:
        obs = json.load(r)['observations']
    s = pd.Series({pd.Timestamp(o['date']): (np.nan if o['value'] == '.' else float(o['value']))
                   for o in obs})
    return s.resample('QS').mean()          # 四半期平均


def hp_trend(x, lam=1600):
    """HP フィルターのトレンド（人口の不連続を均す）"""
    x = np.asarray(x, float)
    n = len(x)
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i:i + 3] = [1, -2, 1]
    return np.linalg.solve(np.eye(n) + lam * D.T @ D, x)


def build(raw):
    q = pd.DataFrame(raw).loc['1959-01-01':]
    pop = pd.Series(np.exp(hp_trend(np.log(q['POPTHM'].interpolate()))), index=q.index)
    pop16 = q['CNP16OV'].copy()
    ok = pop16.notna()
    pop16[ok] = np.exp(hp_trend(np.log(pop16[ok])))
    dl = lambda s: 400 * np.log(s).diff()
    real_pc = lambda s: s / q['GDPDEF'] / pop
    tr = q['BOGZ1FA366404005Q'] + q['BOGZ1FA366403005Q'] + q['BOGZ1FA366402005Q']
    o = pd.DataFrame(index=q.index)
    o['gdp'] = dl(real_pc(q['GDP']))
    o['cons'] = dl(real_pc(q['PCE']))
    o['inv'] = dl(real_pc(q['GPDI']))
    o['gov'] = dl(real_pc(q['GCE']))
    o['tr'] = dl(real_pc(tr))
    o['wage'] = dl(q['COMPNFB'] / q['GDPDEF'])
    o['hours'] = dl(q['AWHNONAG'] * q['CE16OV'] / pop16)
    o['ffr'] = 400 * np.log(1 + q['FEDFUNDS'] / 400)
    o['core'] = dl(q['PCEPILFE'])
    o['tfp'] = np.nan                                   # Fernald TFP（取得不可）
    lo = np.log(q['WTISPLC'] / q['GDPDEF'])
    o['oil'] = lo - lo.mean()
    o['gas'] = dl(q['DGOERG3Q086SBEA'])
    gap = (dl(q['CPIAUCSL']) - dl(q['PCEPI'])).mean()   # CPI と PCE の平均差
    o['infexp'] = q['EXPINF10YR'] - gap
    o['t10'] = q['DGS10']
    o['cy'] = q['AAA'] - q['DGS20']
    return o


RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')
PAPER_URLS = {
    'quarterly_tfp.xlsx': 'https://www.frbsf.org/wp-content/uploads/quarterly_tfp.xlsx',
    'meanLevel.xlsx': 'https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/'
                      'survey-of-professional-forecasters/historical-data/meanLevel.xlsx',
    'Inflation.xlsx': 'https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/'
                      'survey-of-professional-forecasters/historical-data/Inflation.xlsx',
    'Additional-CPIE10.xlsx': 'https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/'
                              'survey-of-professional-forecasters/historical-data/Additional-CPIE10.xlsx',
}
SAMPLE = ('1959-01-01', '2026-04-01')


def fetch_raw(name, raw_dir=RAW):
    path = os.path.join(raw_dir, name)
    if not os.path.exists(path):
        os.makedirs(raw_dir, exist_ok=True)
        req = urllib.request.Request(PAPER_URLS[name], headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as r, open(path, 'wb') as f:
            f.write(r.read())
    return path


def _q(year, quarter):
    return pd.Timestamp(year=int(year), month=3 * int(quarter) - 2, day=1)


def fernald_tfp(path):
    """Fernald の四半期TFPからモデル整合的なTFP成長率（年率%、平均除去前）と構成要素を返す"""
    d = pd.read_excel(path, sheet_name='quarterly', header=None)
    hdr = d.index[d[0] == 'date'][0]
    d = d.iloc[hdr + 1:].copy(); d.columns = pd.read_excel(path, sheet_name='quarterly', header=None).iloc[hdr]
    d = d[d['date'].astype(str).str.match(r'^\d{4}:Q[1-4]$')]
    idx = [_q(*str(x).split(':Q')) for x in d['date']]
    f = d[['dtfp', 'alpha', 'dutil']].astype(float).set_index(pd.DatetimeIndex(idx))
    f['tfp_model'] = f['dtfp'] / (1 - f['alpha']) - f['alpha'] / (1 - f['alpha']) * f['dutil']
    return f


def spf_infexp(meanlevel, inflation, combined):
    """10年インフレ期待（PCEベース, %）と構成要素。PCE10 が無い期間は CPI10 − 平均差"""
    pce = pd.read_excel(meanlevel, sheet_name='PCE10')
    pce = pd.Series(pce['PCE10'].values, index=[_q(y, q) for y, q in zip(pce['YEAR'], pce['QUARTER'])], dtype=float)
    inf = pd.read_excel(inflation)
    cpi_spf = pd.Series(inf['INFCPI10YR'].values, index=[_q(y, q) for y, q in zip(inf['YEAR'], inf['QUARTER'])], dtype=float)
    c = pd.read_excel(combined, header=None)
    hdr = c.index[c[0].astype(str).str.strip() == 'Survey Date'][0]
    c = c.iloc[hdr + 1:]
    c = c[c[0].astype(str).str.match(r'^\d{4}:[1-4]$')]
    cpi_old = pd.Series(pd.to_numeric(c[3], errors='coerce').values,
                        index=[_q(*str(x).split(':')) for x in c[0]], dtype=float)
    cpi = cpi_spf.combine_first(cpi_old.loc[:'1991-07-01'])          # 1991Q4〜は SPF、それ以前は Livingston + Blue Chip
    both = pd.concat([cpi, pce], axis=1, keys=['cpi', 'pce']).dropna()
    gap = float((both['cpi'] - both['pce']).mean())
    out = pd.DataFrame({'pce10': pce, 'cpi10': cpi}).sort_index()
    out['cpi10_minus_gap'] = out['cpi10'] - gap
    out['infexp'] = out['pce10'].combine_first(out['cpi10_minus_gap'])
    out['source'] = np.where(out['pce10'].notna(), 'SPF PCE10',
                             np.where(out['cpi10'].notna(),
                                      np.where(out.index >= '1991-10-01', 'SPF CPI10 − 差', 'Livingston/Blue Chip CPI10 − 差'), ''))
    return out, gap


def paper_sources(raw_dir=RAW):
    """論文 3.2 節どおりの TFP成長率（平均除去済み）と10年インフレ期待"""
    p = {k: fetch_raw(k, raw_dir) for k in PAPER_URLS}
    f = fernald_tfp(p['quarterly_tfp.xlsx'])
    mean = f['tfp_model'].loc[SAMPLE[0]:SAMPLE[1]].mean()
    f['tfp'] = f['tfp_model'] - mean
    e, gap = spf_infexp(p['meanLevel.xlsx'], p['Inflation.xlsx'], p['Additional-CPIE10.xlsx'])
    out = f[['dtfp', 'alpha', 'dutil', 'tfp_model', 'tfp']].join(e, how='outer')
    out.attrs.update(tfp_mean=float(mean), cpi_pce_gap=gap)
    return out


if __name__ == '__main__' and '--paper-sources' in sys.argv:
    here = os.path.dirname(os.path.abspath(__file__))
    src = paper_sources()
    src.to_csv(os.path.join(here, 'data', 'paper_sources.csv'), float_format='%.6f')
    obs = pd.read_csv(os.path.join(here, 'data', 'observables.csv'), index_col=0, parse_dates=True)
    obs['tfp'] = src['tfp'].reindex(obs.index)
    obs['infexp'] = src['infexp'].reindex(obs.index)
    obs.to_csv(os.path.join(here, 'data', 'observables_paper.csv'), float_format='%.6f')
    print('TFP（モデル整合）の平均（除去した値）:', round(src.attrs['tfp_mean'], 4))
    print('CPI10 − PCE10 の平均差（2007Q1〜）:', round(src.attrs['cpi_pce_gap'], 4))
    print('TFP 観測期間:', src['tfp'].dropna().index[0].date(), '〜', src['tfp'].dropna().index[-1].date())
    print('インフレ期待の出どころ:', src.groupby('source')['infexp'].count().to_dict())
    sys.exit(0)


if __name__ == '__main__':
    assert KEY, '環境変数 FRED_API_KEY を設定してください'
    raw = {}
    for s in SERIES:
        raw[s] = fred(s)
        time.sleep(0.5)
    obs = build(raw)
    os.makedirs('data', exist_ok=True)
    obs.to_csv('data/observables.csv', float_format='%.6f')
    print(obs.tail(6).round(2))
    print('期間:', obs.index[0].date(), '〜', obs.dropna(how='all').index[-1].date())
