"""
観測データ15系列を FRED API から取得して四半期データを作る（論文 3.2 節に準拠）。

使い方:  FRED_API_KEY=xxxx python data.py   ->  data/observables.csv

論文との違い（取得できない系列の代替）:
- TFP成長率（Fernald, SF連銀）: FRED に無いため欠損扱い（カルマンフィルタが補完）
- 10年インフレ期待（SPF, フィラデルフィア連銀）: FRED の EXPINF10YR（クリーブランド連銀,
  10年CPIインフレ期待, 1982年〜）で代用し、CPI と PCE インフレの平均差を差し引く
"""
import os
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
