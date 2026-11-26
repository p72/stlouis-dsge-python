"""
観測データ15系列を作る（論文 3.2 節）。原データのスナップショットから、ネットワークなしで再生成できる。

    python data.py                       # 論文再現モード（既定）: 保存済みスナップショットから 2026Q2 までを作る
    python data.py snapshot [YYYY-MM-DD] # 原データを取得してスナップショットを保存（FRED は ALFRED のその日のビンテージ）
    python data.py latest                # 最新データ更新モード: 今日のビンテージを取得し、入手できる最終四半期まで作る

論文再現モード（paper）
- 原データ: data/raw_fred/（FRED 23系列, ALFRED ビンテージ 2026-11-25, 元の頻度のまま）と
  data/raw/（Fernald TFP, SPF・Livingston/Blue Chip のExcel）。取得日・ビンテージ・系列ID・SHA-256 は各 manifest.json
- 期間: 1959Q1〜2026Q2。四半期化したあと、**HPフィルター・平均除去・平均差の調整より前に**この期間へ切り詰める
  （成長率の計算に要る1期前 1958Q4 だけは残す）。期間外の将来データやその後の改定は結果に影響しない
- 出力: data/observables.csv（代替データ版: TFP欠損、インフレ期待は EXPINF10YR で代用）
        data/observables_paper.csv（原典準拠版: Fernald TFP と SPF 期待インフレ）
        data/observables_meta.json（モード・期間・ビンテージ・入力ファイルのハッシュ）

最新データ更新モード（latest）
- 原データ: data/raw_fred_latest/, data/raw_latest/（Git 管理外）。期間は 1959Q1〜各系列がそろう最終四半期
- 出力: data/observables_latest.csv, data/observables_paper_latest.csv, data/observables_latest_meta.json
- 論文再現モードの出力は上書きしない

TFP と 10年インフレ期待（paper_sources()）:
- TFP成長率: SF連銀 Fernald の quarterly_tfp.xlsx から dtfp/(1−alpha) − alpha/(1−alpha) × dutil を作り、
  期間内の平均を引く（dtfp: 稼働率調整なしの事業部門TFP, alpha: 資本シェア, dutil: 稼働率）
- 10年インフレ期待: SPF の10年PCE予測（PCE10, 2007Q1〜）。それより前は10年CPI予測（SPF の INFCPI10YR 1991Q4〜、
  それより前は Livingston と Blue Chip を合わせた系列 1979Q4〜1991Q3）から、期間内で PCE10 と CPI10 が重なる四半期の
  平均差を引いて接続する
"""
import datetime as dt
import hashlib
import io
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')

SERIES = ['GDP', 'GDPDEF', 'POPTHM', 'PCE', 'GPDI', 'GCE',
          'BOGZ1FA366404005Q', 'BOGZ1FA366403005Q', 'BOGZ1FA366402005Q',
          'COMPNFB', 'AWHNONAG', 'CE16OV', 'CNP16OV', 'FEDFUNDS', 'PCEPILFE',
          'WTISPLC', 'DGOERG3Q086SBEA', 'DGS10', 'AAA', 'DGS20',
          'EXPINF10YR', 'CPIAUCSL', 'PCEPI']
ALFRED = 'https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={sid}&vintage_date={vintage}'
PAPER_URLS = {
    'quarterly_tfp.xlsx': 'https://www.frbsf.org/wp-content/uploads/quarterly_tfp.xlsx',
    'meanLevel.xlsx': 'https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/'
                      'survey-of-professional-forecasters/historical-data/meanLevel.xlsx',
    'Inflation.xlsx': 'https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/'
                      'survey-of-professional-forecasters/historical-data/Inflation.xlsx',
    'Additional-CPIE10.xlsx': 'https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/'
                              'survey-of-professional-forecasters/historical-data/Additional-CPIE10.xlsx',
}

MODES = {
    # 名前: (FRED原データ, その他原データ, 終期(None=最終四半期), 出力の接尾辞)
    'paper': (os.path.join(DATA, 'raw_fred'), os.path.join(DATA, 'raw'), '2026-04-01', ''),
    'latest': (os.path.join(DATA, 'raw_fred_latest'), os.path.join(DATA, 'raw_latest'), None, '_latest'),
}
START = '1959-01-01'
SAMPLE = (START, MODES['paper'][2])          # 論文の推定期間 1959Q1〜2026Q2
PAPER_VINTAGE = '2026-11-25'


# ---------------------------------------------------------------------------
# 取得とスナップショット
# ---------------------------------------------------------------------------
def _get(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def snapshot(vintage, fred_dir, raw_dir):
    """FRED（ALFRED の vintage 時点の値）と論文用の原データを保存し、manifest.json を書く"""
    os.makedirs(fred_dir, exist_ok=True); os.makedirs(raw_dir, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    man = dict(retrieved_at=now, vintage_date=vintage, source=ALFRED, series={})
    for sid in SERIES:
        body = _get(ALFRED.format(sid=sid, vintage=vintage))
        path = os.path.join(fred_dir, f'{sid}.csv')
        with open(path, 'wb') as f:
            f.write(body)
        s = read_fred_csv(path)
        man['series'][sid] = dict(file=f'{sid}.csv', column=pd.read_csv(path, nrows=0).columns[1], rows=int(s.notna().sum()),
                                  first=str(s.dropna().index[0].date()), last=str(s.dropna().index[-1].date()),
                                  sha256=_sha256(path))
        time.sleep(1)
    with open(os.path.join(fred_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    rman = dict(retrieved_at=now, files={})
    for name, url in PAPER_URLS.items():
        path = os.path.join(raw_dir, name)
        with open(path, 'wb') as f:
            f.write(_get(url))
        rman['files'][name] = dict(url=url, sha256=_sha256(path), bytes=os.path.getsize(path))
    with open(os.path.join(raw_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(rman, f, ensure_ascii=False, indent=1)
    return man, rman


def read_fred_csv(path):
    """ALFRED/FRED の CSV（元の頻度）を Series で返す。欠損は '.' または空欄"""
    d = pd.read_csv(path, index_col=0, parse_dates=True, na_values=['.', ''])
    return d.iloc[:, 0].astype(float)


def load_fred(fred_dir):
    """スナップショットを読み、四半期平均にする（日次・月次は四半期内の平均）"""
    return {sid: read_fred_csv(os.path.join(fred_dir, f'{sid}.csv')).resample('QS').mean() for sid in SERIES}


# ---------------------------------------------------------------------------
# 加工
# ---------------------------------------------------------------------------
def hp_trend(x, lam=1600):
    """HP フィルターのトレンド（人口の不連続を均す）"""
    x = np.asarray(x, float)
    n = len(x)
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i:i + 3] = [1, -2, 1]
    return np.linalg.solve(np.eye(n) + lam * D.T @ D, x)


def common_end(raw):
    """最新モードの終期: GDP・物価など主要系列がそろう最終四半期"""
    core = ['GDP', 'GDPDEF', 'PCE', 'GPDI', 'GCE', 'COMPNFB', 'PCEPILFE']
    return min(raw[s].dropna().index[-1] for s in core)


def build(raw, start=START, end=None):
    """四半期の原データ raw（{系列ID: Series}）から観測15系列（代替データ版）を作る。

    先に [start の1期前, end] に切り詰めてから、HPフィルター・平均除去・平均差を計算する。
    """
    end = pd.Timestamp(end) if end is not None else common_end(raw)
    start = pd.Timestamp(start)
    pre = start - pd.offsets.QuarterBegin(startingMonth=1)          # 成長率に要る1期前
    q = pd.DataFrame(raw).loc[pre:end]
    win = q.index >= start
    def hp_level(s):                                                # 観測のある区間だけで HP トレンド
        s = s.interpolate(limit_area='inside').copy()
        ok = s.notna()
        s[ok] = np.exp(hp_trend(np.log(s[ok])))
        return s
    pop = hp_level(q['POPTHM'])
    pop16 = hp_level(q['CNP16OV'])
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
    o['tfp'] = np.nan                                               # 代替データ版では欠損
    lo = np.log(q['WTISPLC'] / q['GDPDEF'])
    o['oil'] = lo - lo[win].mean()                                  # 期間内の平均を引く
    o['gas'] = dl(q['DGOERG3Q086SBEA'])
    gap = (dl(q['CPIAUCSL']) - dl(q['PCEPI']))[win].mean()          # 期間内の CPI と PCE の平均差
    o['infexp'] = q['EXPINF10YR'] - gap
    o['t10'] = q['DGS10']
    o['cy'] = q['AAA'] - q['DGS20']
    o = o.loc[start:end]
    o.attrs.update(start=str(start.date()), end=str(end.date()), cpi_pce_realized_gap=float(gap))
    return o


def _q(year, quarter):
    return pd.Timestamp(year=int(year), month=3 * int(quarter) - 2, day=1)


def fernald_tfp(path):
    """Fernald の四半期TFPからモデル整合的なTFP成長率（年率%、平均除去前）と構成要素を返す"""
    d = pd.read_excel(path, sheet_name='quarterly', header=None)
    hdr = d.index[d[0] == 'date'][0]
    cols = d.iloc[hdr]
    d = d.iloc[hdr + 1:].copy(); d.columns = cols
    d = d[d['date'].astype(str).str.match(r'^\d{4}:Q[1-4]$')]
    idx = [_q(*str(x).split(':Q')) for x in d['date']]
    f = d[['dtfp', 'alpha', 'dutil']].astype(float).set_index(pd.DatetimeIndex(idx))
    f['tfp_model'] = f['dtfp'] / (1 - f['alpha']) - f['alpha'] / (1 - f['alpha']) * f['dutil']
    return f


def spf_infexp(meanlevel, inflation, combined, start=START, end=None):
    """10年インフレ期待（PCEベース, %）と構成要素。PCE10 が無い期間は CPI10 − 平均差（期間内の重なりで計算）"""
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
    out = pd.DataFrame({'pce10': pce, 'cpi10': cpi}).sort_index().loc[start:end]
    both = out[['cpi10', 'pce10']].dropna()
    gap = float((both['cpi10'] - both['pce10']).mean())
    out['cpi10_minus_gap'] = out['cpi10'] - gap
    out['infexp'] = out['pce10'].combine_first(out['cpi10_minus_gap'])
    out['source'] = np.where(out['pce10'].notna(), 'SPF PCE10',
                             np.where(out['cpi10'].notna(),
                                      np.where(out.index >= '1991-10-01', 'SPF CPI10 − 差', 'Livingston/Blue Chip CPI10 − 差'), ''))
    return out, gap


def paper_sources(raw_dir=None, start=START, end=SAMPLE[1]):
    """論文 3.2 節どおりの TFP成長率（期間内で平均除去）と10年インフレ期待。期間外の行は使わない"""
    raw_dir = raw_dir or MODES['paper'][1]
    p = {k: os.path.join(raw_dir, k) for k in PAPER_URLS}
    missing = [k for k, v in p.items() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(f'{raw_dir} に原データがありません: {missing}（python data.py snapshot で取得）')
    f = fernald_tfp(p['quarterly_tfp.xlsx']).loc[start:end]
    mean = f['tfp_model'].mean()
    f['tfp'] = f['tfp_model'] - mean
    e, gap = spf_infexp(p['meanLevel.xlsx'], p['Inflation.xlsx'], p['Additional-CPIE10.xlsx'], start, end)
    out = f[['dtfp', 'alpha', 'dutil', 'tfp_model', 'tfp']].join(e, how='outer')
    out.attrs.update(tfp_mean=float(mean), cpi_pce_gap=gap)
    return out


def make(mode='paper'):
    """スナップショットから観測データを作って保存する（ネットワーク不要）"""
    fred_dir, raw_dir, end, sfx = MODES[mode]
    raw = load_fred(fred_dir)
    obs = build(raw, START, end)
    end = obs.attrs['end']
    src = paper_sources(raw_dir, START, end)
    obs_p = obs.copy()
    obs_p['tfp'] = src['tfp'].reindex(obs.index)
    obs_p['infexp'] = src['infexp'].reindex(obs.index)
    obs.to_csv(os.path.join(DATA, f'observables{sfx}.csv'), float_format='%.6f')
    obs_p.to_csv(os.path.join(DATA, f'observables_paper{sfx}.csv'), float_format='%.6f')
    src.to_csv(os.path.join(DATA, f'paper_sources{sfx}.csv'), float_format='%.6f')
    with open(os.path.join(fred_dir, 'manifest.json'), encoding='utf-8') as f:
        fman = json.load(f)
    rman_path = os.path.join(raw_dir, 'manifest.json')
    rman = json.load(open(rman_path, encoding='utf-8')) if os.path.exists(rman_path) else {}
    meta = dict(mode=mode, start=START, end=end, fred_vintage=fman['vintage_date'], fred_retrieved_at=fman['retrieved_at'],
                fred_dir=os.path.relpath(fred_dir, HERE), other_dir=os.path.relpath(raw_dir, HERE),
                other_retrieved_at=rman.get('retrieved_at'),
                inputs={**{f'{os.path.relpath(fred_dir, HERE)}/{s}.csv': v['sha256'] for s, v in fman['series'].items()},
                        **{f'{os.path.relpath(raw_dir, HERE)}/{k}': _sha256(os.path.join(raw_dir, k)) for k in PAPER_URLS}},
                processing=dict(order='四半期平均 → [start の1期前, end] に切り詰め → HP・平均除去・平均差',
                                cpi_pce_realized_gap=obs.attrs['cpi_pce_realized_gap'],
                                tfp_mean_removed=src.attrs['tfp_mean'], spf_cpi10_minus_pce10=src.attrs['cpi_pce_gap']),
                outputs=[f'data/observables{sfx}.csv', f'data/observables_paper{sfx}.csv', f'data/paper_sources{sfx}.csv'])
    with open(os.path.join(DATA, f'observables{sfx}_meta.json' if sfx else 'observables_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return obs, obs_p, meta


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'paper'
    if cmd == 'snapshot':
        vintage = sys.argv[2] if len(sys.argv) > 2 else PAPER_VINTAGE
        snapshot(vintage, *MODES['paper'][:2])
        print('スナップショットを保存:', MODES['paper'][0], MODES['paper'][1], 'ビンテージ', vintage)
    elif cmd == 'latest':
        today = dt.date.today().isoformat()
        snapshot(today, *MODES['latest'][:2])
        obs, obs_p, meta = make('latest')
        print('最新データ更新モード: ビンテージ', today, '期間', meta['start'], '〜', meta['end'])
    elif cmd == 'paper':
        obs, obs_p, meta = make('paper')
        print('論文再現モード: 期間', meta['start'], '〜', meta['end'], ' FREDビンテージ', meta['fred_vintage'])
        print(obs_p.tail(4).round(2))
    else:
        sys.exit(f'不明なコマンド: {cmd}（paper / snapshot [日付] / latest）')
