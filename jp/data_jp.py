"""
日本の観測データを集めて四半期データを作る（お試し版：米国の事後平均パラメータで使う前提）

使い方:  FRED_API_KEY=xxxx ESTAT_APP_ID=yyyy python jp/data_jp.py  ->  jp/data/observables_jp.csv

データ源
- ESRI 四半期別GDP速報（QE, 2026年7-9月期1次速報）: 名目季調の GDP・民間最終消費・民間住宅＋民間企業設備・
  政府最終消費、GDPデフレーター（季調）、雇用者報酬（名目季調）。1994Q1〜
- e-Stat 消費者物価指数（2020年基準, statsDataId 0003427113）: 生鮮食品及びエネルギーを除く総合（季調済み 0906,
  2010年〜。それ以前は原系列 0178 を自前で季節調整）、ガソリン（7301）
- FRED: 就業者数（LFEMTTTTJPQ647S）、総人口（POPTOTJPA647NWDB, 年次→補間）、コールレート
  （IRSTCI01JPM156N）、WTI（DCOILWTICO）、円ドル（EXJPUS）

米国版との違い
- 移転支出・TFP・10年インフレ期待・10年金利・コンビニエンス・イールドは欠損扱い
- 労働時間は「就業者数／人口」で代用（1人当たり労働時間の情報は入れていない）
- 実質賃金は「雇用者報酬／GDPデフレーター／就業者数」
- コールレートは 0.25% 以下（実質的なゼロ金利・マイナス金利）の四半期を欠損にする（ZLB 対策）
- 消費税率引き上げの四半期（1997Q2, 2014Q2, 2019Q4）はコアCPIとガソリンを欠損にする
"""
import io
import os
import json
import time
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd

FRED_KEY = os.environ.get('FRED_API_KEY')
ESTAT_ID = os.environ.get('ESTAT_APP_ID')
QE = 'https://www.esri.cao.go.jp/jp/sna/data/data_list/sokuhou/files/2026/qe263/tables/{}2631.csv'
HERE = os.path.dirname(os.path.abspath(__file__))
ZLB = 0.25                                       # これ以下のコールレートは欠損扱い（%）
TAX_HIKE = ['1997-04-01', '2014-04-01', '2019-10-01']


def get(url, sleep=1.0):
    req = urllib.request.Request(url, headers={'User-Agent': 'dsge-research-script'})
    for k in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                b = r.read()
            time.sleep(sleep)
            return b
        except urllib.error.HTTPError as e:
            if e.code == 403 and k < 2:
                time.sleep(20); continue
            raise


def qe(series, col):
    """ESRI QE の CSV から1列を四半期系列で返す（列番号は 1 始まりのデータ列）"""
    raw = pd.read_csv(io.BytesIO(get(QE.format(series))), encoding='cp932', header=None, dtype=str)
    out, year, q = {}, None, 0
    for _, row in raw.iterrows():
        lab = str(row[0]).strip()
        if '/' in lab and lab[:4].isdigit():
            year, q = int(lab[:4]), 1
        elif year is not None and lab.endswith('.') and '-' in lab:
            q += 1
        else:
            continue
        v = str(row[col]).replace(',', '').strip()
        out[pd.Timestamp(year, 3 * q - 2, 1)] = float(v) if v not in ('', 'nan', '***') else np.nan
    return pd.Series(out)


def fred(sid):
    u = f'https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key={FRED_KEY}&file_type=json'
    obs = json.loads(get(u, 0.5))['observations']
    return pd.Series({pd.Timestamp(o['date']): (np.nan if o['value'] == '.' else float(o['value']))
                      for o in obs})


def estat_cpi(cat):
    """CPI 指数（全国, 月次）"""
    q = dict(appId=ESTAT_ID, statsDataId='0003427113', cdCat01=cat, cdArea='00000', cdTab='1',
             limit=100000)
    u = 'https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData?' + urllib.parse.urlencode(q)
    vals = json.loads(get(u, 3.0))['GET_STATS_DATA']['STATISTICAL_DATA']['DATA_INF']['VALUE']
    s = {}
    for v in vals:
        t = v['@time']                          # 例 2026001010 = 2026年10月, 2026000000 = 年
        if t[4:6] == '00' and t[6:8] == t[8:10] and t[6:8] != '00':
            try:
                s[pd.Timestamp(int(t[:4]), int(t[6:8]), 1)] = float(v['$'])
            except ValueError:
                pass
    return pd.Series(s).sort_index()


def seasonal_adjust(q):
    """四半期系列の簡易季節調整（対数の中心化移動平均からの乖離で四半期別の固定季節要因を推定）"""
    lq = np.log(q)
    ma = lq.rolling(4, center=True).mean().rolling(2).mean().shift(-1)
    f = (lq - ma).groupby(lq.index.quarter).mean()
    f -= f.mean()
    return np.exp(lq - f.reindex(lq.index.quarter).values)


def hp_trend(x, lam=1600):
    x = np.asarray(x, float)
    n = len(x)
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i:i + 3] = [1, -2, 1]
    return np.linalg.solve(np.eye(n) + lam * D.T @ D, x)


def build(raw):
    idx = raw['gdp_n'].dropna().index
    q = pd.DataFrame({k: v.reindex(idx) for k, v in raw.items() if k != 'pop'})
    # 人口: 年次を四半期に対数補間（最終年以降は直近の伸びで延長）→ HP トレンド
    pa = raw['pop'].copy(); pa.index = pa.index + pd.DateOffset(months=6)   # 年央
    lp = np.log(pa).reindex(pa.index.union(idx)).interpolate(limit_direction='both', method='index')
    g = np.log(pa).diff().iloc[-1] / 4
    last = pa.index[-1]
    for t in lp.index[lp.index > last]:
        lp[t] = np.log(pa.iloc[-1]) + g * ((t - last).days / 91.3)
    pop = pd.Series(np.exp(hp_trend(lp.reindex(idx).values)), index=idx)
    dl = lambda s: 400 * np.log(s).diff()
    real_pc = lambda s: s / q['def'] / pop
    o = pd.DataFrame(index=idx)
    o['gdp'] = dl(real_pc(q['gdp_n']))
    o['cons'] = dl(real_pc(q['cons_n']))
    o['inv'] = dl(real_pc(q['inv_n']))
    o['gov'] = dl(real_pc(q['gov_n']))
    o['tr'] = np.nan
    o['wage'] = dl(q['comp'] / q['def'] / q['emp'])
    o['hours'] = dl(q['emp'] / pop)
    r = 400 * np.log(1 + q['call'] / 400)
    o['ffr'] = r.where(q['call'] > ZLB)
    o['core'] = dl(q['core'])
    o['tfp'] = np.nan
    lo = np.log(q['wti'] * q['fx'] / q['def'])
    o['oil'] = lo - lo.mean()
    o['gas'] = dl(q['gas'])
    for t in TAX_HIKE:
        o.loc[t, ['core', 'gas']] = np.nan
    for k in ['infexp', 't10', 'cy']:
        o[k] = np.nan
    o['call_raw'] = q['call']                   # 図用（観測には使わない）
    o['core_yoy_raw'] = 100 * np.log(q['core']).diff(4)
    return o


if __name__ == '__main__':
    assert FRED_KEY and ESTAT_ID, '環境変数 FRED_API_KEY と ESTAT_APP_ID を設定してください'
    raw = {}
    raw['gdp_n'] = qe('gaku-mk', 1)
    raw['cons_n'] = qe('gaku-mk', 2)
    raw['inv_n'] = qe('gaku-mk', 5) + qe('gaku-mk', 6)          # 民間住宅＋民間企業設備
    raw['gov_n'] = qe('gaku-mk', 8)
    raw['def'] = qe('def-qk', 1)
    raw['comp'] = qe('kshotoku-q', 3)                            # 雇用者報酬（名目・季調）
    cpi_sa = estat_cpi('0906').resample('QS').mean()             # 2010年〜 季調済み
    cpi_nsa = seasonal_adjust(estat_cpi('0178').resample('QS').mean())
    link = (cpi_sa / cpi_nsa).dropna().iloc[:4].mean()
    raw['core'] = cpi_sa.combine_first(cpi_nsa * link)
    raw['gas'] = seasonal_adjust(estat_cpi('7301').resample('QS').mean())
    raw['emp'] = fred('LFEMTTTTJPQ647S').resample('QS').mean()
    raw['pop'] = fred('POPTOTJPA647NWDB')
    raw['call'] = fred('IRSTCI01JPM156N').resample('QS').mean()
    raw['wti'] = fred('DCOILWTICO').resample('QS').mean()
    raw['fx'] = fred('EXJPUS').resample('QS').mean()
    obs = build(raw)
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
    obs.to_csv(os.path.join(HERE, 'data', 'observables_jp.csv'), float_format='%.6f')
    print(obs.tail(6).round(2))
    print(obs.describe().T[['count', 'mean', 'std']].round(2))
