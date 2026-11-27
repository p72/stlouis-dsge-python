"""
時間当たり実質賃金の歴史分解（本格版パラメータ）と、1人当たり版との比較

使い方:
  ESTAT_APP_ID=yyyy python jp/hours_jp.py fetch   # 労働力調査の平均週間就業時間（四半期）を data/hours_lfs_jp.csv に保存
  python jp/hours_jp.py                           # 保存済みデータで比較の図と表を作る

データ
- 総務省「労働力調査」基本集計 四半期: 全産業・総数（男女計・従業上の地位総数）の平均週間就業時間（時間）。
  1985Q1〜2010Q4 は statsDataId 0002060025、2011Q1〜 は 0003031528（15歳以上）。季節調整は data_jp.py と同じ簡易法。
  2011Q1〜Q3 は震災で全国値が無いので、前年と翌年の同じ四半期の平均で埋める。
- 観測の実質賃金は 雇用者報酬÷GDPデフレーター÷就業者数 なので、時間当たりにするには 1人当たり週間就業時間 h の伸びを引く。
    時間当たり実質賃金の伸び  wage_h = wage − 400·Δlog h
    総労働時間（1人当たり人口）の伸び hours_h = hours + 400·Δlog h

比べるケース（パラメータは本格版の事後平均のまま。推定し直してはいない）
  A 1人当たり（いまの日本版）
  B 実質賃金だけ時間当たりに
  C 実質賃金を時間当たり、労働時間の観測も「就業者×時間÷人口」に（モデルの n＝総労働時間とそろう）
"""
import os
import sys
import json
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import estimate_jp as E                                         # noqa: E402
import hd                                                       # noqa: E402
from data_jp import seasonal_adjust                             # noqa: E402
from hd_jp_est import posterior_theta, run_est                  # noqa: E402
from hd_jp_wage import GROUPS, SHOCK_JP, to_level, wage_parts   # noqa: E402

CSV = os.path.join(HERE, 'data', 'hours_lfs_jp.csv')
TABLES = [('0002060025', {}), ('0003031528', {'cdCat04': '00'})]


def fetch():
    key = os.environ['ESTAT_APP_ID']
    s = {}
    for sid, extra in TABLES:
        q = dict(appId=key, statsDataId=sid, cdTab='03', cdCat01='000', cdCat02='0', cdCat03='00', cdArea='00000',
                 limit=100000, **extra)
        u = 'https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData?' + urllib.parse.urlencode(q)
        vals = json.loads(urllib.request.urlopen(u, timeout=60).read())['GET_STATS_DATA']['STATISTICAL_DATA']['DATA_INF']['VALUE']
        for v in vals:
            t = v['@time']                                       # 例 2011000103 = 2011年1〜3月期
            try:
                s[pd.Timestamp(int(t[:4]), int(t[6:8]), 1)] = float(v['$'])
            except ValueError:
                pass
        import time; time.sleep(3)                              # e-Stat へは間隔をあける
    df = pd.DataFrame({'hours_per_worker_nsa': pd.Series(s).sort_index()})
    df.index.name = 'date'
    df.to_csv(CSV, float_format='%.2f')
    return df


def load_hours():
    df = pd.read_csv(CSV, index_col=0, parse_dates=True)
    full = pd.date_range(df.index.min(), df.index.max(), freq='QS')
    df = df.reindex(full)
    # 2011Q1〜Q3 は東日本大震災で全国値が無い → 前年と翌年の同じ四半期の平均で埋める（原系列）
    for t in df.index[df['hours_per_worker_nsa'].isna()]:
        a, b = t - pd.DateOffset(years=1), t + pd.DateOffset(years=1)
        df.loc[t, 'hours_per_worker_nsa'] = df.loc[[a, b], 'hours_per_worker_nsa'].mean()
    df['filled'] = ~df.index.isin(pd.read_csv(CSV, index_col=0, parse_dates=True).index)
    df['hours_per_worker_sa'] = seasonal_adjust(df['hours_per_worker_nsa'])
    return df


def make_data(case):
    """ケース B/C の観測データ（E.DATA を元に作る）"""
    d = E.DATA.copy()
    if case == 'A':
        return d
    h = load_hours()['hours_per_worker_sa'].reindex(d.index)
    dh = 400 * np.log(h).diff()
    d['wage'] = d['wage'] - dh
    if case == 'C':
        d['hours'] = d['hours'] + dh
    return d


def run_case(theta, case):
    """E.DATA を差し替えて推定時と同じ手順でカルマン平滑化（定数は各ケースのサンプル平均）"""
    orig = E.DATA
    try:
        E.DATA = make_data(case)
        q, by, mean, ll = wage_parts(theta, E.DATA)
        m, T, R, Z, d, xs, eta, ll2, init, contrib = run_est(theta)
        from hd_jp import parts_of
        hrs = parts_of(m, init, contrib, xs, 'hours_obs', E.DATA.index)
        gdp = parts_of(m, init, contrib, xs, 'gdp_obs', E.DATA.index)
        return dict(q=q, by=by, mean=mean, ll=ll, data=E.DATA, hours=hrs, gdp=gdp)
    finally:
        E.DATA = orig


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'fetch':
        print(fetch().tail()); sys.exit()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from hd_jp import stacked
    plt.rcParams['font.family'] = 'IPAGothic'
    plt.rcParams['axes.unicode_minus'] = False
    th = posterior_theta('mean')
    H = load_hours()
    names = {'A': 'A 1人当たり（いまの日本版）', 'B': 'B 時間当たり賃金', 'C': 'C 時間当たり賃金＋総労働時間'}
    res = {c: run_case(th, c) for c in 'ABC'}
    for c, r in res.items():
        ok = r['data']['wage'].notna()
        print(names[c], ' 平均（年率%）', round(r['mean'], 3), ' 対数尤度', round(r['ll'], 1),
              ' 実績との差', round((r['q']['合計'] - (r['data']['wage'] - r['mean']))[ok].abs().max(), 8))

    # 表: 累積寄与（グループ別・主なショック別）
    rows = []
    for lab, start, end in [('2013Q1〜2026Q3', '2013-01-01', None), ('2013Q1〜2021Q4', '2013-01-01', '2021-10-01'),
                            ('2022Q1〜2026Q3', '2022-01-01', None)]:
        for c, r in res.items():
            s = (r['by'].loc[start:end] / 4).sum()
            g = (r['q'].loc[start:end] / 4).sum()
            act = ((r['data']['wage'] - r['mean']).loc[start:end] / 4).sum()
            rows.append({'期間': lab, 'ケース': names[c], '実績の累積乖離': act, **{k: g[k] for k in GROUPS},
                         'TFP成長': s['GZ'], '投資の限界効率': s['zeta'], '価格マークアップ': s['mup'],
                         '賃金マークアップ': s['muw'], '労働不効用': s['GN']})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(HERE, 'data', 'hours_jp_wage_compare.csv'), index=False, float_format='%.3f')
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        print(tab.round(2).to_string(index=False))

    # 図1: 1人当たり週間就業時間と、実質賃金（1人当たり・時間当たり）の水準
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    hh = H.loc['1994-01-01':]
    axes[0].plot(hh.index, hh['hours_per_worker_nsa'], color='#bbbbbb', lw=1, label='原系列')
    axes[0].plot(hh.index, hh['hours_per_worker_sa'], color='#1f6fb5', lw=2, label='季節調整値')
    axes[0].set_title('1人当たり平均週間就業時間（労働力調査, 全産業・総数）'); axes[0].set_ylabel('時間')
    axes[0].grid(alpha=0.3); axes[0].legend()
    for c, col in [('A', '#d95f02'), ('B', '#1f6fb5')]:
        w = res[c]['data']['wage'].fillna(0)
        lv = 100 * np.exp((w / 400).cumsum())
        lv = lv / lv.loc['2012-10-01'] * 100
        axes[1].plot(lv.index, lv, color=col, lw=2, label=('実質賃金（就業者1人当たり）' if c == 'A' else '実質賃金（1時間当たり）'))
    axes[1].axvline(pd.Timestamp('2012-10-01'), color='gray', ls='--', lw=1)
    axes[1].set_title('実質賃金の水準（2012Q4＝100, GDPデフレーターで実質化）'); axes[1].grid(alpha=0.3); axes[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'figures', 'hours_jp_levels.png'), dpi=130)

    # 図2: 水準の分解（2012Q4＝0）を A と C で並べる
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharey=True)
    for ax, c in zip(axes, ['A', 'C']):
        r = res[c]
        lv = to_level(r['q'], '2012-10-01').loc['2012-10-01':]
        stacked(ax, lv, step=2, line_label='累積乖離（推計値＝実績）')
        ax.set_title(f'{names[c]}：実質賃金の水準（2012Q4＝0, 年{r["mean"]:.2f}%のトレンドからの累積乖離）')
        ax.set_ylabel('%')
    axes[0].legend(ncol=7, loc='lower left', fontsize=9)
    fig.suptitle('実質賃金を「時間当たり」にすると、低迷の説明はどう変わるか', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(HERE, 'figures', 'hours_jp_wage_level.png'), dpi=130)

    # 図3: 前年比の分解（ケースC, 1995〜）
    r = res['C']
    yoy = r['q'].rolling(4).mean().loc['1995-01-01':]
    fig, ax = plt.subplots(figsize=(15, 6.5))
    x = stacked(ax, yoy, step=4, line_label=f'平滑化推計値（平均{r["mean"]:.2f}%からの乖離）')
    act = (r['data']['wage'] - r['mean']).rolling(4).mean().reindex(yoy.index)
    ax.plot(x, act.values, color='w', lw=0, marker='o', ms=3.5, mec='k', mew=0.6, label='実績')
    ax.set_ylabel('%pt（前年比, 平均からの乖離）'); ax.legend(ncol=8, loc='lower left', fontsize=9)
    ax.set_title('時間当たり実質賃金（雇用者報酬÷GDPデフレーター÷総労働時間, 前年比）の歴史分解【ケースC】')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'figures', 'hours_jp_wage_yoy.png'), dpi=130)

    # 図4: 主なショックの累積（2013Q1〜2026Q3）を3ケースで
    sub = tab[tab['期間'] == '2013Q1〜2026Q3'].set_index('ケース')
    cols = ['実績の累積乖離', 'TFP成長', '投資の限界効率', '価格マークアップ', '賃金マークアップ', '労働不効用', '需要', '金融']
    fig, ax = plt.subplots(figsize=(14, 5.5))
    xx = np.arange(len(cols)); w = 0.27
    for i, (cs, colr) in enumerate(zip(sub.index, ['#d95f02', '#9ecae1', '#1f6fb5'])):
        ax.bar(xx + (i - 1) * w, sub.loc[cs, cols].values, w, color=colr, edgecolor='k', lw=0.4, label=cs)
    ax.set_xticks(xx); ax.set_xticklabels(cols); ax.axhline(0, color='k', lw=0.8); ax.grid(axis='y', alpha=0.3)
    ax.set_ylabel('累積乖離への寄与（%）'); ax.legend(fontsize=9, loc='lower right')
    ax.set_title('2013Q1〜2026Q3 の実質賃金の累積乖離：1人当たり（A）と時間当たり（B・C）')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'figures', 'hours_jp_wage_shocks.png'), dpi=130)

    h = H['hours_per_worker_sa']; dh = 400 * np.log(h).diff()
    print('1人当たり時間の伸び 平均(1994Q2〜) 年率', round(dh.loc['1994-04-01':'2026-07-01'].mean(), 2),
          ' 2013Q1〜累積', round((dh.loc['2013-01-01':'2026-07-01'] / 4).sum(), 2))
    print('最新', H.tail(3).round(2))
