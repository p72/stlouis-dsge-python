"""
検討: 日本の年次TFP（Penn World Table）を観測に足すと、実質賃金の歴史分解はどう変わるか

使い方:
  FRED_API_KEY=xxxx python jp/tfp_jp.py fetch   # PWT の TFP と労働分配率を FRED から取って data/tfp_pwt_jp.csv に保存
  python jp/tfp_jp.py                           # 保存済みデータで比較の図と表を作る

やり方（パラメータは本格版の事後平均のまま。推定し直してはいない）
- データ: Penn World Table 11.0 の日本の TFP（RTFPNAJPA632NRUG, 国内価格・2019=1）と労働分配率（LABSHPJPA156NRUG）。年次、1990〜2023。
  稼働率の調整はしていない（米国版の Fernald 系列と違う点）。
- モデルの「TFP」は労働増大型（Y = K^α (Z N)^(1-α)）なので、PWT のヒックス中立型 TFP（A）の伸びを労働分配率で割って
  dlog Z = dlog A / labsh に直す（Fernald の変換と同じ考え方）。サンプル（1995〜2023年）平均を引いて、定数は0。
- 年次の値は「年平均の水準の伸び」なので、四半期の伸び（年率）の 7四半期の三角加重平均（1,2,3,4,3,2,1）/16 に当たる。
  状態に TFP成長率 GZ の6期分のラグを足し、各年の第4四半期にだけ観測がある形にする。
- 測定誤差（稼働率の変動や PWT の計測誤差）を i.i.d. で入れる。標準偏差は年率 ME_SD=1%pt を基本に、
  0.25%pt と、データに強く寄せる 0.03%pt でも計算する（FORCE_SD）。
"""
import os
import sys
import json
import urllib.request
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import estimate_jp as E                                         # noqa: E402
import hd                                                       # noqa: E402
from hd_jp import parts_of                                      # noqa: E402
from hd_jp_est import posterior_theta, run_est                  # noqa: E402
from hd_jp_wage import ALL_SHOCKS, GROUPS, SHOCK_JP, to_level   # noqa: E402

CSV = os.path.join(HERE, 'data', 'tfp_pwt_jp.csv')
SERIES = {'rtfpna': 'RTFPNAJPA632NRUG', 'labsh': 'LABSHPJPA156NRUG'}
W = np.array([1, 2, 3, 4, 3, 2, 1]) / 16                       # 年平均水準の伸び ← 四半期の伸び（年率）
ME_SD = 1.0                                                     # 年次TFPの測定誤差（年率 %pt）
FORCE_SD = 0.03                                                 # データに強く寄せるケース
SAMPLE = ('1995', '2023')


def fetch():
    key = os.environ['FRED_API_KEY']
    out = {}
    for nm, sid in SERIES.items():
        u = f'https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key={key}&file_type=json'
        obs = json.loads(urllib.request.urlopen(u, timeout=60).read())['observations']
        out[nm] = pd.Series({int(o['date'][:4]): float(o['value']) for o in obs if o['value'] != '.'})
    df = pd.DataFrame(out)
    df.index.name = 'year'
    df.to_csv(CSV, float_format='%.6f')
    return df


def load_tfp():
    """年次の労働増大型TFP成長率（%）、サンプル平均からの乖離、平均"""
    df = pd.read_csv(CSV, index_col=0)
    g = 100 * np.log(df['rtfpna']).diff() / df['labsh']
    mean = g.loc[int(SAMPLE[0]):int(SAMPLE[1])].mean()
    return g, g - mean, mean


def tfp_column(index, dev):
    """四半期のインデックスに、各年の第4四半期だけ値を置いた列"""
    y = pd.Series(np.nan, index=index)
    for yr, v in dev.dropna().items():
        t = pd.Timestamp(f'{yr}-10-01')
        if t in y.index and int(SAMPLE[0]) <= yr:
            y[t] = v
    return y


def augment(m, T, R, Z, d, q, me_sd=ME_SD):
    """GZ の6期ラグと年次TFPの測定誤差を状態に足し、年次TFPの観測行を加える"""
    n0, k0 = T.shape[0], R.shape[1]
    nl = len(W) - 1
    n1, k1 = n0 + nl + 1, k0 + 1
    T1 = np.zeros((n1, n1)); T1[:n0, :n0] = T
    R1 = np.zeros((n1, k1)); R1[:n0, :k0] = R
    g = m.idx['GZ']
    T1[n0, g] = 1.0                                             # L1_t = GZ_{t-1}
    for k in range(1, nl):
        T1[n0 + k, n0 + k - 1] = 1.0                            # Lk_t = L(k-1)_{t-1}
    R1[n0 + nl, k0] = me_sd                                     # 測定誤差（i.i.d.）
    row = np.zeros(n1)
    c = 400 / m.p['GZ_bar']                                     # GZ の乖離 → 年率%の伸び（観測方程式 (109) と同じ）
    row[g] = c * W[0]
    row[n0:n0 + nl] = c * W[1:]
    row[n0 + nl] = 1.0
    Z1 = np.hstack([Z, np.zeros((Z.shape[0], nl + 1))])
    Z1 = np.vstack([Z1, row])
    d1 = np.append(d, 0.0)
    q1 = np.hstack([q, np.ones((q.shape[0], 1))])
    return T1, R1, Z1, d1, q1


def run_with_tfp(theta, me_sd=ME_SD):
    m, T, R, Z, d, q = E.build(theta)
    T1, R1, Z1, d1, q1 = augment(m, T, R, Z, d, q, me_sd)
    Y = E.DATA.reindex(columns=hd.OBS).values.copy()
    for j, nm in enumerate(hd.OBS):
        if nm not in E.OBS_USED:
            Y[:, j] = np.nan
    _, dev, _ = load_tfp()
    Y = np.column_stack([Y, tfp_column(E.DATA.index, dev).values])
    xs, eta, ll = hd.kalman_smoother(T1, R1, Z1, d1, Y, q1)
    init, contrib = hd.decompose(T1, R1, xs, eta, m)
    return m, Z1, d1, Y, xs, eta, ll, init, contrib


def implied_annual_tfp(m, xs):
    """平滑化された GZ から、PWT と同じ「年平均水準の伸び」（乖離, %）を作る"""
    gz = pd.Series(400 / m.p['GZ_bar'] * xs[:, m.idx['GZ']], index=E.DATA.index)
    a = sum(W[k] * gz.shift(k) for k in range(len(W)))
    a = a[a.index.month == 10]
    a.index = a.index.year
    return a


def wage_by_shock(m, init, contrib, xs, index):
    q = parts_of(m, init, contrib, xs, 'wage_obs', index)
    j = m.idx['wage_obs']
    by = pd.DataFrame({s: contrib[:, m.sidx[s], j] for s in ALL_SHOCKS}, index=index)
    by['その他'] = q['その他']
    return q, by


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'fetch':
        print(fetch().tail()); sys.exit()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'IPAGothic'
    plt.rcParams['axes.unicode_minus'] = False
    data = E.DATA
    th = posterior_theta('mean')
    g, dev, gmean = load_tfp()
    print(f'PWT 労働増大型TFP成長率の平均（{SAMPLE[0]}〜{SAMPLE[1]}）: {gmean:.2f}%')

    # ベースライン（TFP観測なし）
    m0, T0, R0, Z0, d0, xs0, eta0, ll0, init0, contrib0 = run_est(th)
    q0, by0 = wage_by_shock(m0, init0, contrib0, xs0, data.index)
    a0 = implied_annual_tfp(m0, xs0)

    res = {}
    for sd in [ME_SD, 0.25, FORCE_SD]:
        m1, Z1, d1, Y1, xs1, eta1, ll1, init1, contrib1 = run_with_tfp(th, sd)
        q1, by1 = wage_by_shock(m1, init1, contrib1, xs1, data.index)
        res[sd] = dict(q=q1, by=by1, a=implied_annual_tfp(m1, xs1), ll=ll1)
        ok = data['wage'].notna()
        print(f'ME {sd}: 実質賃金の推計値と実績の差', round((q1['合計'] - q0['合計'])[ok].abs().max(), 6))
    r1, rf = res[ME_SD], res[FORCE_SD]
    name = {ME_SD: f'TFP観測あり（測定誤差{ME_SD}%pt）', 0.25: 'TFP観測あり（測定誤差0.25%pt）',
            FORCE_SD: f'TFP観測あり（測定誤差{FORCE_SD}%pt＝データに強く寄せる）'}

    # 表: 累積寄与（ショック別）
    rows = []
    for lab, start, end in [('2013Q1〜2023Q4', '2013-01-01', '2023-10-01'), ('2013Q1〜2026Q3', '2013-01-01', None),
                            ('2022Q1〜2026Q3', '2022-01-01', None)]:
        for nm, by in [('TFP観測なし', by0)] + [(name[sd], res[sd]['by']) for sd in res]:
            s = (by.loc[start:end] / 4).sum()
            rows.append({'期間': lab, 'ケース': nm, 'TFP成長': s['GZ'], '投資の限界効率': s['zeta'], '価格マークアップ': s['mup'],
                         '賃金マークアップ': s['muw'], '労働不効用': s['GN'], '供給計': s[hd.GROUPS['供給']].sum(),
                         '需要計': s[hd.GROUPS['需要']].sum(), '金融計': s[hd.GROUPS['金融']].sum(),
                         'その他': s['その他'], '合計': s.sum()})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(HERE, 'data', 'tfp_jp_wage_compare.csv'), index=False, float_format='%.3f')
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        print(tab.round(2).to_string(index=False))

    # 図1: 年次TFPの伸び（PWT）とモデルの推計
    yrs = np.arange(1996, 2027)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(dev.loc[1996:2023].index, dev.loc[1996:2023].values, color='#bbbbbb', edgecolor='k', lw=0.4,
           label=f'データ：PWT 11.0（労働増大型, 平均{gmean:.2f}%からの乖離）')
    ax.plot(yrs, a0.reindex(yrs).values, color='#d95f02', lw=2, marker='o', ms=4, label='モデル：TFP観測なし（いまの日本版）')
    ax.plot(yrs, r1['a'].reindex(yrs).values, color='#1f6fb5', lw=1.5, ls='--', marker='s', ms=3, label=name[ME_SD])
    ax.plot(yrs, rf['a'].reindex(yrs).values, color='#2ca02c', lw=2, marker='^', ms=4, label=name[FORCE_SD])
    c0 = a0.loc[2013:2023].sum(); cd = dev.loc[2013:2023].sum()
    ax.text(0.99, 0.97, f'2013〜2023年の累積：データ {cd:+.1f}%、モデル（観測なし） {c0:+.1f}%', transform=ax.transAxes,
            ha='right', va='top', fontsize=11, bbox=dict(fc='w', ec='gray'))
    ax.axhline(0, color='k', lw=0.8); ax.grid(alpha=0.3); ax.legend(loc='lower left', fontsize=9)
    ax.set_ylabel('%pt（年平均水準の伸び, 平均からの乖離）')
    ax.set_title('日本の TFP 成長率：データ（Penn World Table）とモデルが推理した値')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'figures', 'tfp_jp_annual.png'), dpi=130)

    # 図2: 実質賃金の水準（2012Q4=0）の分解を、TFP観測なし・ほぼ一致を強制で並べる
    from hd_jp import stacked
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharey=True)
    for ax, (q, ttl) in zip(axes, [(q0, 'TFP観測なし（いまの日本版）'), (rf['q'], name[FORCE_SD])]):
        lv = to_level(q, '2012-10-01').loc['2012-10-01':]
        stacked(ax, lv, step=2, line_label='累積乖離（推計値＝実績）')
        ax.set_title(f'実質賃金の水準（2012Q4＝0, トレンドからの累積乖離）：{ttl}')
        ax.set_ylabel('%')
    axes[0].legend(ncol=7, loc='lower left', fontsize=9)
    fig.suptitle('TFPのデータに合わせると、実質賃金の低迷の説明はどう変わるか（グループ別）', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(HERE, 'figures', 'tfp_jp_wage_level.png'), dpi=130)

    # 図3: ショック別の累積（2013Q1〜2023Q4: PWT がある期間）
    sub = tab[tab['期間'] == '2013Q1〜2023Q4'].set_index('ケース')
    cols = ['TFP成長', '投資の限界効率', '価格マークアップ', '賃金マークアップ', '労働不効用', '需要計', '金融計', 'その他', '合計']
    fig, ax = plt.subplots(figsize=(14, 5.5))
    xx = np.arange(len(cols)); w = 0.2
    colors = ['#d95f02', '#9ecae1', '#1f6fb5', '#2ca02c']
    for i, (cs, c) in enumerate(zip(sub.index, colors)):
        ax.bar(xx + (i - 1.5) * w, sub.loc[cs, cols].values, w, color=c, edgecolor='k', lw=0.4, label=cs)
    ax.set_xticks(xx); ax.set_xticklabels(cols); ax.axhline(0, color='k', lw=0.8); ax.grid(axis='y', alpha=0.3)
    ax.set_ylabel('実質賃金の累積乖離への寄与（%）'); ax.legend(fontsize=9, loc='lower right')
    ax.set_title('2013Q1〜2023Q4 の実質賃金の累積乖離（合計 −8.7%）をショック別に：TFP観測の有無で比べる')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'figures', 'tfp_jp_wage_shocks.png'), dpi=130)

    print('年次TFP（乖離）: データ / 観測なし / 観測あり')
    print(pd.DataFrame({'PWT': dev, 'なし': a0, 'あり': r1['a'], '強制': rf['a']}).loc[2008:2026].round(2))
    print('対数尤度: 観測なし', round(ll0, 1), ' あり', {sd: round(res[sd]['ll'], 1) for sd in res})
