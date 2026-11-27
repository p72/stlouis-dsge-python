"""
反実仮想: 「消費税率の引き上げ（1997Q2・2014Q2・2019Q4）がなかったら」コアCPI・GDP・政策金利はどうなっていたか

使い方:  python jp/cf_tax_jp.py [mean|mode]   （既定は mean。事後平均の探し方は hd_jp_est.posterior_theta と同じ）

やり方（平滑化ショックを消す反実仮想）
  このモデルには消費税がない（消費税率 τc は定常状態を決める定数）。そこで、日本データで推定したパラメータで
  カルマン平滑化し、増税の前後の四半期に出てくる構造ショックを「増税が生んだもの」とみなしてゼロにする。
  モデルは線形なので、反実仮想の経路 = 平滑化推計値 − （消したショックだけで動かした経路）。
    x_cf(t) = x(t) − x_rem(t),   x_rem(t) = T x_rem(t−1) + R η_rem(t)
  既定（基準ケース）は「駆け込みの四半期（増税の前期）と増税の四半期」の12個の構造ショックをすべて消す。
  ほかに、消すショックや期間を変えた感応度ケースも計算する。

出力:
  figures/cf_tax_jp_paths.png     1995〜2026Q3 の実績（平滑化推計値）と反実仮想（コアCPI・GDP水準・産出ギャップ・政策金利）
  figures/cf_tax_jp_events.png    増税ごとの前後3年（感応度ケースの幅つき）
  figures/cf_tax_jp_shocks.png    増税の前後に平滑化されたショック（標準化）
  data/cf_tax_jp.csv              反実仮想と実績の経路（基準ケースと感応度ケース）
  data/cf_tax_jp_summary.csv      増税ごとの効果（その回の増税だけを消す。増税の4・8・12期後）
  data/cf_tax_jp_by_shock.csv     ケースCの効果をショック別に分けたもの（増税8期後）
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import estimate_jp as E                                         # noqa: E402
import hd                                                       # noqa: E402
from hd_jp_est import posterior_theta, run_est                  # noqa: E402
from stlouis_dsge import SHOCKS                                 # noqa: E402

HIKES = {'1997Q2（3→5%）': '1997-04-01', '2014Q2（5→8%）': '2014-04-01', '2019Q4（8→10%）': '2019-10-01'}
ALL12 = [s[0] for s in SHOCKS]
REAL10 = [s for s in ALL12 if s not in ('mp', 'pistar')]       # 金融（日銀の動き）以外の10ショック
DEMAND = ['chi', 'vt', 'zeta']                                  # 需要（限界効用・CY）と投資の限界効率
# ケース名: (消すショック, 増税の四半期からの相対位置)。None は「観測を欠損にして平滑化し直す」
# 主な3ケース（読み方の違い）
MAIN = {
    'A 一時的な振れだけ消す（観測を欠損）': (None, [-1, 0]),
    'B 需要・投資ショックとして消す': (DEMAND, [-1, 0]),
    'C 金融以外の10ショックを消す': (REAL10, [-1, 0]),
}
# 感応度（幅を見るだけ）
SENS = {
    'C′ 金融ショックも消す（12ショック）': (ALL12, [-1, 0]),
    'C″ 金融以外・当期のみ': (REAL10, [0]),
    'C‴ 金融以外・前期〜翌期': (REAL10, [-1, 0, 1]),
}
CASES = {**MAIN, **SENS}
COL = dict(zip(MAIN, ['#2ca02c', '#d95f02', '#d62728']))
VARS = ['core_yoy', 'gdp_obs', 'cons_obs', 'gap', 'ffr']


def removed_path(T, R, eta, cols, rows):
    """eta の (rows, cols) だけで状態を動かした経路"""
    e = np.zeros_like(eta)
    e[np.ix_(rows, cols)] = eta[np.ix_(rows, cols)]
    x = np.zeros((eta.shape[0], T.shape[0]))
    for t in range(eta.shape[0]):
        x[t] = (T @ x[t - 1] if t else 0) + R @ e[t]
    return x


def counterfactual(m, T, R, xs, eta, index, shocks, offsets, hikes=None):
    hikes = list(HIKES.values()) if hikes is None else hikes
    rows = sorted({index.get_loc(pd.Timestamp(h)) + k for h in hikes for k in offsets})
    cols = [m.sidx[s] for s in shocks]
    return xs - removed_path(T, R, eta, cols, rows)


def masked_smooth(theta, index, offsets, hikes=None):
    """増税の前後の四半期を全系列欠損にして平滑化し直す。前後のデータから内挿されるので、一時的な振れだけが消える"""
    m, T, R, Z, d, q = E.build(theta)
    Y = E.DATA.reindex(columns=hd.OBS).values.copy()
    for j, nm in enumerate(hd.OBS):
        if nm not in E.OBS_USED:
            Y[:, j] = np.nan
    for h in (list(HIKES.values()) if hikes is None else hikes):
        for k in offsets:
            Y[index.get_loc(pd.Timestamp(h)) + k] = np.nan
    return hd.kalman_smoother(T, R, Z, d, Y, q)[0]


def levels(m, d, x):
    """モデルの状態（定常状態からの乖離）を読みやすい水準に直す"""
    j = m.idx
    out = pd.DataFrame({
        'core_yoy': x[:, j['core_yoy']] + 2.0,                           # コアCPI前年比（%、インフレ定常2%）
        'gap': x[:, j['gap']],                                           # 産出ギャップ（%）
        'ffr': x[:, j['ffr']] + d[hd.OBS.index('ffr')],                  # 政策金利（%、ゼロ金利期は影の金利）
        'gdp_obs': x[:, j['gdp_obs']],                                   # 1人当たり実質GDP成長率（年率、平均からの乖離）
        'cons_obs': x[:, j['cons_obs']],
    })
    out['gdp_lvl'] = out['gdp_obs'].cumsum() / 4                        # 1人当たり実質GDPの水準（累積, %）
    out['cons_lvl'] = out['cons_obs'].cumsum() / 4
    return out


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'IPAGothic'
    plt.rcParams['axes.unicode_minus'] = False
    fig_dir, dat_dir = os.path.join(HERE, 'figures'), os.path.join(HERE, 'data')
    which = sys.argv[1] if len(sys.argv) > 1 else 'mean'
    th, src = posterior_theta(which, return_source=True)
    print('使用パラメータ:', src)
    m, T, R, Z, d, xs, eta, ll, init, contrib = run_est(th)
    idx = E.DATA.index
    act = levels(m, d, xs).set_index(idx)
    def make(v, hikes=None):
        x = masked_smooth(th, idx, v[1], hikes) if v[0] is None else counterfactual(m, T, R, xs, eta, idx, *v, hikes=hikes)
        return levels(m, d, x).set_index(idx)
    cfs = {c: make(v) for c, v in CASES.items()}                                  # 3回の増税をまとめて消す（長期の図）
    cfs_h = {n: {c: make(v, [h]) for c, v in CASES.items()} for n, h in HIKES.items()}   # 1回ずつ消す（増税ごとの効果）

    # CSV
    tab = pd.concat({'実績（平滑化推計値）': act, **cfs}, axis=1)
    tab.to_csv(os.path.join(dat_dir, 'cf_tax_jp.csv'), float_format='%.4f')
    rows = []
    for name, h in HIKES.items():
        t0 = idx.get_loc(pd.Timestamp(h))
        for c, cf in cfs_h[name].items():
            for k in [4, 8, 12]:
                t = t0 + k
                if t >= len(idx):
                    continue
                dlt = cf.iloc[t] - act.iloc[t]
                rows.append({'増税': name, 'ケース': c, '何期後': k, '時点': idx[t].strftime('%Y-%m'),
                             'コアCPI前年比の差(%pt)': dlt['core_yoy'],
                             '1人当たり実質GDP水準の差(%)': dlt['gdp_lvl'] - (cf['gdp_lvl'].iloc[t0 - 2] - act['gdp_lvl'].iloc[t0 - 2]),
                             '実質消費水準の差(%)': dlt['cons_lvl'] - (cf['cons_lvl'].iloc[t0 - 2] - act['cons_lvl'].iloc[t0 - 2]),
                             '産出ギャップの差(%pt)': dlt['gap'], '政策金利の差(%pt)': dlt['ffr']})
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(dat_dir, 'cf_tax_jp_summary.csv'), index=False, float_format='%.3f')
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        print(summ[summ['ケース'].isin(MAIN)].round(2).to_string(index=False))
        print(summ[summ['何期後'] == 8].round(2).to_string(index=False))

    # 図1: 長期の経路（主な3ケース）
    raw = E.DATA['core_yoy_raw']
    st = '1995-01-01'
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    spec = [('core_yoy', 'コアCPI（生鮮食品及びエネルギーを除く）前年比（%）', False),
            ('gdp_lvl', '1人当たり実質GDPの水準：「増税なし」−実績（%、累積）', True),
            ('gap', '産出ギャップ（%）', False),
            ('ffr', '政策金利（%、ゼロ金利期はモデルの影の金利）', False)]
    for ax, (v, ttl, diff) in zip(axes, spec):
        if not diff:
            ax.plot(act.loc[st:].index, act.loc[st:, v], color='k', lw=2.2, label='実績（モデルの平滑化推計値）')
        for nm in MAIN:
            y = cfs[nm][v] - act[v] if diff else cfs[nm][v]
            ax.plot(y.loc[st:].index, y.loc[st:], color=COL[nm], lw=1.6, ls='--' if not diff else '-', label=f'増税なし：{nm}')
        if v == 'core_yoy':
            ax.plot(raw.loc[st:].index, raw.loc[st:], color='w', lw=0, marker='o', ms=3.2, mec='#555', mew=0.6,
                    label='実績（消費税込み, 参考）')
        if v == 'ffr':
            ax.plot(E.DATA['call_raw'].loc[st:].index, E.DATA['call_raw'].loc[st:], color='#7f7f7f', lw=1, label='無担保コールレート（実績）')
        for h in HIKES.values():
            ax.axvline(pd.Timestamp(h), color='#7b2fd0', lw=1, alpha=0.6)
        ax.axhline(0, color='k', lw=0.6); ax.grid(alpha=0.3); ax.set_title(ttl); ax.legend(loc='best', fontsize=8.5)
    axes[0].text(0.01, 0.03, '黒線は増税した四半期のコアCPIを欠損にして推計したもので、税の直接効果（物価の段差）はもともと入っていない',
                 transform=axes[0].transAxes, fontsize=9, color='#333')
    fig.suptitle('反実仮想：消費税増税（1997・2014・2019年, 紫の縦線）がなかったら【本格版パラメータ】', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(fig_dir, 'cf_tax_jp_paths.png'), dpi=130)

    # 図2: 増税ごとの前後（主な3ケース＋感応度の幅）
    panels = [('core_yoy', 'コアCPI前年比の差（%pt）'), ('gdp_lvl', '1人当たり実質GDP水準の差（%）'),
              ('cons_lvl', '1人当たり実質消費水準の差（%）'), ('ffr', '政策金利の差（%pt）')]
    fig, axes = plt.subplots(len(panels), 3, figsize=(16, 14), sharey='row')
    q = np.arange(-4, 13)
    for c_i, (name, h) in enumerate(HIKES.items()):
        t0 = idx.get_loc(pd.Timestamp(h))
        rng = [t0 + k for k in q if 0 <= t0 + k < len(idx)]
        kk = np.array(rng) - t0
        for r_i, (v, ttl) in enumerate(panels):
            ax = axes[r_i, c_i]
            cur = {}
            for nm in CASES:
                dd = (cfs_h[name][nm][v] - act[v])
                if v.endswith('_lvl'):
                    dd = dd - dd.iloc[t0 - 2]                   # 駆け込みの前の期を0にそろえる
                cur[nm] = dd.iloc[rng].values
            band = np.array([cur[n] for n in ['C 金融以外の10ショックを消す', *SENS]])
            ax.fill_between(kk, band.min(0), band.max(0), color='#d62728', alpha=0.12, label='Cの感応度の幅')
            for nm in MAIN:
                ax.plot(kk, cur[nm], color=COL[nm], lw=2, marker='o', ms=2.5, label=nm)
            ax.axhline(0, color='k', lw=0.8); ax.axvline(0, color='#7b2fd0', lw=1); ax.grid(alpha=0.3)
            if r_i == 0:
                ax.set_title(name)
            if c_i == 0:
                ax.set_ylabel(ttl)
            if r_i == len(panels) - 1:
                ax.set_xlabel('増税の四半期からの期数')
    axes[1, 0].legend(fontsize=8.5, loc='upper left')
    fig.suptitle('「増税なし」−実績：増税ごとの前後3年（プラス＝増税がなければ高かった）', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(fig_dir, 'cf_tax_jp_events.png'), dpi=130)

    # ショックごとの寄与（ケースCの中身。増税8期後）
    rows = []
    for name, h in HIKES.items():
        t0 = idx.get_loc(pd.Timestamp(h))
        for s_ in ALL12:
            x = removed_path(T, R, eta, [m.sidx[s_]], [t0 - 1, t0])
            rows.append({'増税': name, 'ショック': s_, 'コアCPI前年比(%pt)': -x[t0 + 8, m.idx['core_yoy']],
                         'GDP水準(%)': -x[t0 - 1:t0 + 9, m.idx['gdp_obs']].sum() / 4, '政策金利(%pt)': -x[t0 + 8, m.idx['ffr']]})
    pd.DataFrame(rows).to_csv(os.path.join(dat_dir, 'cf_tax_jp_by_shock.csv'), index=False, float_format='%.3f')
    print(pd.DataFrame(rows).pivot(index='ショック', columns='増税', values='GDP水準(%)').round(2))

    # 図3: 増税の前後の平滑化ショック（標準化）
    S = pd.DataFrame(eta[:, :len(ALL12)], index=idx, columns=ALL12)
    Sz = S / S.std()
    lab = {s[0]: s[2].replace('ショック', '') for s in SHOCKS}
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    ks = [-2, -1, 0, 1, 2]
    w = 0.16
    for ax, (name, h) in zip(axes, HIKES.items()):
        t0 = idx.get_loc(pd.Timestamp(h))
        y = np.arange(len(ALL12))
        for i, k in enumerate(ks):
            ax.barh(y + (i - 2) * w, Sz.iloc[t0 + k].values, w,
                    color=plt.cm.coolwarm(i / (len(ks) - 1)), edgecolor='k', lw=0.3,
                    label=f'{k:+d}期（{idx[t0 + k].year}Q{(idx[t0 + k].month - 1) // 3 + 1}）')
        ax.axvline(0, color='k', lw=0.8); ax.grid(axis='x', alpha=0.3)
        ax.set_yticks(y); ax.set_yticklabels([lab[s] for s in ALL12]); ax.invert_yaxis()
        ax.set_title(name); ax.set_xlabel('標準偏差の何倍か'); ax.legend(fontsize=8, loc='lower right')
    fig.suptitle('増税の前後にモデルが見つけたショック（平滑化ショック ÷ 全期間の標準偏差）', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(os.path.join(fig_dir, 'cf_tax_jp_shocks.png'), dpi=130)
    print(Sz.iloc[[idx.get_loc(pd.Timestamp(h)) + k for h in HIKES.values() for k in [-1, 0]]].round(1))
