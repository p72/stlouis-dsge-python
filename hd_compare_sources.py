"""
歴史分解の観測データ感度（Issue #2）: 代替データ版と原典準拠版（Fernald TFP・SPF期待インフレ）を比べ、
論文の図4・図5（数値化）とグループ別寄与の大きさ・符号・時期を比較する

使い方:  python hd_compare_sources.py
入力:    data/observables.csv（代替）, data/observables_paper.csv（原典準拠）, data/paper_hd_digitized.csv（図4・5の数値化）
出力:    data/hd_compare_groups.csv（変数×グループ×版ごとの指標）, data/hd_compare_periods.csv（期間平均）,
         data/hd_sensitivity_shocks.csv（ショック別の感度）, figures/hd_compare_sources.png
"""
import os

import numpy as np
import pandas as pd

import hd
from stlouis_dsge import Model, SHOCKS, COVID_SHOCKS

HERE = os.path.dirname(os.path.abspath(__file__))
VARS = {'core_yoy': 'コアPCEインフレ（前年比）', 'head_yoy': '総合PCEインフレ（前年比）', 'gap': '産出ギャップ', 'rstar_ann': '自然利子率'}
GROUPS = list(hd.GROUPS) + ['その他']
VERSIONS = {'alt': '代替データ版', 'paper': '原典準拠版'}
PERIODS = [('2020-01-01', '2021-10-01', '2020〜21年'), ('2022-01-01', '2023-10-01', '2022〜23年'),
           ('2024-01-01', '2026-04-01', '2024〜26Q2')]
START = '2019-10-01'


def load_paper_fig():
    d = pd.read_csv(os.path.join(HERE, 'data', 'paper_hd_digitized.csv'), parse_dates=['date'])
    return {v: d[d.figure == v].pivot(index='date', columns='group', values='value')[GROUPS] for v in VARS}, d


def run_versions():
    m = Model().solve(verbose=False)
    return {k: hd.run_hd(hd.load_observables(k, HERE), m) for k in VERSIONS}


def group_metrics(fig, res, digit):
    rows = []
    clipped = digit.groupby(['figure', 'date'])['clipped'].first()
    for v in VARS:
        P = fig[v]
        ok = ~clipped.loc[v].reindex(P.index).fillna(False).values       # 図で切れている棒は除く
        tol = 3 * digit[digit.figure == v]['val_per_px'].iloc[0]
        for g in GROUPS:
            p = P[g].values[ok]
            for k in VERSIONS:
                q = res[k]['out'][v].loc[START:, g].reindex(P.index).values[ok]
                big = np.abs(p) > max(tol, 0.1 * np.abs(p).max())
                rows.append(dict(var=v, 変数=VARS[v], group=g, version=k, 版=VERSIONS[k],
                                 平均_論文=p.mean(), 平均_Python=q.mean(),
                                 平均絶対誤差=np.abs(p - q).mean(), 論文の平均絶対値=np.abs(p).mean(),
                                 相関=np.corrcoef(p, q)[0, 1] if p.std() > 0 and q.std() > 0 else np.nan,
                                 符号一致率=(np.sign(p[big]) == np.sign(q[big])).mean() if big.any() else np.nan,
                                 比較した四半期=int(ok.sum())))
    return pd.DataFrame(rows)


def period_means(fig, res):
    rows = []
    for v in VARS:
        for a, b, lab in PERIODS:
            row = dict(var=v, 変数=VARS[v], 期間=lab)
            for g in GROUPS:
                row[f'{g}_論文'] = fig[v].loc[a:b, g].mean()
                for k in VERSIONS:
                    row[f'{g}_{VERSIONS[k]}'] = res[k]['out'][v].loc[a:b, g].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def shock_sensitivity(res):
    """ショック別寄与の差（原典準拠版 − 代替版）。全期間の二乗平均と 2019Q4 以降の平均"""
    m = res['paper']['m']
    rows = []
    for v in ['core_yoy', 'gap']:
        j = m.idx[v]
        for s, _, lab in SHOCKS + COVID_SHOCKS:
            c = {k: pd.Series(res[k]['contrib'][:, m.sidx[s], j], index=res[k]['out'][v].index) for k in VERSIONS}
            dlt = c['paper'] - c['alt']
            rows.append(dict(var=v, 変数=VARS[v], shock=s, ショック=lab,
                             代替_2019Q4以降平均=c['alt'].loc[START:].mean(), 原典準拠_2019Q4以降平均=c['paper'].loc[START:].mean(),
                             差_2019Q4以降平均=dlt.loc[START:].mean(), 差_全期間RMS=float(np.sqrt((dlt ** 2).mean()))))
    return pd.DataFrame(rows)


def plot(fig, res, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'IPAGothic'
    plt.rcParams['axes.unicode_minus'] = False
    f, axes = plt.subplots(len(VARS), len(GROUPS), figsize=(18, 11), sharex=True)
    for i, v in enumerate(VARS):
        for jj, g in enumerate(GROUPS):
            ax = axes[i, jj]
            x = fig[v].index
            ax.bar(x, fig[v][g], width=70, color=hd.COLORS[g], alpha=0.55, label='論文（図4・5を数値化）')
            ax.plot(x, res['paper']['out'][v].loc[START:, g], color='k', lw=1.8, label='原典準拠版')
            ax.plot(x, res['alt']['out'][v].loc[START:, g], color='#d62728', lw=1.2, ls='--', label='代替データ版')
            ax.axhline(0, color='k', lw=0.6); ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(g)
            if jj == 0:
                ax.set_ylabel(VARS[v], fontsize=9)
            ax.tick_params(axis='x', labelrotation=45, labelsize=7)
    axes[0, 0].legend(fontsize=7, loc='upper left')
    f.suptitle('歴史分解のグループ別寄与：論文の図4・5、原典準拠版、代替データ版（2019Q4〜2026Q2, %pt）', fontsize=13)
    f.tight_layout(rect=[0, 0, 1, 0.97])
    f.savefig(path, dpi=110)


if __name__ == '__main__':
    fig, digit = load_paper_fig()
    res = run_versions()
    gm = group_metrics(fig, res, digit)
    pm = period_means(fig, res)
    ss = shock_sensitivity(res)
    gm.to_csv(os.path.join(HERE, 'data', 'hd_compare_groups.csv'), index=False, float_format='%.4f')
    pm.to_csv(os.path.join(HERE, 'data', 'hd_compare_periods.csv'), index=False, float_format='%.4f')
    ss.to_csv(os.path.join(HERE, 'data', 'hd_sensitivity_shocks.csv'), index=False, float_format='%.4f')
    plot(fig, res, os.path.join(HERE, 'figures', 'hd_compare_sources.png'))
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(gm.pivot_table(index=['var', 'group'], columns='version', values=['平均絶対誤差', '相関', '符号一致率'], sort=False).round(2))
    print(gm.groupby(['var', 'version'], sort=False)['平均絶対誤差'].mean().unstack().round(3))
    print(ss.sort_values('差_全期間RMS', ascending=False).head(10).round(3).to_string())
