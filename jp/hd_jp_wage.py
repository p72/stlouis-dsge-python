"""
日本の実質賃金の歴史分解（本格版：日本データで推定したパラメータの事後平均）

使い方:  python jp/hd_jp_wage.py

観測の実質賃金 = 雇用者報酬（名目・季調）÷ GDPデフレーター ÷ 就業者数 の前期比年率（jp/README.md「観測データ」）。
観測方程式の定数は日本のサンプル平均（年0.59%）なので、分解は「平均的な伸びからの乖離」になる。

出力
- figures/hd_jp_wage_yoy.png    前年比（4四半期平均）の歴史分解 1995〜2026Q3
- figures/hd_jp_wage_level.png  水準（トレンドからの累積乖離）を 2012Q4=0 と 2021Q4=0 で
- figures/hd_jp_wage_shocks.png 2013年以降・2022年以降の累積寄与をショック別に
- data/hd_jp_wage.csv            前年比・水準の寄与（グループ別）
- data/hd_jp_wage_by_shock.csv   前期比年率の寄与（ショック別）
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
from hd_jp import parts_of, stacked                             # noqa: E402
from hd_jp_est import posterior_theta, run_est                  # noqa: E402
from stlouis_dsge import SHOCKS, COVID_SHOCKS                   # noqa: E402

ALL_SHOCKS = [x[0] for x in SHOCKS + COVID_SHOCKS]               # コロナ期だけ動く3ショックも含める

GROUPS = list(hd.GROUPS) + ['その他']
SHOCK_JP = {'GZ': 'TFP成長', 'GN': '人口成長', 'zeta': '定常TFP', 'mup': '価格マークアップ', 'muw': '賃金マークアップ',
            'N_cov': 'コロナ労働供給', 'chi': '割引率（需要）', 'vt': '投資効率', 'chi_cov': 'コロナ需要',
            'mp': '金融政策', 'pistar': 'トレンドインフレ', 'g': '政府支出', 't': '移転', 't_cov': 'コロナ移転',
            'oS': '原油供給'}


def wage_parts(theta, data):
    """実質賃金（前期比年率, 平均からの乖離）のグループ別寄与と、ショック別寄与を返す"""
    m, T, R, Z, d, xs, eta, ll, init, contrib = run_est(theta)
    q = parts_of(m, init, contrib, xs, 'wage_obs', data.index)
    j = m.idx['wage_obs']
    by = pd.DataFrame({s: contrib[:, m.sidx[s], j] for s in ALL_SHOCKS}, index=data.index)
    by['その他'] = q['その他']
    mean = d[hd.OBS.index('wage')]
    return q, by, mean, ll


def to_yoy(q):
    """前期比年率の4四半期平均 ≒ 前年比（対数）。寄与は線形なのでそのまま足せる"""
    return q.rolling(4).mean()


def to_level(q, base):
    """トレンド（サンプル平均の伸び）からの累積乖離（%）。base の四半期を0にする"""
    lv = (q / 4).cumsum()
    return lv - lv.loc[base]


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'IPAGothic'
    plt.rcParams['axes.unicode_minus'] = False
    data = E.DATA
    fig_dir, dat_dir = os.path.join(HERE, 'figures'), os.path.join(HERE, 'data')
    th, src = posterior_theta('mean', return_source=True)
    print('使用パラメータ:', src)
    q, by, mean, ll = wage_parts(th, data)
    print('対数尤度:', round(ll, 1), ' 観測方程式の定数（日本の平均, 年率%）:', round(mean, 3))
    act = data['wage'] - mean                                     # 実績（平均からの乖離）
    fit_err = (q['合計'] - act).abs().max()
    print('実績との当てはまり誤差（前期比年率）:', round(fit_err, 6))

    yoy = to_yoy(q)
    act_yoy = act.rolling(4).mean()
    lv13, lv22 = to_level(q, '2012-10-01'), to_level(q, '2021-10-01')
    out = pd.concat({'qoq_ann': q, 'yoy': yoy, 'level_2012Q4': lv13, 'level_2021Q4': lv22}, axis=1)
    out.to_csv(os.path.join(dat_dir, 'hd_jp_wage.csv'), float_format='%.4f')
    by.to_csv(os.path.join(dat_dir, 'hd_jp_wage_by_shock.csv'), float_format='%.4f')

    # 1) 前年比の歴史分解（長期）
    df = yoy.loc['1995-01-01':]
    fig, ax = plt.subplots(figsize=(15, 6.5))
    x = stacked(ax, df, step=4, line_label=f'平滑化推計値（平均{mean:.2f}%からの乖離）')
    ax.plot(x, act_yoy.reindex(df.index).values, color='w', lw=0, marker='o', ms=3.5, mec='k', mew=0.6, label='実績')
    for t0, lab in [('2013-01-01', '2013〜'), ('2022-01-01', '2022〜')]:
        ax.axvline(df.index.get_loc(pd.Timestamp(t0)) - 0.5, color='gray', ls='--', lw=1)
        ax.text(df.index.get_loc(pd.Timestamp(t0)), ax.get_ylim()[1] * 0.92, lab, color='gray', fontsize=10)
    ax.set_ylabel('%pt（前年比, 平均からの乖離）')
    ax.set_title('日本の実質賃金（雇用者報酬÷GDPデフレーター÷就業者数, 前年比）の歴史分解【本格版：日本データで推定】')
    ax.legend(ncol=8, loc='lower left', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, 'hd_jp_wage_yoy.png'), dpi=130)

    # 2) 水準（トレンドからの累積乖離）
    fig, axes = plt.subplots(2, 1, figsize=(15, 11))
    for ax, lv, base, ttl in [(axes[0], lv13, '2012-10-01', '2012Q4＝0, 2013年以降'),
                              (axes[1], lv22, '2021-10-01', '2021Q4＝0, 2022年以降')]:
        dfl = lv.loc[base:]
        xx = stacked(ax, dfl, step=2 if base < '2020' else 1, line_label='累積乖離（推計値）')
        acl = (act / 4).cumsum(); acl = acl - acl.loc[base]
        ax.plot(xx, acl.reindex(dfl.index).values, color='w', lw=0, marker='o', ms=3.5, mec='k', mew=0.6, label='実績')
        ax.set_ylabel('%（トレンドからの累積乖離）')
        ax.set_title(f'実質賃金の水準：年{mean:.2f}%の平均的な伸びからどれだけ離れたか（{ttl}）')
    axes[0].legend(ncol=8, loc='lower left', fontsize=9)
    fig.suptitle('日本の実質賃金の水準の歴史分解【本格版】', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(fig_dir, 'hd_jp_wage_level.png'), dpi=130)

    # 3) ショック別の累積寄与（2013Q1〜最新 と 2022Q1〜最新）
    last = data.index[-1]
    cols = ALL_SHOCKS + ['その他']
    cum = pd.DataFrame({
        '2013Q1〜': (by.loc['2013-01-01':, cols] / 4).sum(),
        '2022Q1〜': (by.loc['2022-01-01':, cols] / 4).sum()})
    grp_of = {s: g for g, ss in hd.GROUPS.items() for s in ss}; grp_of['その他'] = 'その他'
    keep = cum.abs().max(axis=1) > 0.3
    c2 = cum[keep].sort_values('2013Q1〜')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    for ax, col in zip(axes, cum.columns):
        ax.barh([SHOCK_JP.get(s, s) for s in c2.index], c2[col], color=[hd.COLORS[grp_of[s]] for s in c2.index],
                edgecolor='k', lw=0.4)
        ax.axvline(0, color='k', lw=0.8); ax.grid(axis='x', alpha=0.3)
        ax.set_title(f'{col}{last.year}Q{(last.month - 1) // 3 + 1}（合計 {cum[col].sum():+.1f}%）')
        ax.set_xlabel('トレンドからの累積乖離への寄与（%）')
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=hd.COLORS[g], label=g) for g in GROUPS], loc='lower right', fontsize=9)
    fig.suptitle('実質賃金の水準のずれをショック別に分けると（寄与の絶対値が0.3%未満のショックは省略）', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(os.path.join(fig_dir, 'hd_jp_wage_shocks.png'), dpi=130)

    # 数値の要約
    periods = [('1995-01-01', '2012-10-01', '1995〜2012'), ('2013-01-01', '2021-10-01', '2013〜2021'),
               ('2022-01-01', '2026-07-01', '2022〜2026Q3'), ('2023-01-01', '2023-10-01', '2023')]
    for a, b, lab in periods:
        print(lab, '前期比年率の平均寄与', q.loc[a:b, GROUPS + ['合計']].mean().round(2).to_dict(),
              ' 実績', round(act.loc[a:b].mean(), 2))
    print('水準 2012Q4=0 → 最新', lv13.iloc[-1].round(2).to_dict())
    print('水準 2021Q4=0 → 最新', lv22.iloc[-1].round(2).to_dict())
    print('ショック別累積\n', cum.round(2))
    print('年別前年比（Q4）\n', yoy.loc['2012-10-01'::4].round(2))
