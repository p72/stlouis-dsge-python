"""
日本版 歴史分解（お試し版）: 米国で推定された事後平均パラメータのまま、日本の観測データでカルマン平滑化する

使い方:  python jp/hd_jp.py   （先に jp/data_jp.py で jp/data/observables_jp.csv を作っておく）

米国版 hd.py からの変更点
- 実質成長率・賃金・労働時間・ガソリン価格の観測方程式の定数を日本のサンプル平均にする
  （日本のトレンド成長率は米国より低く、それを恒常的なショックと誤認しないため）
- コアインフレ（生鮮食品及びエネルギーを除くCPI）と政策金利の定数はモデルの定常状態のまま
  （インフレ定常状態＝2%。日銀の物価安定の目標と同じ。図は「2%からの乖離」になる）
- 観測しない系列（移転支出・TFP・10年インフレ期待・10年金利・CY）は全期間欠損
- コールレートはゼロ金利期（0.25%以下）を欠損にしているので、その間の政策金利はモデルの「影の金利」
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stlouis_dsge import Model                                   # noqa: E402
from hd import state_space, shock_cov, kalman_smoother, decompose, OBS, GROUPS, COLORS  # noqa: E402

DEMEAN = ['gdp', 'cons', 'inv', 'gov', 'wage', 'hours', 'gas']   # 定数をサンプル平均にする系列


def run(data):
    m = Model().solve(verbose=False)
    T, R, Z, d = state_space(m, data)
    for k in DEMEAN:
        d[OBS.index(k)] = data[k].mean()
    Y = data[OBS].values
    q = shock_cov(m, data.index)
    xs, eta, ll = kalman_smoother(T, R, Z, d, Y, q)
    init, contrib = decompose(T, R, xs, eta, m)
    return m, T, R, Z, d, xs, eta, ll, init, contrib


def parts_of(m, init, contrib, xs, var, index):
    j = m.idx[var]
    parts = {g: contrib[:, [m.sidx[s] for s in ss_], j].sum(1) for g, ss_ in GROUPS.items()}
    parts['その他'] = init[:, j] + contrib[:, m.ne:, j].sum(1)
    return pd.DataFrame({**parts, '合計': xs[:, j]}, index=index)


def stacked(ax, df, step=1, line_label='平滑化推計値'):
    x = np.arange(len(df)); pos = np.zeros(len(df)); neg = np.zeros(len(df))
    for g in list(GROUPS) + ['その他']:
        v = df[g].values
        ax.bar(x, np.where(v > 0, v, 0), bottom=pos, color=COLORS[g], edgecolor='k', lw=0.3, width=0.85, label=g)
        ax.bar(x, np.where(v < 0, v, 0), bottom=neg, color=COLORS[g], edgecolor='k', lw=0.3, width=0.85)
        pos += np.where(v > 0, v, 0); neg += np.where(v < 0, v, 0)
    ax.plot(x, df['合計'].values, color='k', lw=2, label=line_label)
    ax.axhline(0, color='k', lw=0.8)
    lab = [f"{t.year}Q{(t.month - 1) // 3 + 1}" for t in df.index]
    ax.set_xticks(x[::step]); ax.set_xticklabels(lab[::step], rotation=45, fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    return x


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'IPAGothic'
    plt.rcParams['axes.unicode_minus'] = False
    data = pd.read_csv(os.path.join(HERE, 'data', 'observables_jp.csv'), index_col=0, parse_dates=True)
    m, T, R, Z, d, xs, eta, ll, init, contrib = run(data)
    print('対数尤度:', round(ll, 1), ' 分解の誤差:', np.abs(init + contrib.sum(1) - xs).max())
    Y = data[OBS].values
    fit = d + xs @ Z.T
    print('コアCPIの当てはまり誤差:', np.nanmax(np.abs(fit[:, OBS.index('core')] - Y[:, OBS.index('core')])))
    print('定常状態: コアインフレ', round(d[OBS.index('core')], 2), '% / 政策金利', round(d[OBS.index('ffr')], 2), '%')

    from stlouis_dsge import SHOCKS, COVID_SHOCKS
    from hd import ME
    names = [s[0] for s in SHOCKS + COVID_SHOCKS] + ['me_' + x[0] for x in ME]
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
    os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
    pd.DataFrame(eta, index=data.index, columns=names).to_csv(
        os.path.join(HERE, 'data', 'smoothed_shocks_jp.csv'), float_format='%.5f')

    out = {v: parts_of(m, init, contrib, xs, v, data.index)
           for v in ['core_yoy', 'gap', 'rstar_ann', 'ffr', 'gdp_obs']}
    pd.concat(out, axis=1).to_csv(os.path.join(HERE, 'data', 'historical_decomposition_jp.csv'), float_format='%.4f')

    # 1) コアCPI（前年比）の長期歴史分解
    df = out['core_yoy'].loc['1995-01-01':]
    fig, ax = plt.subplots(figsize=(15, 6.5))
    x = stacked(ax, df, step=4, line_label='平滑化推計値（2%からの乖離）')
    act = (data['core_yoy_raw'] - 2.0).reindex(df.index).values
    ax.plot(x, act, color='w', lw=0, marker='o', ms=3.5, mec='k', mew=0.6, label='実績（消費税込み, 参考）')
    ax.set_ylabel('%pt（2%からの乖離）')
    ax.set_title('日本のコアCPI（生鮮食品及びエネルギーを除く, 前年比）の歴史分解【お試し版：米国パラメータ】')
    ax.legend(ncol=8, loc='lower left', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'figures', 'hd_jp_core_cpi.png'), dpi=130)

    # 2) 2019Q4 以降の4パネル
    targets = [('core_yoy', 'コアCPIインフレ率（前年比, 2%からの乖離, %pt）'), ('gdp_obs', '1人当たり実質GDP成長率（年率, 平均からの乖離, %pt）'),
               ('gap', '産出ギャップ（%）'), ('rstar_ann', '自然利子率 r*（年率, 定常状態からの乖離, %pt）')]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, (v, t) in zip(axes.flat, targets):
        stacked(ax, out[v].loc['2019-10-01':], step=2)
        ax.set_title(t)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=7, fontsize=11)
    fig.suptitle('日本の歴史分解（2019Q4〜2026Q3）【お試し版：米国パラメータ】', fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(os.path.join(HERE, 'figures', 'hd_jp_since2019.png'), dpi=130)

    # 3) 影の政策金利（モデルが推計した金利）とコールレート
    fig, ax = plt.subplots(figsize=(13, 5))
    shadow = out['ffr']['合計'] + d[OBS.index('ffr')]
    ax.plot(data.index, shadow, color='#1f6fb5', lw=2, label='モデルの政策金利（平滑化推計値、ゼロ金利期は「影の金利」）')
    ax.plot(data.index, data['call_raw'], color='#d95f02', lw=1.5, label='無担保コールレート（実績）')
    ax.scatter(data.index[data['ffr'].notna()], data['ffr'].dropna(), s=12, color='k', zorder=3, label='観測に使った四半期')
    ax.axhline(0, color='k', lw=0.8); ax.grid(alpha=0.3); ax.legend(loc='upper right')
    ax.set_ylabel('%（年率）'); ax.set_title('政策金利：モデルの推計値と実績')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'figures', 'hd_jp_shadow_rate.png'), dpi=130)

    print(out['core_yoy'].iloc[::8].round(2))
    print(out['core_yoy'].loc['2021-01-01':].iloc[::2].round(2))
    print('影の金利:', shadow.iloc[::8].round(2).to_dict())
