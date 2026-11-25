"""
本格版の歴史分解: 日本データで推定したパラメータ（事後平均）でカルマン平滑化し、お試し版（米国パラメータ）と比べる

使い方:  python jp/hd_jp_est.py   （先に estimate_jp.py の mode / hessian / mcmc を実行しておく）
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
from hd_jp import parts_of, stacked, run as run_trial           # noqa: E402
from stlouis_dsge import SHOCKS                                 # noqa: E402


def posterior_theta(which='mean'):
    zs = E.load_chains()
    if which == 'mode' or not zs:
        return E.to_theta(np.load(os.path.join(E.OUT, 'mode_z.npy')))
    th = np.array([E.to_theta(z) for z in np.vstack(zs)])
    return th.mean(0)


def run_est(theta):
    m, T, R, Z, d, q = E.build(theta)
    Y = E.DATA.reindex(columns=hd.OBS).values.copy()
    for j, nm in enumerate(hd.OBS):
        if nm not in E.OBS_USED:
            Y[:, j] = np.nan
    xs, eta, ll = hd.kalman_smoother(T, R, Z, d, Y, q)
    init, contrib = hd.decompose(T, R, xs, eta, m)
    return m, T, R, Z, d, xs, eta, ll, init, contrib


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'IPAGothic'
    plt.rcParams['axes.unicode_minus'] = False
    data = E.DATA
    fig_dir = os.path.join(HERE, 'figures')
    th = posterior_theta('mean')
    m, T, R, Z, d, xs, eta, ll, init, contrib = run_est(th)
    print('対数尤度（事後平均パラメータ）:', round(ll, 1))
    out = {v: parts_of(m, init, contrib, xs, v, data.index) for v in ['core_yoy', 'gap', 'rstar_ann', 'ffr', 'gdp_obs']}
    pd.concat(out, axis=1).to_csv(os.path.join(HERE, 'data', 'historical_decomposition_jp_est.csv'), float_format='%.4f')
    j = m.idx['core_yoy']
    det = pd.DataFrame({s[0]: contrib[:, m.sidx[s[0]], j] for s in SHOCKS}, index=data.index)
    det.to_csv(os.path.join(HERE, 'data', 'core_yoy_by_shock_jp_est.csv'), float_format='%.4f')

    # お試し版
    hd.ME = list(E.ME_SRC)
    mt, Tt, Rt, Zt, dt, xst, etat, llt, initt, contribt = run_trial(data)
    out_t = {v: parts_of(mt, initt, contribt, xst, v, data.index) for v in ['core_yoy', 'ffr', 'gap']}

    # 1) コアCPI 長期（本格版）
    df = out['core_yoy'].loc['1995-01-01':]
    fig, ax = plt.subplots(figsize=(15, 6.5))
    x = stacked(ax, df, step=4, line_label='平滑化推計値（2%からの乖離）')
    act = (data['core_yoy_raw'] - 2.0).reindex(df.index).values
    ax.plot(x, act, color='w', lw=0, marker='o', ms=3.5, mec='k', mew=0.6, label='実績（消費税込み, 参考）')
    ax.set_ylabel('%pt（2%からの乖離）')
    ax.set_title('日本のコアCPI（生鮮食品及びエネルギーを除く, 前年比）の歴史分解【本格版：日本データで推定】')
    ax.legend(ncol=8, loc='lower left', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, 'hd_jp_est_core_cpi.png'), dpi=130)

    # 2) お試し版との比較（グループ別寄与、3期間平均）
    periods = [('1995-01-01', '2012-10-01', 'デフレ期\n1995〜2012'), ('2013-01-01', '2021-10-01', '異次元緩和期\n2013〜2021'),
               ('2023-01-01', '2026-07-01', 'インフレ期\n2023〜2026Q3')]
    groups = list(hd.GROUPS) + ['その他']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, (a, b, lab) in zip(axes, periods):
        vt = out_t['core_yoy'].loc[a:b, groups].mean()
        ve = out['core_yoy'].loc[a:b, groups].mean()
        xx = np.arange(len(groups)); w = 0.38
        ax.bar(xx - w / 2, vt.values, w, color=[hd.COLORS[g] for g in groups], alpha=0.45, edgecolor='k', lw=0.5, hatch='//', label='お試し版（米国パラメータ）')
        ax.bar(xx + w / 2, ve.values, w, color=[hd.COLORS[g] for g in groups], edgecolor='k', lw=0.5, label='本格版（日本で推定）')
        ax.set_xticks(xx); ax.set_xticklabels(groups); ax.axhline(0, color='k', lw=0.8)
        ax.set_title(lab); ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('コアCPI前年比への平均寄与（%pt）')
    axes[0].legend(loc='lower left', fontsize=9)
    fig.suptitle('コアCPIの歴史分解：お試し版と本格版の比較（期間平均）', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(os.path.join(fig_dir, 'hd_jp_compare.png'), dpi=130)

    # 3) 影の金利の比較
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(data.index, out_t['ffr']['合計'] + dt[hd.OBS.index('ffr')], color='#999999', lw=1.3, label='お試し版（米国パラメータ）')
    ax.plot(data.index, out['ffr']['合計'] + d[hd.OBS.index('ffr')], color='#1f6fb5', lw=2.2, label='本格版（日本で推定）')
    ax.plot(data.index, data['call_raw'], color='#d95f02', lw=1.5, label='無担保コールレート（実績）')
    ax.axhline(0, color='k', lw=0.8); ax.grid(alpha=0.3); ax.legend(loc='upper right')
    ax.set_ylabel('%（年率）'); ax.set_title('政策金利：モデルの推計値（ゼロ金利期は影の金利）と実績')
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, 'hd_jp_est_shadow_rate.png'), dpi=130)

    # 4) 主要パラメータの事前・事後
    zs = E.load_chains()
    if zs:
        draws = np.array([E.to_theta(z) for z in np.vstack(zs)])
        show = ['etap', 'etaw', 'iotap', 'phiPi', 'phiY', 'rhor', 'rho_mp', 'rho_pistar', 'sig_pistar', 'rho_chi', 'sig_mp', 'psii']
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        for ax, nm in zip(axes.flat, show):
            i = E.NAMES.index(nm)
            lo, hi = np.percentile(draws[:, i], [0.5, 99.5])
            pr = E.PRIORS[i]
            lo2, hi2 = min(lo, pr.ppf(0.01)), max(hi, pr.ppf(0.99))
            g = np.linspace(lo2, hi2, 300)
            hh = ax.hist(draws[:, i], bins=50, density=True, color='#1f6fb5', alpha=0.6, label='事後分布（日本）')[0]
            ax.plot(g, pr.pdf(g), color='k', lw=1.5, ls='--', label='事前分布')
            ax.set_ylim(0, hh.max() * 1.25)                 # 端に張り付く事前密度で目盛りが潰れないように
            ax.axvline(E.US[nm], color='#d95f02', lw=2, label='米国の事後平均')
            ax.set_title(nm)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle('主なパラメータの事前分布と事後分布', fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(fig_dir, 'posterior_params.png'), dpi=130)

    print(out['core_yoy'].iloc[::8].round(2))
    print(det.loc[['2002-01-01', '2010-07-01', '2023-07-01', '2025-07-01', '2026-07-01']].round(2).T)
    for a, b, lab in periods:
        print(lab.replace('\n', ' '), '\n お試し', out_t['core_yoy'].loc[a:b, groups].mean().round(2).to_dict(),
              '\n 本格', out['core_yoy'].loc[a:b, groups].mean().round(2).to_dict())
