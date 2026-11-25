"""
カルマン平滑化による構造ショックの推定と歴史分解（論文 4.1 節, 図4・図5, 付録B）

- 観測方程式 (100)-(114) と AR(1) 測定誤差 7 本を状態空間に組み込む
- COVID 特殊ショックは有効期間（3.3 節）だけ分散1、それ以外は0の時変共分散で扱う
- Durbin-Koopman の平滑化でショックを取り出し、ショックごとの寄与を積み上げる
- パラメータは論文の事後平均（推定はしない）

使い方:  python hd.py   （先に data.py で data/observables.csv を作っておく）
"""
import os
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from stlouis_dsge import Model, SHOCKS, COVID_SHOCKS

ME = [  # 名前, rho, sigma（表C-2の事後平均）
    ('gdp', 0.0807, 0.0030), ('tfp', 0.0773, 0.0088), ('oil', 0.9530, 0.0910),
    ('gas', 0.4501, 0.0051), ('infexp', 0.9195, 0.0017), ('tp', 0.9616, 0.0036),
    ('cy', 0.9638, 0.0013)]
TP = 0.0099                       # 期間プレミアムの切片
OBS = ['gdp', 'cons', 'inv', 'gov', 'tr', 'wage', 'hours', 'ffr', 'core', 'tfp',
       'oil', 'gas', 'infexp', 't10', 'cy']
COVID_WINDOWS = {'chi_cov': ('2020-01-01', '2020-07-01'),
                 't_cov': ('2020-04-01', '2022-01-01'),
                 'N_cov': ('2020-01-01', '2020-10-01')}
GROUPS = {
    '供給': ['GZ', 'GN', 'zeta', 'mup', 'muw', 'N_cov'],
    '需要': ['chi', 'vt', 'chi_cov'],
    '金融': ['mp', 'pistar'],
    '財政': ['g', 't', 't_cov'],
    '原油': ['oS'],
}
COLORS = {'供給': '#1f6fb5', '需要': '#d95f02', '金融': '#e6ab02', '財政': '#7b2fd0',
          '原油': '#2ca02c', 'その他': '#5bc0eb'}


def state_space(m, data):
    """x_t = T x_{t-1} + R eta_t,  y_t = d + Z x_t"""
    n, ne, nme = m.n, m.ne, len(ME)
    ix, ss, p = m.idx, m.ss, m.p
    T = np.zeros((n + nme, n + nme)); T[:n, :n] = m.P
    R = np.zeros((n + nme, ne + nme)); R[:n, :ne] = m.Q
    for j, (_, rho, sig) in enumerate(ME):
        T[n + j, n + j] = rho; R[n + j, ne + j] = sig
    e = lambda v: np.eye(n + nme)[ix[v]]
    me = lambda nm: np.eye(n + nme)[n + [x[0] for x in ME].index(nm)]
    # 将来の期待値の和: sum_i P^i
    Pp = np.zeros((n + nme, n + nme)); Pp[:n, :n] = m.P
    pw, S40_0, S40_1, S80 = np.eye(n + nme), 0, 0, 0
    for i in range(81):
        if i < 40: S40_0 = S40_0 + pw
        if 1 <= i <= 40: S40_1 = S40_1 + pw
        if i < 80: S80 = S80 + pw
        pw = Pp @ pw
    Rb, Pib, Ra = ss['R'], ss['Pi'], ss['Raaa']
    gmean = data['gdp'].mean()
    cobs = {k: data[k].mean() - gmean for k in ['cons', 'inv', 'gov', 'tr', 'wage']}
    rows, d = [], []
    def add(row, const): rows.append(row); d.append(const)
    add(e('gdp_obs') + 400 * me('gdp'), ss['gdp_obs'])                       # (100)
    for k in ['cons', 'inv', 'gov', 'tr', 'wage']:                           # (101)-(105)
        add(e(k + '_obs'), ss[k + '_obs'] + cobs[k])
    add(e('hours_obs'), ss['hours_obs'])                                     # (106)
    add(e('ffr'), ss['ffr'])                                                 # (107)
    add(400 / Pib * e('Pi'), 400 * np.log(Pib))                              # (108)
    add(400 / p['GZ_bar'] * e('GZ') + 400 * me('tfp'), 0.0)                  # (109)
    add(e('oil_obs') + me('oil'), 0.0)                                       # (110)
    add(e('gas_obs') + 400 * me('gas'), ss['gas_obs'])                       # (111)
    add(100 * Pib ** 3 / 10 * e('Pih') @ S40_1 + 100 * me('infexp'),
        100 * (Pib ** 4 - 1))                                                # (112)
    add(100 * Rb ** 3 / 10 * e('R') @ S40_0 + 100 * me('tp'),
        100 * (Rb ** 4 - 1) + 100 * TP)                                      # (113)
    add(100 / 20 * (Ra ** 3 * e('Raaa') - Rb ** 3 * e('R')) @ S80 + 100 * me('cy'),
        100 * (Ra ** 4 - Rb ** 4))                                           # (114)
    return T, R, np.array(rows), np.array(d)


def shock_cov(m, dates):
    """時点ごとのショック共分散（対角）。COVID ショックは有効期間のみ1"""
    q = np.ones((len(dates), m.ne + len(ME)))
    for k, (a, b) in COVID_WINDOWS.items():
        j = m.sidx[k]
        q[:, j] = ((dates >= a) & (dates <= b)).astype(float)
    return q


def kalman_smoother(T, R, Z, d, Y, qdiag):
    nt, ns = Y.shape[0], T.shape[0]
    Q0 = np.diag(qdiag[0])
    P = solve_discrete_lyapunov(T, R @ Q0 @ R.T)                  # 無条件分散で初期化
    a = np.zeros(ns)
    store = []
    loglik = 0.0
    for t in range(nt):
        ok = ~np.isnan(Y[t])
        Zt = Z[ok]
        if ok.any():
            v = Y[t, ok] - d[ok] - Zt @ a
            F = Zt @ P @ Zt.T + 1e-8 * np.eye(ok.sum())
            Fi = np.linalg.inv(F)
            K = T @ P @ Zt.T @ Fi
            loglik += -0.5 * (np.linalg.slogdet(F)[1] + v @ Fi @ v + ok.sum() * np.log(2 * np.pi))
        else:
            v, Fi, K = np.zeros(0), np.zeros((0, 0)), np.zeros((ns, 0))
        L = T - K @ Zt
        store.append((a.copy(), P.copy(), v, Fi, L, Zt))
        Qn = np.diag(qdiag[min(t + 1, nt - 1)])
        a = T @ a + K @ v
        P = T @ P @ L.T + R @ Qn @ R.T
    # 後ろ向き平滑化
    r = np.zeros(ns)
    xs = np.zeros((nt, ns)); eta = np.zeros((nt, R.shape[1]))
    for t in range(nt - 1, -1, -1):
        a_t, P_t, v, Fi, L, Zt = store[t]
        if t < nt - 1:
            eta[t + 1] = qdiag[t + 1] * (R.T @ r)                   # eta_{t+1} = Q R' r_t
        r = Zt.T @ Fi @ v + L.T @ r                                 # r_{t-1}
        xs[t] = a_t + P_t @ r
    return xs, eta, loglik


def decompose(T, R, xs, eta, m):
    """x_t = T^{t} x_1 + sum_s T^{t-s} R eta_s をショックごとに分解"""
    nt, ns = xs.shape
    contrib = np.zeros((nt, R.shape[1], ns))
    init = np.zeros((nt, ns)); init[0] = xs[0]
    for t in range(1, nt):
        init[t] = T @ init[t - 1]
        contrib[t] = contrib[t - 1] @ T.T + (R * eta[t]).T
    return init, contrib


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'IPAGothic'
    plt.rcParams['axes.unicode_minus'] = False
    here = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(here, 'data', 'observables.csv'), index_col=0, parse_dates=True)
    data = data.loc[:'2026-04-01']                                  # 論文と同じ 2026Q2 まで
    m = Model().solve(verbose=False)
    T, R, Z, d = state_space(m, data)
    Y = data[OBS].values
    q = shock_cov(m, data.index)
    xs, eta, ll = kalman_smoother(T, R, Z, d, Y, q)
    init, contrib = decompose(T, R, xs, eta, m)
    print('対数尤度:', round(ll, 1), ' 分解の誤差:', np.abs(init + contrib.sum(1) - xs).max())
    fit = d + xs @ Z.T
    ok = ~np.isnan(Y)
    print('観測値との誤差（測定誤差なしの系列で0になるはず）: core', np.nanmax(np.abs(fit[:, 8] - Y[:, 8])))

    names = [s[0] for s in SHOCKS + COVID_SHOCKS] + ['me_' + x[0] for x in ME]
    pd.DataFrame(eta, index=data.index, columns=names).to_csv(os.path.join(here, 'data', 'smoothed_shocks.csv'), float_format='%.5f')

    targets = [('core_yoy', 'コアPCEインフレ率（前年比, %pt）'), ('head_yoy', '総合インフレ率（前年比, %pt）'),
               ('gap', '産出ギャップ（%）'), ('rstar_ann', '自然利子率 r*（年率, %pt）')]
    sel = data.index >= '2019-10-01'
    dates = data.index[sel]
    labels = [f"{d_.year}Q{(d_.month - 1) // 3 + 1}" for d_ in dates]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    out = {}
    for ax, (var, title) in zip(axes.flat, targets):
        j = m.idx[var]
        parts = {g: contrib[:, [m.sidx[s] for s in ss_], j].sum(1) for g, ss_ in GROUPS.items()}
        me_part = contrib[:, m.ne:, j].sum(1)
        parts['その他'] = init[:, j] + me_part
        total = xs[:, j]
        out[var] = pd.DataFrame({**parts, '合計': total}, index=data.index)
        x = np.arange(sel.sum())
        pos = np.zeros(sel.sum()); neg = np.zeros(sel.sum())
        for g, val in parts.items():
            v = val[sel]
            ax.bar(x, np.where(v > 0, v, 0), bottom=pos, color=COLORS[g], edgecolor='k', lw=0.4, label=g)
            ax.bar(x, np.where(v < 0, v, 0), bottom=neg, color=COLORS[g], edgecolor='k', lw=0.4)
            pos += np.where(v > 0, v, 0); neg += np.where(v < 0, v, 0)
        ax.plot(x, total[sel], color='k', lw=2, label='実績・平滑化推計値')
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(x[::2]); ax.set_xticklabels(labels[::2], rotation=45, fontsize=8)
        ax.set_title(title); ax.grid(axis='y', alpha=0.3)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=7, fontsize=11)
    fig.suptitle('歴史分解（2019Q4〜2026Q2, 定常状態からの乖離）', fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(os.path.join(here, 'figures', 'hd_simplified.png'), dpi=130)

    # コアPCE単体（論文図4(a)と同じ形）
    fig, ax = plt.subplots(figsize=(11, 6))
    df = out['core_yoy'][sel]
    x = np.arange(len(df)); pos = np.zeros(len(df)); neg = np.zeros(len(df))
    for g in list(GROUPS) + ['その他']:
        v = df[g].values
        ax.bar(x, np.where(v > 0, v, 0), bottom=pos, color=COLORS[g], edgecolor='k', lw=0.4, label=g)
        ax.bar(x, np.where(v < 0, v, 0), bottom=neg, color=COLORS[g], edgecolor='k', lw=0.4)
        pos += np.where(v > 0, v, 0); neg += np.where(v < 0, v, 0)
    ax.plot(x, df['合計'], color='k', lw=2, label='実績')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, fontsize=9)
    ax.set_ylabel('%pt（定常状態2%からの乖離）')
    ax.set_title('コアPCEインフレ率（前年比）の歴史分解')
    ax.legend(ncol=4, loc='upper right'); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(here, 'figures', 'hd_core_pce.png'), dpi=130)
    pd.concat(out, axis=1).to_csv(os.path.join(here, 'data', 'historical_decomposition.csv'), float_format='%.4f')
    print(out['core_yoy'].loc['2021-01-01':'2026-04-01'].iloc[::2].round(2))
