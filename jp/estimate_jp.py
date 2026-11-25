"""
日本版 本格推定: 日本の観測データでパラメータをベイズ推定する（事後モード → ランダムウォーク・メトロポリス）

使い方:
    python jp/estimate_jp.py mode            # 事後モード（途中経過を jp/estimation/mode_progress.json に保存）
    python jp/estimate_jp.py hessian         # モードでのヘッセ行列（MCMC の提案分布用）
    python jp/estimate_jp.py mcmc 6000 4     # 4本のチェーン × 6000 回（500回ごとに保存。再実行で続きから）
    python jp/estimate_jp.py summary         # 事後分布の要約表

方針
- 定常状態を決めるパラメータ（トレンド成長率・割引率・インフレ目標2%・財政比率など）は表1・C-1の値で固定。
  成長率系列の観測方程式の定数は日本のサンプル平均（hd_jp.py と同じ）。
- 推定するのは、名目・実物の摩擦、テイラー則、ショック過程（持続性・大きさ・MA項）、COVID ショックの大きさ。
- 事前分布は米国の事後平均を中心に置く（「米国の推定値を出発点に、日本のデータで更新する」）。
  持続性・指数化など 0〜1 のもの: ベータ分布、正のもの: ガンマ分布、ショックの標準偏差: 対数正規（対数の標準偏差1）。
- 観測しない移転支出・コンビニエンス・イールドのショックは米国の値で固定。
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import solve_discrete_lyapunov

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stlouis_dsge import Model, default_params          # noqa: E402
import hd                                                # noqa: E402

OUT = os.path.join(HERE, 'estimation')
OBS_USED = ['gdp', 'cons', 'inv', 'gov', 'wage', 'hours', 'ffr', 'core', 'oil', 'gas']
DEMEAN = ['gdp', 'cons', 'inv', 'gov', 'wage', 'hours', 'gas']

US = default_params()
ME_US = {nm: (rho, sig) for nm, rho, sig in hd.ME}
US.update(me_gdp=ME_US['gdp'][1], me_gas=ME_US['gas'][1], me_oil=ME_US['oil'][1])
# (名前, 分布, 標準偏差の指定)  beta: 平均=米国値, sd; gamma: 平均=米国値, sd=比率*平均; lnorm: 中央値=米国値, 対数sd
SPEC = [
    ('phi', 'gamma', 0.3), ('etap', 'gamma', 0.4), ('etaw', 'gamma', 0.4), ('psii', 'gamma', 0.4),
    ('sigmaa', 'gamma', 0.4), ('iotap', 'beta', 0.15), ('iotaw', 'beta', 0.15),
    ('phiPi', 'gamma', 0.3), ('phiY', 'gamma', 0.4), ('rhor', 'beta', 0.1),
    ('eta_mup', 'beta', 0.15), ('eta_muw', 'beta', 0.15),
]
for s in ['GZ', 'GN', 'zeta', 'mup', 'muw', 'oS', 'chi', 'g', 'mp', 'pistar']:
    SPEC.append(('rho_' + s, 'beta', 0.1))
for s in ['GZ', 'GN', 'zeta', 'mup', 'muw', 'oS', 'chi', 'g', 'mp', 'pistar', 'chi_cov', 'N_cov']:
    SPEC.append(('sig_' + s, 'lnorm', 1.0))
for s in ['me_gdp', 'me_gas', 'me_oil']:                       # 測定誤差の標準偏差
    SPEC.append((s, 'lnorm', 1.0))
NAMES = [s[0] for s in SPEC]
K = len(NAMES)


def _beta_ab(m, sd):
    sd = min(sd, 0.5 * np.sqrt(m * (1 - m)))
    v = m * (1 - m) / sd ** 2 - 1
    return m * v, (1 - m) * v


PRIORS = []
for nm, dist, a in SPEC:
    m = US[nm]
    if dist == 'beta':
        m = float(np.clip(m, 0.03, 0.97 if not nm.startswith('rho_pistar') else 0.995))
        PRIORS.append(stats.beta(*_beta_ab(m, a)))
    elif dist == 'gamma':
        sd = a * m
        PRIORS.append(stats.gamma(a=(m / sd) ** 2, scale=sd ** 2 / m))
    else:
        PRIORS.append(stats.lognorm(s=a, scale=m))
KIND = [s[1] for s in SPEC]


def to_theta(z):
    z = np.asarray(z, float)
    return np.array([1 / (1 + np.exp(-zi)) if k == 'beta' else np.exp(zi) for zi, k in zip(z, KIND)])


def to_z(th):
    return np.array([np.log(t / (1 - t)) if k == 'beta' else np.log(t) for t, k in zip(th, KIND)])


def log_jac(z):
    th = to_theta(z)
    return float(sum(np.log(t * (1 - t)) if k == 'beta' else np.log(t) for t, k in zip(th, KIND)))


def load_data():
    return pd.read_csv(os.path.join(HERE, 'data', 'observables_jp.csv'), index_col=0, parse_dates=True)


DATA = load_data()
ME_SRC = list(hd.ME)


def build(theta):
    th = dict(zip(NAMES, theta))
    hd.ME = [(nm, rho, th.get('me_' + nm, sig)) for nm, rho, sig in ME_SRC]
    m = Model({k: v for k, v in th.items() if not k.startswith('me_')})
    m.solve(verbose=False)
    T, R, Z, d = hd.state_space(m, DATA)
    for k in DEMEAN:
        d[hd.OBS.index(k)] = DATA[k].mean()
    q = hd.shock_cov(m, DATA.index)
    return m, T, R, Z, d, q


def kalman_loglik(T, R, Z, d, Y, qdiag, jitter=1e-6):
    """カルマンフィルターの対数尤度（数値安定のため P を毎期対称化し、コレスキー分解で解く）"""
    from scipy.linalg import cho_factor, cho_solve
    ns = T.shape[0]
    P = solve_discrete_lyapunov(T, R @ np.diag(qdiag[0]) @ R.T)
    P = 0.5 * (P + P.T)
    a = np.zeros(ns)
    ll = 0.0
    for t in range(Y.shape[0]):
        ok = ~np.isnan(Y[t])
        if ok.any():
            Zt = Z[ok]
            v = Y[t, ok] - d[ok] - Zt @ a
            PZ = P @ Zt.T
            F = Zt @ PZ + jitter * np.eye(ok.sum())
            cf = cho_factor(0.5 * (F + F.T))
            ll += -0.5 * (2 * np.log(np.diag(cf[0])).sum() + v @ cho_solve(cf, v) + ok.sum() * np.log(2 * np.pi))
            a = a + PZ @ cho_solve(cf, v)
            P = P - PZ @ cho_solve(cf, PZ.T)
        RQ = R * qdiag[min(t + 1, Y.shape[0] - 1)]
        a = T @ a
        P = T @ P @ T.T + RQ @ R.T
        P = 0.5 * (P + P.T)
    return ll


def log_prior(theta):
    return float(sum(p.logpdf(t) for p, t in zip(PRIORS, theta)))


def log_post_z(z):
    """変換後の空間での対数事後密度（ヤコビアン込み）"""
    z = np.asarray(z, float)
    if np.any(np.abs(z) > 30):
        return -1e10
    th = to_theta(z)
    lp = log_prior(th)
    if not np.isfinite(lp):
        return -1e10
    try:
        _, T, R, Z, d, q = build(th)
        Y = DATA.reindex(columns=hd.OBS).values.copy()
        for j, nm in enumerate(hd.OBS):
            if nm not in OBS_USED:
                Y[:, j] = np.nan
        ll = kalman_loglik(T, R, Z, d, Y, q)
    except Exception:
        return -1e10
    if not np.isfinite(ll):
        return -1e10
    return ll + lp + log_jac(z)


# ---------------------------------------------------------------------------
def _fd_grad(args):
    z, i, h = args
    e = np.zeros_like(z); e[i] = h
    return (log_post_z(z + e) - log_post_z(z - e)) / (2 * h)


def find_mode(z0, pool, maxiter=200):
    from scipy.optimize import minimize
    os.makedirs(OUT, exist_ok=True)
    best = {'f': -np.inf, 'z': z0}
    t0 = time.time()

    def fg(z):
        f = log_post_z(z)
        g = np.array(pool.map(_fd_grad, [(z, i, 1e-3) for i in range(K)]))
        if f > best['f']:
            best.update(f=f, z=z.copy())
            json.dump({'logpost': f, 'theta': dict(zip(NAMES, to_theta(z).tolist())),
                       'z': z.tolist(), 'elapsed_min': (time.time() - t0) / 60},
                      open(os.path.join(OUT, 'mode_progress.json'), 'w'), indent=1, ensure_ascii=False)
            print(f'{(time.time() - t0) / 60:6.1f}分  log事後 {f:10.2f}', flush=True)
        return -f, -g

    res = minimize(fg, z0, jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'gtol': 1e-3})
    return best['z'], best['f'], res


def _hess_entry(args):
    z, i, j, h, f0 = args
    def f(di, dj):
        e = np.zeros_like(z); e[i] += di; e[j] += dj
        return log_post_z(z + e)
    if i == j:
        return i, j, (f(h, 0) - 2 * f0 + f(-h, 0)) / h ** 2
    return i, j, (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h)) / (4 * h * h)


def hessian(z, pool, h=1e-3):
    f0 = log_post_z(z)
    tasks = [(z, i, j, h, f0) for i in range(K) for j in range(i, K)]
    H = np.zeros((K, K))
    for i, j, v in pool.map(_hess_entry, tasks):
        H[i, j] = H[j, i] = v
    return H


def _chain(args):
    cid, n, z0, L, scale, seed = args
    path = os.path.join(OUT, f'chain{cid}.npz')
    rng = np.random.default_rng(seed)
    if os.path.exists(path):                                     # 続きから
        old = np.load(path)
        draws, lps, acc = list(old['z']), list(old['lp']), int(old['acc'])
        z, f = draws[-1].copy(), lps[-1]
        rng = np.random.default_rng(seed + len(draws))
    else:
        z, f = z0 + scale * L @ rng.standard_normal(K) * 0.5, None
        f = log_post_z(z)
        draws, lps, acc = [], [], 0
    t0 = time.time()
    while len(draws) < n:
        zp = z + scale * L @ rng.standard_normal(K)
        fp = log_post_z(zp)
        if np.log(rng.uniform()) < fp - f:
            z, f = zp, fp; acc += 1
        draws.append(z.copy()); lps.append(f)
        if len(draws) % 500 == 0 or len(draws) == n:
            np.savez(path, z=np.array(draws), lp=np.array(lps), acc=acc)
            print(f'chain{cid}: {len(draws)}/{n}  採択率 {acc / len(draws):.2f}  '
                  f'{(time.time() - t0) / 60:.1f}分', flush=True)
    return cid


def z_start():
    return to_z(np.array([US[n] if KIND[i] != 'beta' else float(np.clip(US[n], 0.03, 0.995))
                          for i, n in enumerate(NAMES)]))


def load_chains(burn=0.3):
    zs = []
    for f in sorted(os.listdir(OUT)):
        if f.startswith('chain') and f.endswith('.npz'):
            z = np.load(os.path.join(OUT, f))['z']
            zs.append(z[int(burn * len(z)):])
    return zs


if __name__ == '__main__':
    from multiprocessing import Pool
    cmd = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    if cmd == 'mode':
        z0 = z_start()
        prog = os.path.join(OUT, 'mode_progress.json')
        if os.path.exists(prog):
            z0 = np.array(json.load(open(prog))['z'])
        print('出発点の log事後:', round(log_post_z(z0), 2))
        f_old = -np.inf
        with Pool(4) as pool:
            for rnd in range(30):                       # L-BFGS が途中で止まったら再起動する
                z0, f, res = find_mode(z0, pool, maxiter=int(sys.argv[2]) if len(sys.argv) > 2 else 200)
                print(f'ラウンド{rnd}: {res.message}  log事後 {f:.2f}', flush=True)
                if f - f_old < 0.05:
                    break
                f_old = f
        z = z0
        np.save(os.path.join(OUT, 'mode_z.npy'), z)
    elif cmd == 'hessian':
        z = np.load(os.path.join(OUT, 'mode_z.npy'))
        with Pool(4) as pool:
            H = hessian(z, pool)
        np.save(os.path.join(OUT, 'hessian.npy'), H)
        print('固有値（負なら極大）:', np.round(np.linalg.eigvalsh(H)[-5:], 3))
    elif cmd == 'mcmc':
        n, nc = int(sys.argv[2]), int(sys.argv[3])
        z = np.load(os.path.join(OUT, 'mode_z.npy'))
        H = np.load(os.path.join(OUT, 'hessian.npy'))
        w, V = np.linalg.eigh(-H)
        w = np.maximum(w, 1e-2 * w.max() * 1e-3 + 1e-6)          # 正定値化
        Sigma = V @ np.diag(1 / w) @ V.T
        L = np.linalg.cholesky(Sigma)
        scale = float(sys.argv[4]) if len(sys.argv) > 4 else 2.38 / np.sqrt(K) * 0.8
        with Pool(nc) as pool:
            pool.map(_chain, [(c, n, z, L, scale, 1000 + c) for c in range(nc)])
    elif cmd == 'summary':
        zs = load_chains()
        allz = np.vstack(zs)
        th = np.array([to_theta(z) for z in allz])
        mode = to_theta(np.load(os.path.join(OUT, 'mode_z.npy')))
        # R-hat
        chains = [np.array([to_theta(z) for z in c]) for c in zs]
        nmin = min(len(c) for c in chains)
        C = np.array([c[:nmin] for c in chains])
        W = C.var(axis=1, ddof=1).mean(0); B = nmin * C.mean(1).var(0, ddof=1)
        rhat = np.sqrt(((nmin - 1) / nmin * W + B / nmin) / W)
        rows = []
        for i, nm in enumerate(NAMES):
            rows.append(dict(パラメータ=nm, 事前分布=KIND[i], 米国=US[nm], 事前平均=PRIORS[i].mean(),
                             事後モード=mode[i], 事後平均=th[:, i].mean(),
                             下限5=np.percentile(th[:, i], 5), 上限95=np.percentile(th[:, i], 95),
                             Rhat=rhat[i]))
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT, 'posterior_summary.csv'), index=False, float_format='%.4f')
        print(df.round(4).to_string())
        print('総ドロー数:', len(allz))
