"""Issue #4: 通常の simulate() で COVID特殊ショックが毎期発生しないこと"""
import numpy as np
import pandas as pd
import pytest
from scipy.linalg import solve_discrete_lyapunov

import hd
from stlouis_dsge import Model, SHOCKS, COVID_SHOCKS, COVID_WINDOWS


@pytest.fixture(scope='module')
def m():
    return Model().solve(verbose=False)


def _old_simulate(m, T, seed, q):
    rng = np.random.default_rng(seed)
    x = np.zeros(m.n); out = np.zeros((T, m.n))
    for t in range(T):
        x = m.P @ x + m.Q @ (q[t] * rng.standard_normal(m.ne))
        out[t] = x
    return m.ybar + out


def _covid_idx(m):
    return [m.sidx[k] for k, _, _ in COVID_SHOCKS]


def test_default_has_no_covid(m):
    q = m.shock_scale(50)
    assert (q[:, _covid_idx(m)] == 0).all()
    assert (np.delete(q, _covid_idx(m), axis=1) == 1).all()
    # 既定の simulate は通常12ショックだけで動かした結果と一致（通常ショックの実現値は同じ seed で同じ）
    np.testing.assert_allclose(m.simulate(T=50, seed=3), _old_simulate(m, 50, 3, q))


def test_normal_shocks_reproducible(m):
    a = m.simulate(T=40, seed=7)
    b = m.simulate(T=40, seed=7)
    np.testing.assert_array_equal(a, b)
    # covid=True（旧来の挙動）は旧実装と一致し、既定とは違う
    allon = m.simulate(T=40, seed=7, covid=True)
    np.testing.assert_allclose(allon, _old_simulate(m, 40, 7, np.ones((40, m.ne))))
    assert not np.allclose(allon, a)


def test_dates_activate_only_inside_windows(m):
    dates = pd.date_range('2018-01-01', '2023-10-01', freq='QS')
    q = m.shock_scale(dates=dates)
    for k, (a, b) in COVID_WINDOWS.items():
        inside = (dates >= a) & (dates <= b)
        assert (q[inside, m.sidx[k]] == 1).all()
        assert (q[~inside, m.sidx[k]] == 0).all()     # 期間外の革新はゼロ
    assert q[dates == '2020-07-01', m.sidx['chi_cov']] == 1
    assert q[dates == '2020-10-01', m.sidx['chi_cov']] == 0
    y = m.simulate(seed=1, dates=dates)
    np.testing.assert_allclose(y, _old_simulate(m, len(dates), 1, q))


def test_explicit_shock_scale(m):
    s = np.zeros(m.ne); s[m.sidx['mp']] = 1.0
    y = m.simulate(T=30, seed=0, shock_scale=s)
    assert np.isfinite(y).all()
    with pytest.raises(ValueError):
        m.shock_scale(T=5, dates=pd.date_range('2020-01-01', periods=4, freq='QS'))


def test_hd_shares_windows(m):
    dates = pd.date_range('2019-01-01', '2022-10-01', freq='QS')
    q = hd.shock_cov(m, dates)
    np.testing.assert_array_equal(q[:, :m.ne], m.shock_scale(dates=dates))
    assert (q[:, m.ne:] == 1).all()
    assert hd.COVID_WINDOWS is COVID_WINDOWS


def test_unconditional_sd_normal_shocks(m):
    """Issue記載の値: 通常12ショックのGDP成長率の無条件標準偏差は約4.20（全15本だと約16.46）"""
    j = m.idx['gdp_obs']
    sd = {}
    for k, Q in (('all', m.Q), ('normal', m.Q[:, :len(SHOCKS)])):
        V = solve_discrete_lyapunov(m.P, Q @ Q.T)
        sd[k] = np.sqrt(V[j, j])
    assert sd['normal'] == pytest.approx(4.20, abs=0.01)
    assert sd['all'] == pytest.approx(16.46, abs=0.01)
    # 既定の shock_scale で作った分散も通常12ショックと同じ
    Qd = m.Q * m.shock_scale(1)[0]
    V = solve_discrete_lyapunov(m.P, Qd @ Qd.T)
    assert np.sqrt(V[j, j]) == pytest.approx(sd['normal'])
