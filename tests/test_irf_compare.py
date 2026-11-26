"""Issue #1: IRF の定量比較（定常状態からの乖離）"""
import numpy as np
import pandas as pd
import pytest

import irf_compare as C
from stlouis_dsge import Model, SHOCKS


def test_issue_reproduction_value():
    m = Model().solve(verbose=False)
    assert m.series(m.irf('GZ'), 'gdp_obs')[0] == pytest.approx(2.542336, abs=1e-5)


def test_digitized_data_complete():
    d = pd.read_csv(C.os.path.join(C.HERE, 'data', 'paper_irf_digitized.csv'))
    assert len(d) == len(SHOCKS) * 4 * 40
    assert d['dev'].notna().all()
    np.testing.assert_allclose(d['level'] - d['ss_line'], d['dev'], atol=2e-5)
    meta = pd.read_csv(C.os.path.join(C.HERE, 'data', 'paper_irf_digitized_meta.csv'))
    assert (meta['tick_fit_err'] < 0.005).all()
    # 定常状態の破線: GDP 1.875, FF 3.14, インフレ 1.98 付近
    med = meta.groupby('var')['ss_line'].median()
    assert med['gdp_obs'] == pytest.approx(1.875, abs=0.005)
    assert med['ffr'] == pytest.approx(3.1405, abs=0.005)
    assert med['core_yoy'] == pytest.approx(1.980, abs=0.005)


def test_metrics_identical_paths():
    p = np.sin(np.arange(40) / 5.0)
    r = C.metrics(p, p.copy(), tol=1e-4)
    assert r['RMSE'] == 0 and r['ピーク比'] == pytest.approx(1.0) and r['相関'] == pytest.approx(1.0)
    assert C.judge(r) == '近い'
    r2 = C.metrics(p, 1.5 * p, tol=1e-4)
    assert r2['ピーク比'] == pytest.approx(1.5) and C.judge(r2) == '形は近いが大きさに差'


def test_comparison_known_results():
    res, _, _ = C.compare()
    g = res.set_index(['shock', 'var'])
    # Issue #1 の例: 投資効率ショックのコアインフレのピーク（乖離）は論文 約0.25 vs Python 約0.36
    assert g.loc[('zeta', 'core_yoy'), 'ピーク_論文'] == pytest.approx(0.25, abs=0.02)
    assert g.loc[('zeta', 'core_yoy'), 'ピーク_Python'] == pytest.approx(0.355, abs=0.005)
    close = res.groupby('shock')['判定'].apply(lambda x: (x == '近い').all())
    assert set(close[close].index) == {'oS', 'chi', 'vt', 'mp', 'pistar'}
