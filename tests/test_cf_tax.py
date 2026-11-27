"""消費税の反実仮想（jp/cf_tax_jp.py）: 消したショックの経路が歴史分解の寄与と一致すること"""
import numpy as np
import pandas as pd

import cf_tax_jp as C
import estimate_jp as E
from hd_jp_est import posterior_theta, run_est


def test_removed_path_matches_decomposition():
    th = posterior_theta('mean')
    m, T, R, Z, d, xs, eta, ll, init, contrib = run_est(th)
    idx = E.DATA.index
    # 全期間・1ショックを消した経路 = 歴史分解のそのショックの寄与
    j = m.sidx['chi']
    x = C.removed_path(T, R, eta, [j], list(range(len(idx))))
    np.testing.assert_allclose(x, contrib[:, j, :], atol=1e-10)
    # 何も消さなければ実績のまま
    cf = C.counterfactual(m, T, R, xs, eta, idx, [], [-1, 0])
    np.testing.assert_allclose(cf, xs)
    # 消すのは増税の前期と当期だけ（それより前は実績と同じ）
    cf = C.counterfactual(m, T, R, xs, eta, idx, C.REAL10, [-1, 0], hikes=['2014-04-01'])
    t = idx.get_loc(pd.Timestamp('2014-01-01'))
    np.testing.assert_allclose(cf[:t], xs[:t])
    assert not np.allclose(cf[t], xs[t])
