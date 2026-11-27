"""日本の実質賃金の歴史分解（jp/hd_jp_wage.py）"""
import numpy as np

import estimate_jp as E
import hd_jp_est as H
import hd_jp_wage as W


def test_wage_decomposition_adds_up_and_fits_data():
    th = H.posterior_theta('mean')
    q, by, mean, _ = W.wage_parts(th, E.DATA)
    groups = [c for c in q.columns if c != '合計']
    # グループ別寄与の和 = 平滑化推計値、ショック別寄与の和も同じ
    assert np.nanmax(np.abs(q[groups].sum(1) - q['合計'])) < 1e-8
    assert np.nanmax(np.abs(by.sum(1) - q['合計'])) < 1e-8
    # 賃金は測定誤差なしで観測しているので、推計値は実績（平均からの乖離）と一致する
    act = E.DATA['wage'] - mean
    assert np.nanmax(np.abs(q['合計'] - act)) < 1e-6
    assert abs(mean - E.DATA['wage'].mean()) < 1e-10
    # 水準は前期比/4 の累積で、基準期は0
    lv = W.to_level(q, '2012-10-01')
    assert abs(lv.loc['2012-10-01', '合計']) < 1e-12
    assert abs(lv['合計'].iloc[-1] - (q.loc['2013-01-01':, '合計'] / 4).sum()) < 1e-8
