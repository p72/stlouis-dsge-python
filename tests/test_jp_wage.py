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


def test_tfp_obs_weights_and_augmentation():
    """年次TFPの観測: 三角加重は合計1、観測行は GZ とそのラグにだけ乗る"""
    import tfp_jp as F
    assert abs(F.W.sum() - 1) < 1e-12
    th = H.posterior_theta('mean')
    m, T, R, Z, d, q = E.build(th)
    T1, R1, Z1, d1, q1 = F.augment(m, T, R, Z, d, q, 1.0)
    n0 = T.shape[0]
    assert T1.shape[0] == n0 + 7 and Z1.shape[0] == Z.shape[0] + 1 and q1.shape[1] == q.shape[1] + 1
    row = Z1[-1]
    c = 400 / m.p['GZ_bar']
    assert abs(row[m.idx['GZ']] - c / 16) < 1e-12 and np.allclose(row[n0:n0 + 6], c * F.W[1:])
    assert np.count_nonzero(row) == 8                     # GZ・ラグ6つ・測定誤差
    # 年次TFPを測定誤差大きめで足しても、賃金の推計値は実績のまま
    m1, Z1, d1, Y, xs, eta, ll, init, contrib = F.run_with_tfp(th, 1.0)
    q1, _ = F.wage_by_shock(m1, init, contrib, xs, E.DATA.index)
    ok = E.DATA['wage'].notna()
    assert np.nanmax(np.abs(q1['合計'][ok] - (E.DATA['wage'] - d[5])[ok])) < 1e-6


def test_hours_cases_fit_and_adjust():
    """時間当たりケース: 賃金は実績どおり、E.DATA は元に戻る、2011年の欠損は埋まる"""
    import hours_jp as HJ
    th = H.posterior_theta('mean')
    before = E.DATA.copy()
    r = HJ.run_case(th, 'C')
    assert E.DATA.equals(before)
    ok = r['data']['wage'].notna()
    assert np.nanmax(np.abs(r['q']['合計'] - (r['data']['wage'] - r['mean']))[ok]) < 1e-6
    h = HJ.load_hours()
    assert h['hours_per_worker_nsa'].notna().all() and h['filled'].sum() == 3
    dh = 400 * np.log(h['hours_per_worker_sa']).diff().reindex(before.index)
    assert np.allclose((r['data']['wage'] + dh).dropna(), before['wage'].reindex((r['data']['wage'] + dh).dropna().index))
