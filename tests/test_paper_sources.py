"""Issue #2: 論文3.2節どおりの TFP・10年インフレ期待と、歴史分解の再現"""
import os

import numpy as np
import pandas as pd
import pytest

import data as D
import hd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = pd.read_csv(os.path.join(HERE, 'data', 'paper_sources.csv'), index_col=0, parse_dates=True)


def test_tfp_model_consistent_and_demeaned():
    s = SRC.dropna(subset=['dtfp', 'alpha', 'dutil'])
    expect = s['dtfp'] / (1 - s['alpha']) - s['alpha'] / (1 - s['alpha']) * s['dutil']
    np.testing.assert_allclose(s['tfp_model'], expect, atol=1e-5)
    assert SRC['tfp'].loc[D.SAMPLE[0]:D.SAMPLE[1]].mean() == pytest.approx(0, abs=1e-5)


def test_infexp_splice():
    e = SRC.dropna(subset=['infexp'])
    assert e.index[0] == pd.Timestamp('1979-10-01')                       # 論文: 1979Q4 から
    pce = SRC['pce10'].dropna()
    assert pce.index[0] == pd.Timestamp('2007-01-01')
    np.testing.assert_allclose(e.loc['2007-01-01':, 'infexp'], e.loc['2007-01-01':, 'pce10'], atol=1e-6)
    before = e.loc[:'2006-10-01']
    gap = (before['cpi10'] - before['infexp']).round(4).unique()
    assert len(gap) == 1 and 0.1 < gap[0] < 0.3                           # CPI10 − 一定の平均差
    assert set(before['source']) == {'Livingston/Blue Chip CPI10 − 差', 'SPF CPI10 − 差'}


def test_paper_observables_only_replace_tfp_and_infexp():
    a = pd.read_csv(os.path.join(HERE, 'data', 'observables.csv'), index_col=0, parse_dates=True)
    p = pd.read_csv(os.path.join(HERE, 'data', 'observables_paper.csv'), index_col=0, parse_dates=True)
    other = [c for c in a.columns if c not in ('tfp', 'infexp')]
    pd.testing.assert_frame_equal(a[other], p[other])
    assert a['tfp'].isna().all() and p['tfp'].loc['1959':'2026-04'].notna().sum() == 270


@pytest.mark.parametrize('which,sfx', [('paper', ''), ('alt', '_alt')])
def test_hd_reproduces_published_csv(which, sfx):
    r = hd.run_hd(hd.load_observables(which, HERE))
    pub = pd.read_csv(os.path.join(HERE, 'data', f'historical_decomposition{sfx}.csv'), header=[0, 1], index_col=0)
    pub.index = pd.to_datetime(pub.index)
    for v in ['core_yoy', 'gap']:
        np.testing.assert_allclose(r['out'][v].values, pub[v][r['out'][v].columns].values, atol=1e-4)


def test_paper_figure_digitized():
    d = pd.read_csv(os.path.join(HERE, 'data', 'paper_hd_digitized.csv'), parse_dates=['date'])
    assert len(d) == 4 * 27 * 6
    core = d[d.figure == 'core_yoy'].groupby('date')['value'].sum()
    # 図4(a) の実績線（目視）: 2022Q1 約3.4, 2019Q4 約−0.45
    assert core.loc['2022-01-01'] == pytest.approx(3.35, abs=0.1)
    assert core.loc['2019-10-01'] == pytest.approx(-0.45, abs=0.1)


def test_rebuild_from_raw_matches_committed():
    s = D.paper_sources()
    for c in ['tfp', 'infexp']:
        np.testing.assert_allclose(s[c].reindex(SRC.index).values, SRC[c].values, atol=1e-5, equal_nan=True)
