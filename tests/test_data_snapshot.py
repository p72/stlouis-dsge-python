"""Issue #5: 原データのスナップショットから観測データを再生成でき、期間外の将来データで結果が変わらないこと"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

import data as D

HERE = D.HERE
FRED_DIR, RAW_DIR, END, _ = D.MODES['paper']


@pytest.fixture(scope='module')
def raw():
    return D.load_fred(FRED_DIR)


def _pub(name):
    return pd.read_csv(os.path.join(HERE, 'data', name), index_col=0, parse_dates=True)


def test_manifest_matches_files():
    man = json.load(open(os.path.join(FRED_DIR, 'manifest.json'), encoding='utf-8'))
    assert man['vintage_date'] == D.PAPER_VINTAGE and set(man['series']) == set(D.SERIES)
    for sid, v in man['series'].items():
        assert hashlib.sha256(open(os.path.join(FRED_DIR, v['file']), 'rb').read()).hexdigest() == v['sha256']
        assert v['column'].endswith('_' + D.PAPER_VINTAGE.replace('-', ''))     # ALFRED のビンテージ列
    rman = json.load(open(os.path.join(RAW_DIR, 'manifest.json'), encoding='utf-8'))
    for name, v in rman['files'].items():
        assert hashlib.sha256(open(os.path.join(RAW_DIR, name), 'rb').read()).hexdigest() == v['sha256']
    meta = json.load(open(os.path.join(HERE, 'data', 'observables_meta.json'), encoding='utf-8'))
    assert meta['mode'] == 'paper' and meta['start'] == D.START and meta['end'] == END
    assert meta['fred_vintage'] == D.PAPER_VINTAGE


def test_rebuild_from_snapshot_matches_published(raw):
    obs = D.build(raw, D.START, END)
    pub = _pub('observables.csv')
    pd.testing.assert_index_equal(obs.index, pub.index, check_names=False)
    np.testing.assert_allclose(obs.values, pub[obs.columns].values, atol=1e-6, equal_nan=True)
    src = D.paper_sources(RAW_DIR, D.START, END)
    pp = _pub('observables_paper.csv')
    np.testing.assert_allclose(src['tfp'].reindex(pp.index), pp['tfp'], atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(src['infexp'].reindex(pp.index), pp['infexp'], atol=1e-6, equal_nan=True)


def test_future_rows_and_revisions_do_not_change_paper_mode(raw):
    base = D.build(raw, D.START, END)
    fut = pd.date_range('2026-07-01', '2032-10-01', freq='QS')
    rng = np.random.default_rng(0)
    raw2 = {}
    for sid, s in raw.items():
        s = s.copy()
        s.loc[s.index > END] = s.loc[s.index > END] * 1.5       # 期間外の改定
        extra = pd.Series(s.dropna().iloc[-1] * np.exp(rng.normal(0, 0.3, len(fut))), index=fut)
        raw2[sid] = pd.concat([s[~s.index.isin(fut)], extra]).sort_index()
    again = D.build(raw2, D.START, END)
    pd.testing.assert_frame_equal(base, again, check_names=False)
    # 最新モード（終期なし）だと将来データで過去の加工値が変わる（HP・平均除去が全期間にかかる）
    latest = D.build(raw2, D.START, None)
    assert latest.index[-1] > pd.Timestamp(END)
    assert not np.allclose(latest.loc[:END, 'oil'], base['oil'])


def test_paper_sources_use_only_in_sample_rows():
    src = D.paper_sources(RAW_DIR, D.START, END)
    assert src.index.max() <= pd.Timestamp(END)
    f = D.fernald_tfp(os.path.join(RAW_DIR, 'quarterly_tfp.xlsx'))
    assert f.index.max() > pd.Timestamp(END)                     # 原データには期間外の行がある
    assert src.attrs['tfp_mean'] == pytest.approx(f.loc[D.START:END, 'tfp_model'].mean())
    late = D.paper_sources(RAW_DIR, D.START, f.index.max())
    assert late.attrs['tfp_mean'] != pytest.approx(src.attrs['tfp_mean'])


def test_growth_uses_previous_quarter(raw):
    obs = D.build(raw, D.START, END)
    assert obs.index[0] == pd.Timestamp(D.START)
    assert np.isfinite(obs.loc[D.START, 'wage'])                 # 1958Q4 を残しているので 1959Q1 の成長率が出る
