"""Issue #3: 日本版の事後平均がチェーン不在時にモードへ黙って切り替わらないこと"""
import os
import shutil

import numpy as np
import pandas as pd
import pytest

import estimate_jp as E
import hd_jp_est as H

PUB = E.OUT
MODE = E.to_theta(np.load(os.path.join(PUB, 'mode_z.npy')))
THIN = pd.read_csv(os.path.join(PUB, H.DRAWS_CSV))[E.NAMES].to_numpy().mean(0)


def _copy(tmp_path, *names):
    for n in names:
        shutil.copy(os.path.join(PUB, n), tmp_path / n)
    return str(tmp_path)


def test_mode_explicit():
    th, src = H.posterior_theta('mode', return_source=True)
    np.testing.assert_allclose(th, MODE)
    assert 'モード' in src


def test_mean_without_chains_uses_published_draws_not_mode(tmp_path):
    d = _copy(tmp_path, 'mode_z.npy', H.DRAWS_CSV)
    th, src = H.posterior_theta('mean', out_dir=d, return_source=True)
    np.testing.assert_allclose(th, THIN)
    assert not np.allclose(th, MODE)
    assert H.DRAWS_CSV in src and '平均' in src


def test_mean_csv_takes_precedence_and_columns_by_name(tmp_path):
    d = _copy(tmp_path, H.DRAWS_CSV)
    vals = np.arange(1, E.K + 1, dtype=float) / 1000
    pd.DataFrame([vals], columns=E.NAMES)[E.NAMES[::-1]].to_csv(tmp_path / H.MEAN_CSV, index=False)
    th, src = H.posterior_theta('mean', out_dir=d, return_source=True)
    np.testing.assert_allclose(th, vals)
    assert H.MEAN_CSV in src


def test_mean_from_chains(tmp_path):
    rng = np.random.default_rng(0)
    z0 = E.to_z(THIN)
    chains = [z0 + 0.01 * rng.standard_normal((50, E.K)) for _ in range(2)]
    for c, z in enumerate(chains):
        np.savez(tmp_path / f'chain{c}.npz', z=z)
    th, src = H.posterior_theta('mean', out_dir=str(tmp_path), return_source=True)
    burn = int(0.3 * 50)
    expect = np.array([E.to_theta(z) for z in np.vstack([c[burn:] for c in chains])]).mean(0)
    np.testing.assert_allclose(th, expect)
    assert 'チェーン 2 本' in src


def test_mean_missing_artifacts_raises(tmp_path):
    _copy(tmp_path, 'mode_z.npy')                      # モードだけあっても平均の代わりにはしない
    with pytest.raises(FileNotFoundError):
        H.posterior_theta('mean', out_dir=str(tmp_path))


def test_mode_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        H.posterior_theta('mode', out_dir=str(tmp_path))


def test_bad_which():
    with pytest.raises(ValueError):
        H.posterior_theta('median')


def test_clean_checkout_reproduces_published_hd():
    """チェーン無しの公開成果物だけで、README掲載の事後平均による歴史分解を再生成できる"""
    from hd_jp import parts_of
    th = H.posterior_theta('mean')
    m, T, R, Z, d, xs, eta, ll, init, contrib = H.run_est(th)
    out = parts_of(m, init, contrib, xs, 'core_yoy', E.DATA.index)
    ref = pd.read_csv(os.path.join(E.HERE, 'data', 'historical_decomposition_jp_est.csv'), header=[0, 1], index_col=0)['core_yoy']
    ref.index = pd.to_datetime(ref.index)
    ref = ref.reindex(out.index)[out.columns]
    # 公開CSVは全チェーン（42,000ドロー）の平均、こちらは間引き4,200ドローの平均なので差は数千分の1%pt
    assert np.nanmax(np.abs(out.values - ref.values)) < 0.01
    assert round(out.loc['2023-01-01':, '金融'].mean(), 2) == 1.22
