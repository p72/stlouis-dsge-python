"""
Python版のIRFと原論文 付録DのIRF（図から数値化）を、各モデルの定常状態からの乖離で定量比較する（Issue #1）

使い方:  python irf_compare.py
入力:    data/paper_irf_digitized.csv, data/paper_irf_digitized_meta.csv（tools/digitize_paper_irf.py で作成）
出力:    data/irf_comparison.csv（ショック×変数ごとの指標）, figures/irf_compare_deviation.png

比較の条件
- ショックの大きさ: どちらも1標準偏差。Python版は表C-2の事後平均（default_params）の標準偏差
- 単位: 変数はすべて %pt（GDP成長率は年率, FF金利は年率, インフレは前年比）。図の縦軸「pp」と同じ
- 期間: 1〜40四半期。h=1 がショックの当期（Python版 Model.irf の t=0）
- 乖離: 論文は図中の破線（定常状態）からの差、Python版は Model.ss からの差
"""
import os

import numpy as np
import pandas as pd

from stlouis_dsge import Model, SHOCKS

HERE = os.path.dirname(os.path.abspath(__file__))
VARS = {'gdp_obs': 'GDP成長率', 'ffr': 'FF金利', 'core_yoy': 'コアインフレ', 'head_yoy': '総合インフレ'}
LABEL = {s[0]: s[2] for s in SHOCKS}
NOISE_PX = 2.0          # 読取り誤差の目安（ピクセル）。線の太さの中心を取るため


def python_irfs(m=None, T=40):
    m = m or Model().solve(verbose=False)
    rows = []
    for s, _, _ in SHOCKS:
        x = m.irf(s, T=T)
        for v in VARS:
            for h in range(T):
                rows.append(dict(shock=s, var=v, h=h + 1, dev=x[h, m.idx[v]], ss=m.ss[v]))
    return pd.DataFrame(rows), m


def metrics(p, q, tol):
    """p: 論文の乖離, q: Python版の乖離 (40,), tol: 論文側の読取り誤差"""
    ip = int(np.argmax(np.abs(p)))
    iq = int(np.argmax(q)) if p[ip] > 0 else int(np.argmin(q))   # Python側は論文のピークと同じ向きの極値
    rmse = float(np.sqrt(np.mean((p - q) ** 2)))
    scale = float(np.max(np.abs(p)))
    return dict(
        初期_論文=p[0], 初期_Python=q[0],
        ピーク_論文=p[ip], ピーク時点_論文=ip + 1, ピーク_Python=q[iq], ピーク時点_Python=iq + 1,
        ピーク比=q[iq] / p[ip] if abs(p[ip]) > tol else np.nan,
        累積_論文=p.sum(), 累積_Python=q.sum(),
        RMSE=rmse, 相対RMSE=rmse / scale if scale > tol else np.nan,
        相関=float(np.corrcoef(p, q)[0, 1]) if np.std(p) > 0 and np.std(q) > 0 else np.nan,
        符号一致=bool(np.sign(p[ip]) == np.sign(q[ip])) if abs(p[ip]) > tol else None,
        読取り誤差=tol,
    )


def judge(r):
    """ピーク比と相対RMSEによる大まかな判定（目安。しきい値は README に記載）"""
    if not np.isfinite(r['相対RMSE']):
        return '反応ほぼゼロ'
    if r['符号一致'] is False or not r['相関'] >= 0.5:
        return '不一致'
    if 0.8 <= r['ピーク比'] <= 1.25 and r['相対RMSE'] <= 0.25:
        return '近い'
    if 0.5 <= r['ピーク比'] <= 2.0 and r['相対RMSE'] <= 0.5:
        return '形は近いが大きさに差'
    return '差が大きい'


def compare():
    paper = pd.read_csv(os.path.join(HERE, 'data', 'paper_irf_digitized.csv'))
    meta = pd.read_csv(os.path.join(HERE, 'data', 'paper_irf_digitized_meta.csv')).set_index(['shock', 'var'])
    py, m = python_irfs()
    rows = []
    for s, _, _ in SHOCKS:
        for v in VARS:
            p = paper.query('shock == @s and var == @v').sort_values('h')['dev'].to_numpy()
            q = py.query('shock == @s and var == @v').sort_values('h')['dev'].to_numpy()
            tol = NOISE_PX * meta.loc[(s, v), 'val_per_px']
            r = dict(shock=s, ショック=LABEL[s], var=v, 変数=VARS[v],
                     定常_論文破線=meta.loc[(s, v), 'ss_line'], 定常_Python=m.ss[v], **metrics(p, q, tol))
            r['判定'] = judge(r)
            rows.append(r)
    return pd.DataFrame(rows), paper, py


def plot(paper, py, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'IPAGothic'
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(len(SHOCKS), len(VARS), figsize=(14, 2.1 * len(SHOCKS)), sharex=True)
    for i, (s, _, lab) in enumerate(SHOCKS):
        for j, v in enumerate(VARS):
            ax = axes[i, j]
            p = paper.query('shock == @s and var == @v').sort_values('h')
            q = py.query('shock == @s and var == @v').sort_values('h')
            ax.plot(p['h'], p['dev'], color='#0072BD', lw=2, label='論文（図から数値化）')
            ax.plot(q['h'], q['dev'], color='#D95319', lw=1.6, ls='--', label='Python版')
            ax.axhline(0, color='k', lw=0.6); ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(VARS[v])
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
    axes[0, 0].legend(fontsize=7, loc='upper right')
    for ax in axes[-1]:
        ax.set_xlabel('四半期')
    fig.suptitle('IRFの比較：定常状態からの乖離（%pt, 1標準偏差ショック）', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(path, dpi=110)


if __name__ == '__main__':
    res, paper, py = compare()
    res.to_csv(os.path.join(HERE, 'data', 'irf_comparison.csv'), index=False, float_format='%.4f')
    plot(paper, py, os.path.join(HERE, 'figures', 'irf_compare_deviation.png'))
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(res[['shock', 'var', '初期_論文', '初期_Python', 'ピーク_論文', 'ピーク時点_論文', 'ピーク_Python', 'ピーク時点_Python',
               'ピーク比', '累積_論文', '累積_Python', '相対RMSE', '相関', '判定']].round(3).to_string())
    print(res.groupby('shock', sort=False)['判定'].agg(lambda x: ' / '.join(x)))
