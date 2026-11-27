"""
日本版インパルス応答（IRF）: 日本データで推定したパラメータ（事後平均）で12ショックのIRFを描き、米国版と重ねて比べる

使い方:  python jp/irf_jp.py [mean|mode]   （既定は mean。事後平均の探し方は hd_jp_est.posterior_theta と同じ）
出力:
  figures/irf_jp_us_group{1..4}.png   12ショック×4変数（GDP成長率・政策金利・コア/総合インフレ前年比）の日米比較
  figures/irf_jp_us_gap_rstar.png     産出ギャップと自然利子率 r* の日米比較
  figures/irf_jp_us_mp_per1pt.png    金融政策ショックを「衝撃時の政策金利 +1%pt」にそろえた日米比較（効き方の比較）
  data/irf_jp_us_summary.csv          ショックごとの衝撃時・ピーク・半減期の日米比較
IRFはモデルとパラメータだけで決まる（観測データは使わない）。日本版の差はすべて推定パラメータの違いから来る。
定常状態（インフレ2%, 政策金利3.15%）は日米共通（推定で固定したパラメータで決まるため）。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import estimate_jp as E                                         # noqa: E402
from hd_jp_est import posterior_theta                           # noqa: E402
from stlouis_dsge import Model, SHOCKS                          # noqa: E402

PANELS = [('gdp_obs', 'GDP成長率（年率）'), ('ffr', '政策金利（年率, 日本=コールレート/米国=FF金利）'),
          ('core_yoy', 'コアインフレ率（前年比）'), ('head_yoy', '総合インフレ率（前年比）')]
PANELS_SHORT = {'gdp_obs': 'GDP成長率（年率）', 'ffr': '政策金利（年率）',
                'core_yoy': 'コアインフレ（前年比）', 'head_yoy': '総合インフレ（前年比）'}
C_US, C_JP = '#1f6fb5', '#d62728'


def models(which='mean'):
    th, src = posterior_theta(which, return_source=True)
    jp = Model({k: v for k, v in zip(E.NAMES, th) if not k.startswith('me_')}).solve(verbose=False)
    us = Model().solve(verbose=False)
    return us, jp, src


def half_life(x):
    """|応答| のピーク後、ピークの半分を下回るまでの四半期数（40期以内に戻らなければ NaN）"""
    a = np.abs(x); k = int(np.argmax(a))
    if a[k] < 1e-12:
        return np.nan
    below = np.where(a[k:] < a[k] / 2)[0]
    return float(below[0]) if len(below) else np.nan


def summary(us, jp, T=40):
    rows = []
    for sk, _, lab in SHOCKS:
        for var, vlab in PANELS_SHORT.items():
            r = {'ショック': lab, '変数': vlab}
            for tag, m in [('米国', us), ('日本', jp)]:
                x = m.irf(sk, T)[:, m.idx[var]]
                k = int(np.argmax(np.abs(x)))
                r[f'{tag}_衝撃時'] = x[0]; r[f'{tag}_ピーク'] = x[k]; r[f'{tag}_ピーク期'] = k + 1
                r[f'{tag}_半減期'] = half_life(x)
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'IPAGothic'
    plt.rcParams['axes.unicode_minus'] = False
    which = sys.argv[1] if len(sys.argv) > 1 else 'mean'
    us, jp, src = models(which)
    print('日本のパラメータ:', src)
    for v in ['ffr', 'core_yoy']:
        assert abs(us.ss[v] - jp.ss[v]) < 1e-8, '定常状態が日米で違う'
    fig_dir = os.path.join(HERE, 'figures')
    T = 40; tt = np.arange(1, T + 1)
    groups = [SHOCKS[0:3], SHOCKS[3:6], SHOCKS[6:9], SHOCKS[9:12]]
    for gi, grp in enumerate(groups):
        fig, axes = plt.subplots(len(grp), 4, figsize=(15, 3.1 * len(grp) + 0.6))
        for r, (sk, _, lab) in enumerate(grp):
            iu, ij = us.irf(sk, T), jp.irf(sk, T)
            for c, (var, vlab) in enumerate(PANELS):
                ax = axes[r, c]
                ax.plot(tt, us.series(iu, var), color=C_US, lw=2, label='米国（論文の事後平均）')
                ax.plot(tt, jp.series(ij, var), color=C_JP, lw=2, ls='--', label='日本（日本データで推定）')
                ax.axhline(us.ss[var], color='gray', ls=':', lw=1)
                ax.set_title(vlab if r == 0 else PANELS_SHORT[var], fontsize=9.5)
                ax.grid(alpha=0.3)
                if c == 0:
                    ax.set_ylabel(lab + '\n(%)', fontsize=9)
                if r == len(grp) - 1:
                    ax.set_xlabel('四半期')
        axes[0, 0].legend(fontsize=8, loc='best')
        fig.suptitle('インパルス応答の日米比較（1標準偏差ショック, 点線は定常状態: インフレ2%・政策金利3.15%）', fontsize=12.5)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(fig_dir, f'irf_jp_us_group{gi + 1}.png'), dpi=130)
        plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    for j, (sk, _, lab) in enumerate(SHOCKS):
        ax = axes.flat[j]
        iu, ij = us.irf(sk, T), jp.irf(sk, T)
        ax.plot(tt, iu[:, us.idx['gap']], color=C_US, lw=2, label='米国 産出ギャップ(%)')
        ax.plot(tt, ij[:, jp.idx['gap']], color=C_JP, lw=2, label='日本 産出ギャップ(%)')
        ax.plot(tt, iu[:, us.idx['rstar_ann']], color=C_US, lw=1.3, ls='--', label='米国 r*(年率%pt)')
        ax.plot(tt, ij[:, jp.idx['rstar_ann']], color=C_JP, lw=1.3, ls='--', label='日本 r*(年率%pt)')
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_title(lab, fontsize=10); ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7.5)
    fig.suptitle('産出ギャップと自然利子率 r* の応答の日米比較（定常状態からの乖離）', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(fig_dir, 'irf_jp_us_gap_rstar.png'), dpi=130)
    plt.close(fig)

    # 金融政策ショックを衝撃時の政策金利 +1%pt にそろえる（1標準偏差の大きさの違いを取り除き、効き方だけを比べる）
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for tag, m, col, ls in [('米国', us, C_US, '-'), ('日本', jp, C_JP, '--')]:
        x = m.irf('mp', T); x = x / x[0, m.idx['ffr']]
        for c, (var, vlab) in enumerate(PANELS):
            axes[c].plot(tt, x[:, m.idx[var]], color=col, lw=2, ls=ls, label=tag)
        print(f'{tag}: 政策金利+1%pt あたり 4期後の金利 {x[3, m.idx["ffr"]]:.2f}, GDP成長率(衝撃時) {x[0, m.idx["gdp_obs"]]:.2f}, '
              f'コアインフレ(最大) {x[:, m.idx["core_yoy"]].min():.2f}, 産出ギャップ(最大) {x[:, m.idx["gap"]].min():.2f}')
    for c, (var, vlab) in enumerate(PANELS):
        axes[c].axhline(0, color='gray', lw=0.8); axes[c].grid(alpha=0.3); axes[c].set_title(PANELS_SHORT[var], fontsize=10)
        axes[c].set_xlabel('四半期')
    axes[0].set_ylabel('定常状態からの乖離(%pt)'); axes[0].legend(fontsize=9)
    fig.suptitle('金融政策ショック: 衝撃時の政策金利を +1%pt にそろえた日米比較', fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(fig_dir, 'irf_jp_us_mp_per1pt.png'), dpi=130)
    plt.close(fig)

    S = summary(us, jp, T)
    S.to_csv(os.path.join(HERE, 'data', 'irf_jp_us_summary.csv'), index=False, float_format='%.4f', encoding='utf-8-sig')
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 100)
    print(S.round(3).to_string())
    print('保存完了:', fig_dir)
