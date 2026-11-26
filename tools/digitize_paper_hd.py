"""
原論文の図4・図5（簡略化した歴史分解, 2019Q4〜2026Q2 の積み上げ棒）を数値化する（Issue #2）

使い方:
    python tools/digitize_paper_hd.py 論文.pdf   ->  data/paper_hd_digitized.csv（figure, date, group, value）

必要なもの: poppler-utils（pdfimages）, tesseract-ocr, Pillow

手順:
  1. pdfimages で18〜19ページの4枚（コアPCE, 総合PCE, 産出ギャップ, 自然利子率）を取り出す
  2. y目盛りをOCRして ピクセル→値 の一次式を当てはめる（tools/digitize_paper_irf.py と同じ方法）
  3. 棒ごとに中央付近の列を縦に走査し、凡例と同じ6色（供給・需要・金融・財政・原油・その他）の区間を測る。
     棒の枠線（濃い灰色）は両隣の区間で半分ずつ分け、ゼロの線をまたぐ区間はゼロで切る。列ごとの値の中央値を取る
     （黒い実績線が横切る列の影響を除くため）

読取り誤差: 1ピクセルは約0.007%pt（コアPCE）。枠線の割り振りを含めて ±0.02%pt 程度を目安にする。
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from digitize_paper_irf import _groups, ocr_ticks, fit_ticks     # noqa: E402

FIGS = ['core_yoy', 'head_yoy', 'gap', 'rstar_ann']               # 図4(a), 図4(b), 図5(a), 図5(b)
PAGES = (18, 19)
PALETTE = {'供給': (17, 113, 190), '需要': (221, 84, 0), '金融': (237, 177, 32),
           '財政': (133, 22, 209), '原油': (59, 170, 50), 'その他': (47, 190, 239)}
DATES = pd.date_range('2019-10-01', '2026-04-01', freq='QS')      # 27本


def classify(col):
    """1列分のピクセル (H,3) を 色番号(0..5) / -1=枠線・線 / -2=白 に分類"""
    pal = np.array(list(PALETTE.values()))
    d = np.abs(col[:, None, :].astype(int) - pal[None]).sum(2)
    k = d.argmin(1)
    out = np.where(d.min(1) <= 30, k, -2)
    out[(col.max(1) < 100)] = -1
    return out


def column_values(lab, y0):
    """色区間の長さ（ピクセル, 上が正）を色ごとに合計。枠線は両隣で半分ずつ、ゼロ線で切る"""
    runs = []
    i = 0
    while i < len(lab):
        j = i
        while j + 1 < len(lab) and lab[j + 1] == lab[i]:
            j += 1
        runs.append([lab[i], i, j + 1])
        i = j + 1
    # 境界: 色区間の端は、隣が枠線なら枠線の中点まで延ばす
    seg = []
    for n, (k, a, b) in enumerate(runs):
        if k < 0:
            continue
        top, bot = a, b
        if n > 0 and runs[n - 1][0] == -1:
            top = (runs[n - 1][1] + runs[n - 1][2]) / 2
        if n + 1 < len(runs) and runs[n + 1][0] == -1:
            bot = (runs[n + 1][1] + runs[n + 1][2]) / 2
        seg.append((k, top, bot))
    v = np.zeros(len(PALETTE))
    for k, top, bot in seg:
        up = max(0.0, min(bot, y0) - top)          # ゼロより上の部分（正の寄与）
        dn = max(0.0, bot - max(top, y0))          # ゼロより下の部分（負の寄与）
        v[k] += up - dn
    return v


def digitize_image(path):
    img = Image.open(path).convert('RGB'); a = np.asarray(img)
    dark = (np.abs(a.astype(int) - 33).sum(2) < 12)
    rows = _groups(np.where(dark.sum(1) > 0.5 * a.shape[1])[0]); cols = _groups(np.where(dark.sum(0) > 0.3 * a.shape[0])[0])
    y_top, y_bot = rows[0][1] + 1, rows[-1][0] - 1                # 軸の上辺と下辺（間の線はゼロの線）
    x_l, x_r = cols[0][1] + 1, cols[-1][0] - 1
    A, ticks, err = fit_ticks(ocr_ticks(img, (max(0, x_l - 200), y_top - 40, x_l - 8, y_bot + 40)))
    y0 = (0 - A[1]) / A[0]                                          # 値0のピクセル
    sub = a[y_top:y_bot, x_l:x_r]
    # 棒の中心は横軸の両端と26等分点（xlim が最初と最後の棒の中心。両端の棒は半分だけ見える）
    step = (sub.shape[1] - 1) / (len(DATES) - 1)
    vals, clipped = [], []
    for t in range(len(DATES)):
        c = t * step
        cs = range(max(2, int(c - 0.12 * step)), min(sub.shape[1] - 2, int(c + 0.12 * step)))
        V = np.array([column_values(classify(sub[:, x]), y0 - y_top) for x in cs])
        vals.append(np.median(V, 0) * abs(A[0]))
        edge = classify(sub[:, int(c)])
        clipped.append(bool((edge[:25] >= 0).any() or (edge[-25:] >= 0).any()))  # 軸の端に届いている（切れている可能性）
    return np.array(vals), abs(A[0]), err, clipped


def main(pdf):
    rows = []
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(['pdfimages', '-f', str(PAGES[0]), '-l', str(PAGES[1]), '-png', pdf, os.path.join(td, 'h')], check=True)
        files = sorted(f for f in os.listdir(td) if f.endswith('.png'))
        assert len(files) == len(FIGS), files
        for fig, f in zip(FIGS, files):
            v, vpp, err, clipped = digitize_image(os.path.join(td, f))
            print(f'{fig}: 1ピクセル={vpp:.4f}, 目盛りの当てはめ誤差={err:.4f}, 切れている棒={[str(d.date()) for d, c in zip(DATES, clipped) if c]}')
            for t, d in enumerate(DATES):
                for k, g in enumerate(PALETTE):
                    rows.append(dict(figure=fig, date=d.date(), group=g, value=round(v[t, k], 4),
                                     val_per_px=round(vpp, 5), clipped=clipped[t]))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'data', 'paper_hd_digitized.csv'), index=False)


if __name__ == '__main__':
    main(sys.argv[1])
