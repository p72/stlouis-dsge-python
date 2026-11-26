"""
原論文 付録D（図D-1〜D-12）のIRF図を数値化する（Issue #1）

使い方:
    python tools/digitize_paper_irf.py 論文.pdf
    -> data/paper_irf_digitized.csv（長い形式: shock, var, h, level, ss_line, dev）
       data/paper_irf_digitized_meta.csv（パネルごとの目盛り・読取り精度）

必要なもの: poppler-utils（pdfimages）, tesseract-ocr（目盛りラベルのOCR）, Pillow

手順:
  1. pdfimages で付録Dの12枚（3334x1667ピクセル, 720dpi）を取り出す。順番は SHOCKS と同じ
     （生産性, 労働不効用, 投資効率, 価格マークアップ, 賃金マークアップ, 原油供給, 限界効用, コンビニエンス・イールド,
      政府消費, 移転支出, 金融政策, インフレ目標）
  2. 軸の枠（濃い灰色の線）から4パネルの位置を出す
  3. y目盛りラベルをOCRし、等間隔に並ぶ組を RANSAC で選んで ピクセル→値 の一次式を当てはめる（OCRの小数点落ちを除外）
  4. 青い線のピクセルの縦方向の中心を列ごとに取り、x 軸（1〜40四半期）に補間する
  5. 黒い破線（定常状態）の高さを読む。値域の外などで破線が見えないパネルは、他パネルの中央値を使う

読取り誤差: 線の太さ（約8ピクセル）の中心を取るので、1ピクセル相当の値（meta の val_per_px）の ±2 倍程度を目安にする。
"""
import io
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from stlouis_dsge import SHOCKS                                   # noqa: E402

VARS = ['gdp_obs', 'ffr', 'core_yoy', 'head_yoy']                 # 左上, 右上, 左下, 右下
PAGES = (33, 36)


def _groups(idx):
    out = []
    for i in idx:
        if out and i - out[-1][-1] <= 3:
            out[-1].append(i)
        else:
            out.append([i])
    return [(g[0], g[-1]) for g in out]


def _png(im):
    b = io.BytesIO(); im.save(b, 'PNG'); return b.getvalue()


def ocr_ticks(img, box):
    crop = img.crop(box)
    crop = crop.resize((crop.width * 3 // 2, crop.height * 3 // 2))
    r = subprocess.run(['tesseract', 'stdin', 'stdout', '--psm', '6',
                        '-c', 'tessedit_char_whitelist=0123456789.-', 'tsv'], input=_png(crop), capture_output=True)
    df = pd.read_csv(io.StringIO(r.stdout.decode()), sep='\t', quoting=3)
    df = df[df.conf > 30].dropna(subset=['text'])
    res = []
    for _, w in df.iterrows():
        try:
            v = float(str(w.text))
        except ValueError:
            continue
        res.append((box[1] + (w.top + w.height / 2) * 2 / 3, v))
    return res


def fit_ticks(ticks):
    """等間隔に並ぶ目盛りの組を RANSAC で選び、ピクセル y → 値 の一次式を返す"""
    t = sorted(ticks)
    Y = np.array([q[0] for q in t]); V = np.array([q[1] for q in t])
    best = None
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            if Y[j] - Y[i] < 20 or V[i] == V[j]:
                continue
            A = np.polyfit(Y[[i, j]], V[[i, j]], 1)
            if A[0] >= 0:
                continue
            inl = np.abs(V - np.polyval(A, Y)) < abs(A[0]) * 6
            key = (inl.sum(), np.ptp(Y[inl]))
            if best is None or key > best[0]:
                best = (key, inl)
    if best is None or best[1].sum() < 2:
        raise ValueError(f'目盛りを読めませんでした: {t}')
    inl = best[1]
    A = np.polyfit(Y[inl], V[inl], 1)
    return A, V[inl].tolist(), float(np.max(np.abs(V[inl] - np.polyval(A, Y[inl]))))


def digitize_image(path):
    img = Image.open(path).convert('RGB'); a = np.asarray(img).astype(int)
    dark = a.max(2) < 120
    rows = _groups(np.where(dark.sum(1) > 1500)[0]); cols = _groups(np.where(dark.sum(0) > 800)[0])
    assert len(rows) == 4 and len(cols) == 4, (path, rows, cols)
    out = []
    for rt, rb in [(rows[0], rows[1]), (rows[2], rows[3])]:
        for cl, cr in [(cols[0], cols[1]), (cols[2], cols[3])]:
            y0, y1, x0, x1 = rt[1] + 1, rb[0] - 1, cl[1] + 1, cr[0] - 1
            sub = a[y0:y1, x0:x1]
            blue = (sub[:, :, 2] > 140) & (sub[:, :, 0] < 90) & (sub[:, :, 1] < 160)
            xs, ys = [], []
            for x in range(sub.shape[1]):
                yy = np.where(blue[:, x])[0]
                if len(yy):
                    xs.append(x); ys.append(yy.mean())
            dk = sub.max(2) < 90
            rs = dk.sum(1); rs[:3] = 0; rs[-3:] = 0
            dash = y0 + np.argmax(rs) if rs.max() > 0.15 * sub.shape[1] else np.nan
            A, tick_vals, tick_res = fit_ticks(ocr_ticks(img, (max(0, x0 - 260), y0 - 40, x0 - 8, y1 + 40)))
            h = 1 + np.array(xs) / (x1 + 1 - x0) * 39                       # 横軸 1〜40 が枠の左端〜右端
            lev = np.interp(np.arange(1, 41), h, np.polyval(A, np.array(ys) + y0))
            out.append(dict(level=lev, ss_line=np.polyval(A, dash) if not np.isnan(dash) else np.nan,
                            val_per_px=abs(A[0]), ticks=tick_vals, tick_fit_err=tick_res,
                            ymin=np.polyval(A, y1), ymax=np.polyval(A, y0)))
    return out


def main(pdf):
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(['pdfimages', '-f', str(PAGES[0]), '-l', str(PAGES[1]), '-png', pdf, os.path.join(td, 'd')], check=True)
        files = sorted(f for f in os.listdir(td) if f.endswith('.png'))
        assert len(files) == len(SHOCKS), files
        panels = {s[0]: digitize_image(os.path.join(td, f)) for s, f in zip(SHOCKS, files)}
    # 破線が見えないパネルは、同じ変数の他パネルで読んだ定常値の中央値を使う
    ss_med = {v: np.nanmedian([panels[s][k]['ss_line'] for s in panels]) for k, v in enumerate(VARS)}
    rows, meta = [], []
    for s, P in panels.items():
        for k, v in enumerate(VARS):
            p = P[k]
            ss = p['ss_line'] if not np.isnan(p['ss_line']) else ss_med[v]
            for h in range(40):
                rows.append(dict(shock=s, var=v, h=h + 1, level=round(p['level'][h], 5), ss_line=round(ss, 5),
                                 dev=round(p['level'][h] - ss, 5)))
            meta.append(dict(shock=s, var=v, ss_line_read=not np.isnan(p['ss_line']), ss_line=round(ss, 5),
                             val_per_px=round(p['val_per_px'], 6), tick_fit_err=round(p['tick_fit_err'], 6),
                             ymin=round(p['ymin'], 4), ymax=round(p['ymax'], 4), ticks=' '.join(map(str, p['ticks']))))
    os.makedirs(os.path.join(ROOT, 'data'), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'data', 'paper_irf_digitized.csv'), index=False)
    pd.DataFrame(meta).to_csv(os.path.join(ROOT, 'data', 'paper_irf_digitized_meta.csv'), index=False)
    print('定常状態の破線（中央値）:', {v: round(x, 4) for v, x in ss_med.items()})


if __name__ == '__main__':
    main(sys.argv[1])
