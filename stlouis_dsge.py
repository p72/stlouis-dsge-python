"""
セントルイス連銀 DSGE モデル (Faria-e-Castro, 2026, FRB St. Louis Review Q4 2026) の Python 移植

- 付録Aの均衡条件 (47)-(99) をトレンド除去後の変数でそのまま記述
- 粘着価格経済 + 伸縮価格経済（産出ギャップ・自然利子率 r* 用）を同時に解く
- 定常状態は解析的に計算（労働 n=1, 実質原油価格 pO=1 に正規化）
- 線形化は複素ステップ微分（機械精度のヤコビアン）
- 解法は QZ 分解 (scipy.linalg.ordqz) による Blanchard-Kahn / Klein 型の一次近似解
    A E[y_{t+1}] + B y_t + C y_{t-1} + D eps_t = 0  ->  y_t = P y_{t-1} + Q eps_t
- パラメータは表1（カリブレーション）と表C-1, C-2（事後平均）

使い方:
    python stlouis_dsge.py          # 解いてインパルス応答を描画・保存
    from stlouis_dsge import Model; m = Model(); m.solve(); m.irf('mp')
"""
import numpy as np
from scipy.linalg import ordqz, solve

# ---------------------------------------------------------------------------
# 1. パラメータ
# ---------------------------------------------------------------------------
def default_params():
    p = dict(
        # 表1 カリブレーション
        delta=0.025, mup_bar=1.2, muw_bar=1.2, Pi_bar=1.005,
        tauk_bar=0.218, taul_bar=0.186, rho_taul=0.9, rho_tauk=0.9,
        g_gdp=0.202, t_gdp=0.137, bg_gdp=4 * 0.639,
        omegaC=0.02, omegaY=0.03,
        # 表C-1 構造パラメータ（事後平均）
        GZ_bar=1.0049, GN_bar=0.9998, disc=0.0400,   # 100(1/beta-1)
        phi=0.9960, etap=77.8666, etaw=105.8931, psii=1.9496,
        iotap=0.2614, iotaw=0.7590, sigmaa=0.4727, alpha=0.1946,
        psiw=0.2794, lam=0.6148, vartheta_bar=1.0022,
        nuC=0.4953, nuY=0.1371, phiPi=2.8470, phiY=0.7554, rhor=0.8138,
        lbw_gdp=1.1993, phitaul=0.0681, phitauk=0.0744,
        # 表C-2 ショック過程（事後平均）
        rho_GZ=0.0676, rho_mup=0.9887, rho_muw=0.5429, rho_zeta=0.3438,
        rho_chi=0.9376, rho_vt=0.9611, rho_g=0.9642, rho_t=0.9884,
        rho_pistar=0.9904, rho_mp=0.0686, rho_GN=0.2804, rho_oS=0.9596,
        sig_GZ=0.0078, sig_mup=0.0285, sig_muw=0.2177, sig_zeta=0.0698,
        sig_chi=0.0222, sig_vt=0.0010, sig_g=0.0095, sig_t=0.0261,
        sig_pistar=0.0004, sig_mp=0.0024, sig_GN=0.0028, sig_oS=0.0167,
        eta_mup=0.7567, eta_muw=0.5043,
        # 表1 パネルB: COVID特殊ショック
        sig_chi_cov=0.045, sig_t_cov=0.220, sig_N_cov=0.015,
        # 正規化
        n_ss=1.0, pO_ss=1.0,
    )
    return p


# 構造ショック: (内部名, 外生変数名, 日本語ラベル)
SHOCKS = [
    ('GZ', 'GZ', '生産性（TFP成長率）ショック'),
    ('GN', 'GN', '労働不効用ショック'),
    ('zeta', 'zeta', '投資の限界効率ショック'),
    ('mup', 'mup', '価格マークアップショック'),
    ('muw', 'muw', '賃金マークアップショック'),
    ('oS', 'oS', '原油供給ショック'),
    ('chi', 'chi', '限界効用（需要）ショック'),
    ('vt', 'vt', 'コンビニエンス・イールドショック'),
    ('g', 'g', '政府消費ショック'),
    ('t', 't', '移転支出ショック'),
    ('mp', 'mp', '金融政策ショック'),
    ('pistar', 'pistar', 'インフレ目標ショック'),
]
# COVID 特殊ショック（2020〜21年のみ有効, 3.3節）
COVID_SHOCKS = [
    ('chi_cov', 'chi', 'COVID需要ショック'),
    ('t_cov', 't', 'COVID移転支出ショック'),
    ('N_cov', 'GN', 'COVID労働供給ショック'),
]

# 内生変数（粘着価格ブロック）。伸縮価格ブロックは末尾 _f を付け、賃金/価格インデックス関連を除く
CORE_VARS = ['cw', 'cNw', 'cOw', 'bw', 'cs', 'cNs', 'cOs', 'Qk', 'i', 'u', 'Rk', 'k', 'ku',
             'n', 'w', 'mrs', 'Pi', 'R', 'y', 'v', 'oY', 'mc', 'pV', 'pO', 'cN', 'cO',
             'c', 'gdp', 'bg', 'taul', 'tauk', 'd', 'rr', 'Pih']
STICKY_ONLY = ['Piw', 'Pind', 'Pwind']
EXO_VARS = ['GZ', 'GN', 'zeta', 'mup', 'muw', 'oS', 'chi', 'vt', 'g', 't', 'mp', 'pistar',
            'emup_lag', 'emuw_lag', 'chit', 'tt', 'GNt', 'ecovN_lag']
OBS_VARS = ['Pi_l1', 'Pi_l2', 'Pi_l3', 'Pih_l1', 'Pih_l2', 'Pih_l3',
            'gdp_obs', 'ffr', 'core_yoy', 'head_yoy', 'gap', 'rstar', 'rstar_ann',
            'cons_obs', 'inv_obs', 'gov_obs', 'tr_obs', 'wage_obs', 'hours_obs',
            'gas_obs', 'oil_obs', 'Raaa']


# ---------------------------------------------------------------------------
# 2. 定常状態
# ---------------------------------------------------------------------------
def steady_state(p):
    s = {}
    beta = 1.0 / (1.0 + p['disc'] / 100.0)
    p['beta'] = beta
    Gam = p['GZ_bar'] * p['GN_bar']
    p['Gam_bar'] = Gam
    Pi = p['Pi_bar']
    R = Gam * Pi / (beta * p['vartheta_bar'])
    p['rbar'] = R / Pi
    Rk = (Gam / beta - 1 + p['delta']) / (1 - p['tauk_bar'])
    p['kappaa'] = (1 - p['tauk_bar']) * Rk
    pO = p['pO_ss']
    mc = 1.0 / p['mup_bar']
    nuY, wY = p['nuY'], p['omegaY']
    pV = ((mc ** (1 - nuY) - wY * pO ** (1 - nuY)) / (1 - wY)) ** (1 / (1 - nuY))
    a = p['alpha']
    w = (pV / (Rk / a) ** a) ** (1 / (1 - a)) * (1 - a)
    n = p['n_ss']
    ku = a / (1 - a) * w / Rk * n
    v = ku ** a * n ** (1 - a)
    y = v / ((1 - wY) * (pV / mc) ** (-nuY))
    oY = wY * (pO / mc) ** (-nuY) * y
    k = Gam * ku
    i = k * (1 - (1 - p['delta']) / Gam)
    d = y - w * n - Rk * ku - pO * oY
    kapC = p['omegaC'] / (1 - p['omegaC']) * pO ** (-p['nuC'])
    # gdp = y + pO*cO, cO = kapC*cN, cN = y - g - i, g = g_gdp*gdp
    gdp = (y * (1 + pO * kapC) - pO * kapC * i) / (1 + pO * kapC * p['g_gdp'])
    g = p['g_gdp'] * gdp
    t = p['t_gdp'] * gdp
    bg = p['bg_gdp'] * gdp
    cN = y - g - i
    cO = kapC * cN
    oS = cO + oY
    taul, tauk = p['taul_bar'], p['tauk_bar']
    tauc = (g + t + bg / (Pi * Gam) - bg / R - taul * w * n - tauk * (Rk * ku + d)) / (cN + pO * cO)
    p['tauc'] = tauc
    lam = p['lam']
    bw = p['lbw_gdp'] * gdp / lam
    p['bw_bar'] = bw
    rhs = (1 - taul) * w * n / lam + bw / (Pi * Gam) + t - bw / R
    cNw = rhs / ((1 + tauc) * (1 + pO * kapC))
    cOw = kapC * cNw
    cNs = (cN - lam * cNw) / (1 - lam)
    cOs = kapC * cNs
    nuC, wC = p['nuC'], p['omegaC']
    ces = lambda cn, co: ((1 - wC) ** (1 / nuC) * cn ** ((nuC - 1) / nuC)
                          + wC ** (1 / nuC) * co ** ((nuC - 1) / nuC)) ** (nuC / (nuC - 1))
    cw, cs = ces(cNw, cOw), ces(cNs, cOs)
    mrs = w / p['muw_bar']
    # MRS式から xi_bar を逆算
    p['xibar'] = mrs * (1 - taul) * (1 - wC) ** (1 / nuC) / (
        (1 + tauc) * (n / lam) ** p['phi'] * cw ** ((nuC - 1) / nuC) * cNw ** (1 / nuC))
    c = cN + pO * cO
    s.update(cw=cw, cNw=cNw, cOw=cOw, bw=bw, cs=cs, cNs=cNs, cOs=cOs, Qk=1.0, i=i, u=1.0,
             Rk=Rk, k=k, ku=ku, n=n, w=w, mrs=mrs, Pi=Pi, R=R, y=y, v=v, oY=oY, mc=mc,
             pV=pV, pO=pO, cN=cN, cO=cO, c=c, gdp=gdp, bg=bg, taul=taul, tauk=tauk, d=d,
             rr=R / Pi, Pih=Pi)
    full = dict(s)
    full.update(Piw=p['GZ_bar'] * Pi, Pind=Pi, Pwind=p['GZ_bar'] * Pi)
    for key, val in s.items():
        full[key + '_f'] = val
    full.update(GZ=p['GZ_bar'], GN=p['GN_bar'], zeta=1.0, mup=p['mup_bar'], muw=p['muw_bar'],
                oS=oS, chi=1.0, vt=p['vartheta_bar'], g=g, t=t, mp=1.0, pistar=Pi,
                emup_lag=0.0, emuw_lag=0.0, chit=1.0, tt=t, GNt=p['GN_bar'], ecovN_lag=0.0)
    p['oS_bar'], p['g_bar'], p['t_bar'] = oS, g, t
    full.update(Pi_l1=Pi, Pi_l2=Pi, Pi_l3=Pi, Pih_l1=Pi, Pih_l2=Pi, Pih_l3=Pi,
                gdp_obs=400 * np.log(Gam), ffr=400 * np.log(R), core_yoy=400 * np.log(Pi),
                head_yoy=400 * np.log(Pi), gap=0.0, rstar=R / Pi, rstar_ann=400 * np.log(R / Pi),
                cons_obs=400 * np.log(Gam), inv_obs=400 * np.log(Gam), gov_obs=400 * np.log(Gam),
                tr_obs=400 * np.log(Gam), wage_obs=400 * np.log(p['GZ_bar']),
                hours_obs=400 * np.log(p['GN_bar']), gas_obs=400 * np.log(Pi), oil_obs=np.log(pO),
                Raaa=R * p['vartheta_bar'])
    # 資本家の予算制約（ワルラス法則でモデルからは落としている）のチェック
    bs = (bg - lam * bw) / (1 - lam)
    lhs = (1 - lam) * ((1 + tauc) * (cNs + pO * cOs) + bs / R - bs / (Pi * Gam) - t) + i
    rhs = (1 - tauk) * (Rk * ku + d) + pO * oS
    p['walras_check'] = lhs - rhs
    return full


# ---------------------------------------------------------------------------
# 3. 均衡条件（付録A）
# ---------------------------------------------------------------------------
def block_equations(L, C, F, E, p, sfx, flex):
    """L, C, F: ラグ・当期・リードの変数アクセサ。sfx='' (粘着) or '_f' (伸縮)."""
    x = lambda nm: C(nm + sfx)
    xl = lambda nm: L(nm + sfx)
    xf = lambda nm: F(nm + sfx)
    nuC, wC, nuY, wY, a = p['nuC'], p['omegaC'], p['nuY'], p['omegaY'], p['alpha']
    lam, tauc = p['lam'], p['tauc']
    Gam, Gamf = C('GZ') * C('GNt'), F('GZ') * F('GNt')
    Pistar = C('pistar')
    vt, chi, chif, zeta, zetaf = C('vt'), C('chit'), F('chit'), C('zeta'), F('zeta')
    mup = p['mup_bar'] if flex else C('mup')
    muw = p['muw_bar'] if flex else C('muw')

    def sdf(cc, cn):
        return (p['beta'] / Gamf * (chif / chi) * (xf(cc) / x(cc)) ** (-(nuC - 1) / nuC)
                * (xf(cn) / x(cn)) ** (-1 / nuC))
    mw, ms = sdf('cw', 'cNw'), sdf('cs', 'cNs')
    ces = lambda cn, co: ((1 - wC) ** (1 / nuC) * cn ** ((nuC - 1) / nuC)
                          + wC ** (1 / nuC) * co ** ((nuC - 1) / nuC)) ** (nuC / (nuC - 1))
    S = lambda z: p['psii'] / 2 * (z - p['Gam_bar']) ** 2
    Sp = lambda z: p['psii'] * (z - p['Gam_bar'])
    A = lambda u: p['kappaa'] * (u - 1) + 0.5 * p['sigmaa'] * (u - 1) ** 2
    Ap = lambda u: p['kappaa'] + p['sigmaa'] * (u - 1)
    xinv, xinvf = Gam * x('i') / xl('i'), Gamf * xf('i') / x('i')
    Pi, Pif, R = x('Pi'), xf('Pi'), x('R')
    if flex:
        price_cost = 0.0
    else:
        price_cost = 0.5 * p['etap'] * (Pi / x('Pind') - 1) ** 2
    eq = []
    # --- 労働者 ---
    eq.append(R * vt * mw / Pif / (1 + p['psiw'] * (x('bw') - p['bw_bar'])) - 1)            # (48)
    eq.append((1 + tauc) * (x('cNw') + x('pO') * x('cOw')) + x('bw') / R
              - (1 - x('taul')) * x('w') * x('n') / lam - xl('bw') / (Pi * Gam) - C('tt'))      # (49)
    eq.append(x('cOw') - x('pO') ** (-nuC) * wC / (1 - wC) * x('cNw'))                     # (50)
    eq.append(x('cw') - ces(x('cNw'), x('cOw')))                                           # (51)
    # --- 資本家 ---
    eq.append(R * vt * ms / Pif - 1)                                                        # (53)
    eq.append((1 - x('tauk')) * x('Rk') - Ap(x('u')))                                      # (54)
    eq.append(x('Qk') - ms * ((1 - xf('tauk')) * xf('Rk') * xf('u')
                              + (1 - p['delta']) * xf('Qk') - A(xf('u'))))                  # (55)
    eq.append(1 - x('Qk') * zeta * (1 - S(xinv) - Sp(xinv) * xinv)
              - ms * xf('Qk') * zetaf * Sp(xinvf) * xinvf ** 2)                             # (56)
    eq.append(x('cOs') - x('pO') ** (-nuC) * wC / (1 - wC) * x('cNs'))                     # (57)
    eq.append(x('cs') - ces(x('cNs'), x('cOs')))                                           # (58)
    # --- 労働市場 ---
    eq.append(x('mrs') - (1 + tauc) * p['xibar'] * (x('n') / lam) ** p['phi']
              * x('cw') ** ((nuC - 1) / nuC) * x('cNw') ** (1 / nuC)
              / ((1 - x('taul')) * (1 - wC) ** (1 / nuC)))                                  # (60)
    if flex:
        eq.append(x('w') - muw * x('mrs'))                                                  # 伸縮賃金
    else:
        Piw, Pwind = x('Piw'), x('Pwind')
        eq.append(x('w') + muw / (muw - 1) * (x('mrs') - x('w'))
                  - p['etaw'] * Piw / Pwind * (Piw / Pwind - 1)
                  + p['beta'] * p['etaw'] * F('GNt') * xf('n') / x('n')
                  * xf('Piw') / xf('Pwind') * (xf('Piw') / xf('Pwind') - 1))                # (59)
        eq.append(Pwind - p['GZ_bar'] * xl('Pi') ** p['iotaw'] * Pistar ** (1 - p['iotaw']))  # (61)
        eq.append(Piw - x('w') / xl('w') * C('GZ') * Pi)                                     # (62)
    # --- 資本 ---
    eq.append(x('k') - (1 - p['delta']) * xl('k') / Gam - zeta * (1 - S(xinv)) * x('i'))    # (63)
    eq.append(x('ku') - x('u') * xl('k') / Gam)                                             # (64)
    # --- 企業 ---
    if flex:
        eq.append(mup * x('mc') - 1)
    else:
        Pind, Pindf = x('Pind'), xf('Pind')
        eq.append(Pi / Pind * (Pi / Pind - 1)
                  - ms * Gamf * xf('y') / x('y') * Pif / Pindf * (Pif / Pindf - 1)
                  - (mup * x('mc') - 1) / (p['etap'] * (mup - 1)))                          # (65)
        eq.append(Pind - xl('Pi') ** p['iotap'] * Pistar ** (1 - p['iotap']))                # (68)
    eq.append(x('mc') - (wY * x('pO') ** (1 - nuY) + (1 - wY) * x('pV') ** (1 - nuY)) ** (1 / (1 - nuY)))  # (66)
    eq.append(x('pV') - (x('Rk') / a) ** a * (x('w') / (1 - a)) ** (1 - a))                # (67)
    eq.append(x('y') - ((1 - wY) ** (1 / nuY) * x('v') ** ((nuY - 1) / nuY)
                        + wY ** (1 / nuY) * x('oY') ** ((nuY - 1) / nuY)) ** (nuY / (nuY - 1)))  # (69)
    eq.append(x('v') - x('ku') ** a * x('n') ** (1 - a))                                  # (70)
    eq.append(x('ku') - a / (1 - a) * x('w') / x('Rk') * x('n'))                           # (71)
    eq.append(x('oY') - (x('pV') / x('pO')) ** nuY * wY / (1 - wY) * x('v'))               # (72)
    # --- 資源制約・原油・GDP ---
    eq.append(x('cN') + C('g') + x('i') + A(x('u')) * xl('k') / Gam - x('y') * (1 - price_cost))  # (73)
    eq.append(x('cN') - lam * x('cNw') - (1 - lam) * x('cNs'))                             # (74)
    eq.append(x('cO') - lam * x('cOw') - (1 - lam) * x('cOs'))                             # (75)
    eq.append(x('c') - x('cN') - x('pO') * x('cO'))                                        # (76)
    eq.append(x('cO') + x('oY') - C('oS'))                                                  # (77)
    eq.append(x('gdp') - x('y') - x('pO') * (C('oS') - x('oY')))                           # (78)
    # --- 政府 ---
    eq.append(C('g') + C('tt') + xl('bg') / (Pi * Gam)
              - tauc * (x('cN') + x('pO') * x('cO')) - x('taul') * x('w') * x('n')
              - x('tauk') * (x('Rk') * x('ku') + x('d')) - x('bg') / R)                     # (79)
    debt_ratio = (xl('bg') / xl('gdp')) / p['bg_gdp']
    eq.append(x('taul') - xl('taul') ** p['rho_taul']
              * (p['taul_bar'] * debt_ratio ** p['phitaul']) ** (1 - p['rho_taul']))       # (80)
    eq.append(x('tauk') - xl('tauk') ** p['rho_tauk']
              * (p['tauk_bar'] * debt_ratio ** p['phitauk']) ** (1 - p['rho_tauk']))       # (81)
    if flex:
        eq.append(Pi - Pistar)                                                             # Pi^f = Pi*
    else:
        eq.append(R - xl('R') ** p['rhor'] * (p['rbar'] * Pistar * (Pi / Pistar) ** p['phiPi']
                  * (x('gdp') / xl('gdp')) ** p['phiY']) ** (1 - p['rhor']) * C('mp'))       # (82)
    # --- その他 ---
    eq.append(x('d') - (x('y') - x('w') * x('n') - x('Rk') * x('ku') - x('pO') * x('oY')
                        - x('y') * price_cost))                                             # (83)
    eq.append(x('rr') - R / Pif)                                                            # (84)
    ag = lambda po: 1 - wC + wC * po ** (1 - nuC)
    eq.append(x('Pih') - Pi * (ag(x('pO')) / ag(xl('pO'))) ** (1 / (1 - nuC)))            # (85)
    return eq, ms


def exo_equations(L, C, E, p):
    lg = np.log
    eq = []
    ar = lambda nm, bar, rho, sig, ek: lg(C(nm)) - (1 - rho) * lg(bar) - rho * lg(L(nm)) - sig * E(ek)
    eq.append(ar('GZ', p['GZ_bar'], p['rho_GZ'], p['sig_GZ'], 'GZ'))                       # (87)
    eq.append(ar('GN', p['GN_bar'], p['rho_GN'], p['sig_GN'], 'GN'))                       # (88)
    eq.append(ar('zeta', 1.0, p['rho_zeta'], p['sig_zeta'], 'zeta'))                       # (89)
    eq.append(ar('mup', p['mup_bar'], p['rho_mup'], p['sig_mup'], 'mup')
              + p['eta_mup'] * p['sig_mup'] * L('emup_lag'))                               # (90) ARMA(1,1)
    eq.append(ar('muw', p['muw_bar'], p['rho_muw'], p['sig_muw'], 'muw')
              + p['eta_muw'] * p['sig_muw'] * L('emuw_lag'))                               # (91) ARMA(1,1)
    eq.append(ar('oS', p['oS_bar'], p['rho_oS'], p['sig_oS'], 'oS'))                       # (92)
    eq.append(ar('chi', 1.0, p['rho_chi'], p['sig_chi'], 'chi'))                           # (93)
    eq.append(ar('vt', p['vartheta_bar'], p['rho_vt'], p['sig_vt'], 'vt'))                 # (94)
    eq.append(ar('g', p['g_bar'], p['rho_g'], p['sig_g'], 'g'))                            # (95)
    eq.append(ar('t', p['t_bar'], p['rho_t'], p['sig_t'], 't'))                            # (96)
    eq.append(ar('mp', 1.0, p['rho_mp'], p['sig_mp'], 'mp'))                               # (97)
    eq.append(ar('pistar', p['Pi_bar'], p['rho_pistar'], p['sig_pistar'], 'pistar'))       # (98)
    eq.append(C('emup_lag') - E('mup'))
    eq.append(C('emuw_lag') - E('muw'))
    # COVID 特殊ショック (44)-(46)
    eq.append(lg(C('chit')) - lg(C('chi')) - p['sig_chi_cov'] * E('chi_cov'))
    eq.append(lg(C('tt')) - lg(C('t')) - p['sig_t_cov'] * E('t_cov'))
    eq.append(lg(C('GNt')) - lg(C('GN')) - p['sig_N_cov'] * E('N_cov')
              + p['sig_N_cov'] * L('ecovN_lag'))
    eq.append(C('ecovN_lag') - E('N_cov'))
    return eq


def obs_equations(L, C, F, p, ms_f, ms, Pif):
    lg = np.log
    eq = []
    eq.append(C('Pi_l1') - L('Pi')); eq.append(C('Pi_l2') - L('Pi_l1')); eq.append(C('Pi_l3') - L('Pi_l2'))
    eq.append(C('Pih_l1') - L('Pih')); eq.append(C('Pih_l2') - L('Pih_l1')); eq.append(C('Pih_l3') - L('Pih_l2'))
    eq.append(C('gdp_obs') - 400 * (lg(C('gdp')) - lg(L('gdp')) + lg(C('GZ')) + lg(C('GNt'))))  # (100)
    eq.append(C('ffr') - 400 * lg(C('R')))                                                      # (107)
    eq.append(C('core_yoy') - 100 * lg(C('Pi') * C('Pi_l1') * C('Pi_l2') * C('Pi_l3')))
    eq.append(C('head_yoy') - 100 * lg(C('Pih') * C('Pih_l1') * C('Pih_l2') * C('Pih_l3')))
    eq.append(C('gap') - 100 * lg(C('gdp') / C('gdp_f')))                                       # 産出ギャップ(%)
    eq.append(C('rstar') * C('vt') * ms_f - 1)                                                  # (99)
    eq.append(C('rstar_ann') - 400 * lg(C('rstar')))
    growth = lambda v: 400 * (lg(C(v)) - lg(L(v)) + lg(C('GZ')) + lg(C('GNt')))
    eq.append(C('cons_obs') - growth('c'))                                                    # (101)
    eq.append(C('inv_obs') - growth('i'))                                                     # (102)
    eq.append(C('gov_obs') - growth('g'))                                                     # (103)
    eq.append(C('tr_obs') - growth('tt'))                                                     # (104)
    eq.append(C('wage_obs') - 400 * (lg(C('w')) - lg(L('w')) + lg(C('GZ'))))                 # (105)
    eq.append(C('hours_obs') - 400 * (lg(C('n')) - lg(L('n')) + lg(C('GNt'))))                # (106)
    eq.append(C('gas_obs') - 400 * (lg(C('Pi')) + lg(C('pO')) - lg(L('pO'))))                 # (111)
    eq.append(C('oil_obs') - lg(C('pO')))                                                     # (110)
    eq.append(C('Raaa') * ms / Pif - 1)                                                       # (86)
    return eq


# ---------------------------------------------------------------------------
# 4. モデルクラス：線形化と QZ 解法
# ---------------------------------------------------------------------------
class Model:
    def __init__(self, params=None):
        self.p = default_params()
        if params:
            self.p.update(params)
        self.vars = (CORE_VARS + STICKY_ONLY + [v + '_f' for v in CORE_VARS] + EXO_VARS + OBS_VARS)
        self.idx = {v: j for j, v in enumerate(self.vars)}
        self.shocks = [s[0] for s in SHOCKS + COVID_SHOCKS]
        self.sidx = {s: j for j, s in enumerate(self.shocks)}
        self.n, self.ne = len(self.vars), len(self.shocks)
        self.ss = steady_state(self.p)
        self.ybar = np.array([self.ss[v] for v in self.vars])

    def residuals(self, yl, yc, yf, e):
        p, idx, sidx = self.p, self.idx, self.sidx
        L = lambda nm: yl[idx[nm]]
        C = lambda nm: yc[idx[nm]]
        F = lambda nm: yf[idx[nm]]
        E = lambda nm: e[sidx[nm]]
        eq1, ms = block_equations(L, C, F, E, p, '', flex=False)
        eq2, ms_f = block_equations(L, C, F, E, p, '_f', flex=True)
        eq3 = exo_equations(L, C, E, p)
        eq4 = obs_equations(L, C, F, p, ms_f, ms, F('Pi'))
        return np.array(eq1 + eq2 + eq3 + eq4)

    def check_steady_state(self):
        y = self.ybar.astype(complex)
        r = self.residuals(y, y, y, np.zeros(self.ne, complex))
        return np.max(np.abs(r)), r

    def linearize(self, h=1e-30):
        n, ne = self.n, self.ne
        z0 = np.concatenate([self.ybar, self.ybar, self.ybar, np.zeros(ne)]).astype(complex)
        m = len(z0)
        J = np.zeros((n, m))
        for j in range(m):
            z = z0.copy()
            z[j] += 1j * h
            r = self.residuals(z[:n], z[n:2 * n], z[2 * n:3 * n], z[3 * n:])
            J[:, j] = r.imag / h          # 複素ステップ微分
        self.Cm, self.Bm, self.Am, self.Dm = J[:, :n], J[:, n:2 * n], J[:, 2 * n:3 * n], J[:, 3 * n:]
        return self

    def solve(self, verbose=True):
        err, _ = self.check_steady_state()
        assert err < 1e-9, f'定常状態の残差が大きい: {err}'
        self.linearize()
        n, A, B, C, D = self.n, self.Am, self.Bm, self.Cm, self.Dm
        I, Z0 = np.eye(n), np.zeros((n, n))
        # s_t = [y_{t-1}; y_t]:  E s_{t+1} = F s_t
        Emat = np.block([[I, Z0], [Z0, A]])
        Fmat = np.block([[Z0, I], [-C, -B]])
        AA, BB, alpha, beta, Q, Z = ordqz(Fmat, Emat, sort='iuc', output='complex')
        with np.errstate(divide='ignore', invalid='ignore'):
            lam = np.abs(alpha / beta)
        nstable = int(np.sum(lam < 1 - 1e-9))
        self.eigs = lam
        if verbose:
            print(f'変数の数 n = {n}, 安定根の数 = {nstable}（BK条件: n と一致すれば一意解）')
        if nstable != n:
            raise RuntimeError(f'Blanchard-Kahn 条件不成立: 安定根 {nstable} != {n}')
        Z11, Z21 = Z[:n, :n], Z[n:, :n]
        P = np.real(Z21 @ np.linalg.inv(Z11))
        Qs = -solve(A @ P + B, D)
        self.P, self.Q = P, Qs
        res = np.max(np.abs(A @ P @ P + B @ P + C))
        if verbose:
            print(f'行列二次方程式の残差 max|AP^2+BP+C| = {res:.2e}')
        return self

    def irf(self, shock, T=40, size=1.0):
        """1標準偏差ショックへの応答（水準からの乖離, 元の単位）を返す (T x n)"""
        out = np.zeros((T, self.n))
        e = np.zeros(self.ne); e[self.sidx[shock]] = size
        x = self.Q @ e
        for t in range(T):
            out[t] = x
            x = self.P @ x
        return out

    def series(self, irf, var):
        return self.ss[var] + irf[:, self.idx[var]]

    def simulate(self, T=200, seed=0):
        rng = np.random.default_rng(seed)
        x = np.zeros(self.n); out = np.zeros((T, self.n))
        for t in range(T):
            x = self.P @ x + self.Q @ rng.standard_normal(self.ne)
            out[t] = x
        return self.ybar + out


# ---------------------------------------------------------------------------
# 5. 実行・作図
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'IPAGothic'
    plt.rcParams['axes.unicode_minus'] = False

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(out, exist_ok=True)
    m = Model().solve()
    print('ワルラス法則チェック（資本家予算制約の残差）:', m.p['walras_check'])
    print('消費税率 tau_c (内生):', round(m.p['tauc'], 4))

    panels = [('gdp_obs', 'GDP成長率（年率）'), ('ffr', 'FF金利（年率）'),
              ('core_yoy', 'コアインフレ率（前年比）'), ('head_yoy', '総合インフレ率（前年比）')]
    groups = [SHOCKS[0:3], SHOCKS[3:6], SHOCKS[6:9], SHOCKS[9:12]]
    T = 40
    tt = np.arange(1, T + 1)
    for gi, grp in enumerate(groups):
        fig, axes = plt.subplots(len(grp), 4, figsize=(15, 3.1 * len(grp)))
        for r, (sk, _, lab) in enumerate(grp):
            irf = m.irf(sk, T)
            for cidx, (var, vlab) in enumerate(panels):
                ax = axes[r, cidx]
                ax.plot(tt, m.series(irf, var), color='#1f6fb5', lw=2)
                ax.axhline(m.ss[var], color='gray', ls=':', lw=1)
                ax.set_title(vlab, fontsize=10)
                ax.grid(alpha=0.3)
                if cidx == 0:
                    ax.set_ylabel(lab + '\n(%pt)', fontsize=9)
                if r == len(grp) - 1:
                    ax.set_xlabel('四半期')
        fig.suptitle('インパルス応答（1標準偏差ショック, 事後平均パラメータ）', fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'irf_group{gi + 1}.png'), dpi=130)
        plt.close(fig)

    # 産出ギャップと自然利子率
    fig, axes = plt.subplots(3, 4, figsize=(15, 8.5))
    for j, (sk, _, lab) in enumerate(SHOCKS):
        ax = axes.flat[j]
        irf = m.irf(sk, T)
        ax.plot(tt, irf[:, m.idx['gap']], label='産出ギャップ(%)', lw=2)
        ax.plot(tt, irf[:, m.idx['rstar_ann']], label='自然利子率 r*(年率%pt)', lw=2, ls='--')
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_title(lab, fontsize=10); ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('産出ギャップと自然利子率 r* の応答（定常状態からの乖離）', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'irf_gap_rstar.png'), dpi=130)
    plt.close(fig)
    print('保存完了:', out)
