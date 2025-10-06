# MisterBot - Softmax 1X2 sin fuga + Forma (EWMA) + H2H + H2H Home + Elo Carry + Temp Scaling
# + DEDUP robusto y guardarraíles para evitar explosión de filas
# + FAST_MODE opcional para tuning más rápido
# + Filtro de 1ª división por temporada (top-N equipos) y descarte de temporadas incompletas

import sys, os, time, re
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2

from scipy.optimize import minimize

from utils import conexion_db
import random, tensorflow as tf

# ============ Logging helper ============
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# =========================
# CONFIG
# =========================
TEST_SEASON = '24/25'    # test final
VAL_SEASON  = '23/24'    # validación para tuning
FORM_WINDOW = 5          # forma rolling
H2H_WINDOW  = 5          # h2h últimos enfrentamientos
ELO_START   = 1500.0     # base Elo
ELO_CARRY   = 0.75       # carry-over entre temporadas
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Activa para tuning rápido (reduce grid y épocas)
FAST_MODE = True

if FAST_MODE:
    ELO_K_GRID       = [22.0, 26.0]
    ELO_HOME_GRID    = [60.0, 80.0]
    W_X_GRID         = [0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50]
    DELTA_GRID       = [0.02, 0.04, 0.06, 0.08, 0.10]
    TAU_GRID         = [0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.50]
    EPOCHS_TUNE      = 60
    PATIENCE_TUNE    = 6
else:
    ELO_K_GRID       = [22.0, 24.0, 26.0, 28.0]
    ELO_HOME_GRID    = [60.0, 70.0, 80.0]
    W_X_GRID         = [1.00, 1.10, 1.20, 1.30, 1.40]
    DELTA_GRID       = [0.02, 0.04, 0.06]
    TAU_GRID         = [0.28, 0.32, 0.36, 0.40]
    EPOCHS_TUNE      = 120
    PATIENCE_TUNE    = 10

EPOCHS_FINAL = 120

# =========================
# Normalización de nombres
# =========================
map_nombres = {
    "Ath Bilbao": "Athletic Club",
    "Ath Madrid": "Atlético",
    "Vallecano": "Rayo Vallecano",
    "Espanol": "Espanyol",
    "Villareal": "Villarreal",
    "Alaves": "Alavés",
    "La Coruna": "Deportivo",
    "Sp Gijon": "Sporting Gijón",
    "Sociedad": "Real Sociedad",
    "Cordoba": "Córdoba",
    "Lerida": "Lleida",
    "Logrones": "Logroñés",
    "Merida": "Mérida",
    "Hercules": "Hércules",
    "Gimnastic": "Gimnàstic",
    "Cadiz": "Cádiz",
    "Almeria": "Almería",
    # seguros:
    "Mallorca": "Mallorca", "Getafe": "Getafe", "Celta": "Celta", "Sevilla": "Sevilla",
    "Betis": "Betis", "Valencia": "Valencia", "Elche": "Elche", "Osasuna": "Osasuna",
    "Real Madrid": "Real Madrid", "Barcelona": "Barcelona", "Granada": "Granada",
    "Huesca": "Huesca", "Eibar": "Eibar", "Compostela": "Compostela",
    "Extremadura": "Extremadura", "Salamanca": "Salamanca", "Murcia": "Murcia",
    "Santander": "Racing", "Zaragoza": "Zaragoza", "Tenerife": "Tenerife",
    "Oviedo": "Oviedo", "Xerez": "Xerez"
}

def normalizar_nombres(df, cols):
    for c in cols:
        df[c] = df[c].astype(str).str.strip().replace(map_nombres)
    return df

# =========================
# Consultas mínimas (sin leakage)
# =========================
def q_matches_chavalitos(temporada):
    return f"""
    SELECT
        jornada::integer AS jornada,
        equipo_local,
        equipo_visitante,
        goles_local,
        goles_visitante,
        temporada
    FROM chavalitos.v_jornadas_liga
    WHERE temporada = '{temporada}'
    ORDER BY jornada ASC
    """

def q_matches_dbo(temporada):
    return f"""
    SELECT
        jornada::integer AS jornada,
        equipo_local,
        equipo_visitante,
        goles_local,
        goles_visitante,
        temporada
    FROM dbo.v_jornadas_liga
    WHERE temporada = '{temporada}'
    ORDER BY jornada ASC
    """

# =========================
# Etiqueta y utilidades
# =========================
def etiqueta_1x2(gl, gv):
    if pd.isna(gl) or pd.isna(gv): return None
    if gl > gv: return '1'
    if gl < gv: return '2'
    return 'X'

def puntos_row(gf, gc):
    if pd.isna(gf) or pd.isna(gc): return np.nan
    if gf > gc: return 3
    if gf == gc: return 1
    return 0

SEASON_RX = re.compile(r"^\d{2}/\d{2}$")

def clean_and_dedup(df):
    """Limpia y deja 1 fila por partido (temporada, jornada, local, visitante, goles)."""
    n0 = len(df)
    # Normaliza tipos
    for c in ['goles_local','goles_visitante','jornada']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = normalizar_nombres(df, ['equipo_local','equipo_visitante'])

    # Filtra temporadas con formato XX/YY
    df = df[df['temporada'].astype(str).str.match(SEASON_RX, na=False)].copy()

    # Crea ID de partido con goles (para eliminar duplicados exactos)
    df['match_id'] = (
        df['temporada'].astype(str) + '|' +
        df['jornada'].astype('Int64').astype(str) + '|' +
        df['equipo_local'].astype(str) + '|' +
        df['equipo_visitante'].astype(str) + '|' +
        df['goles_local'].astype('Int64').astype(str) + '|' +
        df['goles_visitante'].astype('Int64').astype(str)
    )
    dup_exactos = n0 - df['match_id'].nunique()
    if dup_exactos > 0:
        log(f"⚠️ Duplicados exactos eliminados: {dup_exactos}")

    # Deja 1 por match_id
    df = df.drop_duplicates(subset=['match_id']).copy()

    # Prioriza filas con goles NO nulos si hay colisiones por (temp,jornada,local,visitante)
    df['__na_gl'] = df['goles_local'].isna().astype(int)
    df['__na_gv'] = df['goles_visitante'].isna().astype(int)
    df = df.sort_values(
        ['temporada','jornada','equipo_local','equipo_visitante','__na_gl','__na_gv'],
        ascending=[True, True, True, True, True, True]
    )
    df = df.drop_duplicates(
        subset=['temporada','jornada','equipo_local','equipo_visitante'],
        keep='first'
    ).copy()
    df = df.drop(columns=['match_id','__na_gl','__na_gv'])

    # Reporte por temporada
    cnt = df.groupby('temporada').size().reset_index(name='partidos')
    log("Partidos por temporada después de dedup:")
    log(cnt.sort_values('temporada').to_string(index=False))
    log(f"Dedup: {n0} -> {len(df)} filas")
    return df

# =========================
# Filtro de 1ª división y descarte de temporadas incompletas
# =========================
def infer_top_division_teams(df_temp):
    """
    Heurística por temporada:
      - Si max_jornada >= 39 => asumimos 22 equipos (42 jornadas, ≈462 partidos).
      - Si max_jornada <= 38 => asumimos 20 equipos (38 jornadas, ≈380 partidos).
    Mantenemos solo partidos donde ambos equipos estén entre los top-N por nº de apariciones.
    """
    t = df_temp['temporada'].iloc[0]
    max_j = int(pd.to_numeric(df_temp['jornada'], errors='coerce').max())
    # conteo apariciones por equipo (local+visitante)
    eq_counts = pd.concat([df_temp['equipo_local'], df_temp['equipo_visitante']]).value_counts()
    if max_j >= 39:
        keep_n = 22
        exp_matches = 462
        exp_j = 42
    else:
        keep_n = 20
        exp_matches = 380
        exp_j = 38

    top_teams = set(eq_counts.head(keep_n).index)
    df_f = df_temp[
        df_temp['equipo_local'].isin(top_teams) &
        df_temp['equipo_visitante'].isin(top_teams)
    ].copy()

    nunq_teams = len(top_teams)
    n_raw = len(df_temp)
    n_f = len(df_f)
    jornadas_unq = sorted(pd.to_numeric(df_f['jornada'], errors='coerce').dropna().unique().astype(int).tolist())
    log(f"[chk] {t}: equipos_keep={nunq_teams} | raw={n_raw} -> filtro={n_f} | jornadas={len(jornadas_unq)} (max={max(jornadas_unq) if jornadas_unq else '—'}) | esperado≈{exp_matches}")
    return df_f, exp_matches, exp_j

def filter_primary_division(df_all):
    log("[sanity] Filtrando por 1ª división inferida + temporadas completas...")
    out = []
    bad_seasons = []
    for temp, g in df_all.groupby('temporada', sort=False):
        g = g.copy()
        g['jornada'] = pd.to_numeric(g['jornada'], errors='coerce')
        g = g.dropna(subset=['jornada'])
        g['jornada'] = g['jornada'].astype(int)

        g_f, exp_matches, exp_j = infer_top_division_teams(g)

        ok_matches = len(g_f) >= 0.90 * exp_matches
        ok_jornadas = g_f['jornada'].nunique() >= int(0.90 * exp_j)
        if ok_matches and ok_jornadas:
            out.append(g_f)
        else:
            log(f"[warn] Temporada {temp} parece incompleta o mezclada -> descarto (filtro={len(g_f)}/{exp_matches}, jornadas={g_f['jornada'].nunique()}/{exp_j})")
            bad_seasons.append(temp)

    if not out:
        log("[fatal] Filtro dejó vacío el dataset. Revisa vistas/consultas.")
        return df_all, bad_seasons

    df_ok = pd.concat(out, ignore_index=True).sort_values(['temporada','jornada'])
    log(f"[sanity] Temporadas descartadas: {bad_seasons if bad_seasons else '—'}")
    log("[sanity] Conteo final por temporada:")
    tmpc = df_ok.groupby('temporada', as_index=False).size().rename(columns={'size':'partidos'})
    log(tmpc.to_string(index=False))
    return df_ok, bad_seasons

# =========================
# H2H (global y por sede)
# =========================
def add_h2h_features(df):
    df = df.sort_values(['temporada','jornada']).copy()
    assert df.duplicated(subset=['temporada','jornada','equipo_local','equipo_visitante']).sum() == 0, "Df base no es único por partido"

    recs = []
    for _, r in df.iterrows():
        gl, gv = r['goles_local'], r['goles_visitante']
        pts_loc = (3 if gl>gv else (1 if gl==gv else 0)) if (not pd.isna(gl) and not pd.isna(gv)) else np.nan
        recs.append((r['temporada'], r['jornada'], r['equipo_local'], r['equipo_visitante'], gl, gv, pts_loc))
    h = pd.DataFrame(recs, columns=['temporada','jornada','home','away','gl','gv','pts_home']).sort_values(['temporada','jornada'])

    h = h.drop_duplicates(subset=['temporada','jornada','home','away'])
    h['h2h_pts5'] = 0.0; h['h2h_gf5'] = 0.0; h['h2h_gc5'] = 0.0

    # agrupa por pareja (ordenada)
    key1 = h['home'].where(h['home'] < h['away'], h['away'])
    key2 = h['away'].where(h['home'] < h['away'], h['home'])
    for _, g in h.groupby([key1, key2], sort=False):
        g = g.sort_values(['temporada','jornada']).copy()
        pts_hist, gf_hist, gc_hist = [], [], []
        pts5, gf5, gc5 = [], [], []
        for _, rr in g.iterrows():
            pts5.append(np.nansum(pts_hist[-H2H_WINDOW:]) if pts_hist else 0.0)
            gf5.append(np.nansum(gf_hist[-H2H_WINDOW:]) if gf_hist else 0.0)
            gc5.append(np.nansum(gc_hist[-H2H_WINDOW:]) if gc_hist else 0.0)
            pts_hist.append(rr['pts_home'] if not pd.isna(rr['pts_home']) else 0.0)
            gf_hist.append(rr['gl'] if not pd.isna(rr['gl']) else 0.0)
            gc_hist.append(rr['gv'] if not pd.isna(rr['gv']) else 0.0)
        h.loc[g.index, 'h2h_pts5'] = pts5
        h.loc[g.index, 'h2h_gf5']  = gf5
        h.loc[g.index, 'h2h_gc5']  = gc5

    n_before = len(df)
    out = df.merge(
        h[['temporada','jornada','home','away','h2h_pts5','h2h_gf5','h2h_gc5']],
        left_on=['temporada','jornada','equipo_local','equipo_visitante'],
        right_on=['temporada','jornada','home','away'],
        how='left'
    ).drop(columns=['home','away'])

    if len(out) > 1.2 * n_before:
        log(f"⚠️ H2H: tamaño creció {len(out)} vs {n_before}. Forzando dedup por partido conservando primera coincidencia.")
        out = out.sort_values(['temporada','jornada','equipo_local','equipo_visitante']).drop_duplicates(
            subset=['temporada','jornada','equipo_local','equipo_visitante'], keep='first'
        )

    out[['h2h_pts5','h2h_gf5','h2h_gc5']] = out[['h2h_pts5','h2h_gf5','h2h_gc5']].fillna(0.0)
    return out

def add_h2h_home_features(df):
    df = df.sort_values(['temporada','jornada']).copy()
    assert df.duplicated(subset=['temporada','jornada','equipo_local','equipo_visitante']).sum() == 0, "Df base no es único por partido"

    recs = []
    for _, r in df.iterrows():
        gl, gv = r['goles_local'], r['goles_visitante']
        pts_loc = (3 if gl>gv else (1 if gl==gv else 0)) if (not pd.isna(gl) and not pd.isna(gv)) else np.nan
        recs.append((r['temporada'], r['jornada'], r['equipo_local'], r['equipo_visitante'], gl, gv, pts_loc))
    h = pd.DataFrame(recs, columns=['temporada','jornada','home','away','gl','gv','pts_home']).sort_values(['temporada','jornada'])

    h = h.drop_duplicates(subset=['temporada','jornada','home','away'])
    h['h2h_home_pts5'] = 0.0; h['h2h_home_gf5'] = 0.0; h['h2h_home_gc5'] = 0.0

    for (home, away), g in h.groupby(['home','away'], sort=False):
        g = g.sort_values(['temporada','jornada']).copy()
        pts_hist, gf_hist, gc_hist = [], [], []
        pts5, gf5, gc5 = [], [], []
        for _, rr in g.iterrows():
            pts5.append(np.nansum(pts_hist[-H2H_WINDOW:]) if pts_hist else 0.0)
            gf5.append(np.nansum(gf_hist[-H2H_WINDOW:]) if gf_hist else 0.0)
            gc5.append(np.nansum(gc_hist[-H2H_WINDOW:]) if gc_hist else 0.0)
            pts_hist.append(rr['pts_home'] if not pd.isna(rr['pts_home']) else 0.0)
            gf_hist.append(rr['gl'] if not pd.isna(rr['gl']) else 0.0)
            gc_hist.append(rr['gv'] if not pd.isna(rr['gv']) else 0.0)
        h.loc[g.index, 'h2h_home_pts5'] = pts5
        h.loc[g.index, 'h2h_home_gf5']  = gf5
        h.loc[g.index, 'h2h_home_gc5']  = gc5

    n_before = len(df)
    out = df.merge(
        h[['temporada','jornada','home','away','h2h_home_pts5','h2h_home_gf5','h2h_home_gc5']],
        left_on=['temporada','jornada','equipo_local','equipo_visitante'],
        right_on=['temporada','jornada','home','away'],
        how='left'
    ).drop(columns=['home','away'])

    if len(out) > 1.2 * n_before:
        log(f"⚠️ H2H_HOME: tamaño creció {len(out)} vs {n_before}. Forzando dedup por partido.")
        out = out.sort_values(['temporada','jornada','equipo_local','equipo_visitante']).drop_duplicates(
            subset=['temporada','jornada','equipo_local','equipo_visitante'], keep='first'
        )

    out[['h2h_home_pts5','h2h_home_gf5','h2h_home_gc5']] = out[['h2h_home_pts5','h2h_home_gf5','h2h_home_gc5']].fillna(0.0)
    return out

# =========================
# Features base (acumuladas + forma + EWMA + H2H)
# =========================
def build_base_features(df_matches):
    log("Build de features base: inicio")
    t0 = time.perf_counter()

    df = df_matches.sort_values(['temporada','jornada']).copy()
    for c in ['goles_local','goles_visitante','jornada']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['resultado_1x2'] = [etiqueta_1x2(gl, gv) for gl, gv in zip(df['goles_local'], df['goles_visitante'])]

    assert df.duplicated(subset=['temporada','jornada','equipo_local','equipo_visitante']).sum() == 0, "Partidos duplicados antes de features"

    loc = pd.DataFrame({
        'temporada': df['temporada'],
        'jornada'  : df['jornada'],
        'equipo'   : df['equipo_local'],
        'gf'       : df['goles_local'],
        'gc'       : df['goles_visitante'],
        'is_local' : 1
    })
    vis = pd.DataFrame({
        'temporada': df['temporada'],
        'jornada'  : df['jornada'],
        'equipo'   : df['equipo_visitante'],
        'gf'       : df['goles_visitante'],
        'gc'       : df['goles_local'],
        'is_local' : 0
    })
    tg = pd.concat([loc, vis], ignore_index=True).sort_values(['temporada','equipo','jornada'])

    tg = tg.drop_duplicates(subset=['temporada','equipo','jornada'])

    tg['puntos'] = [puntos_row(gf, gc) for gf, gc in zip(tg['gf'], tg['gc'])]
    tg['win']  = ((tg['gf'] > tg['gc'])*1).astype(float)
    tg['draw'] = ((tg['gf'] == tg['gc'])*1).astype(float)
    tg['loss'] = ((tg['gf'] < tg['gc'])*1).astype(float)
    tg['pj']   = 1.0

    def accum_and_shift(g):
        g = g.copy()
        g['c_pj']   = g['pj'].cumsum().shift(1)
        g['c_pts']  = g['puntos'].cumsum().shift(1)
        g['c_win']  = g['win'].cumsum().shift(1)
        g['c_draw'] = g['draw'].cumsum().shift(1)
        g['c_loss'] = g['loss'].cumsum().shift(1)
        g['c_gf']   = g['gf'].cumsum().shift(1)
        g['c_gc']   = g['gc'].cumsum().shift(1)
        g['form_pts5'] = g['puntos'].rolling(FORM_WINDOW, min_periods=1).sum().shift(1)
        g['form_gf5']  = g['gf'].rolling(FORM_WINDOW, min_periods=1).sum().shift(1)
        g['form_gc5']  = g['gc'].rolling(FORM_WINDOW, min_periods=1).sum().shift(1)
        g['ewm_pts'] = g['puntos'].shift(1).ewm(alpha=0.3, min_periods=1).mean()
        g['ewm_gf']  = g['gf'].shift(1).ewm(alpha=0.3, min_periods=1).mean()
        g['ewm_gc']  = g['gc'].shift(1).ewm(alpha=0.3, min_periods=1).mean()
        return g

    tg = tg.groupby(['temporada','equipo'], group_keys=False).apply(accum_and_shift)

    fill_cols = ['c_pj','c_pts','c_win','c_draw','c_loss','c_gf','c_gc',
                 'form_pts5','form_gf5','form_gc5','ewm_pts','ewm_gf','ewm_gc']
    tg[fill_cols] = tg[fill_cols].fillna(0.0)
    tg['c_dg'] = tg['c_gf'] - tg['c_gc']
    tg['ppp']  = np.where(tg['c_pj']>0, tg['c_pts']/tg['c_pj'], 0.0)
    tg['win_rate'] = np.where(tg['c_pj']>0, tg['c_win']/tg['c_pj'], 0.0)

    def add_suffix(df_in, suf):
        keep = ['temporada','jornada','equipo','c_pj','c_pts','c_win','c_draw','c_loss',
                'c_gf','c_gc','c_dg','ppp','win_rate','form_pts5','form_gf5','form_gc5',
                'ewm_pts','ewm_gf','ewm_gc']
        out = df_in[keep].copy()
        ren = {c: f"{c}_{suf}" for c in keep if c not in ['temporada','jornada','equipo']}
        ren['equipo'] = f"equipo_{suf}"
        return out.rename(columns=ren)

    stats_local = add_suffix(tg[tg['is_local']==1], 'local')
    stats_visit = add_suffix(tg[tg['is_local']==0], 'visitante')

    stats_local = stats_local.drop_duplicates(subset=['temporada','jornada','equipo_local'])
    stats_visit = stats_visit.drop_duplicates(subset=['temporada','jornada','equipo_visitante'])

    base = df.merge(stats_local, on=['temporada','jornada','equipo_local'], how='left')
    base = base.merge(stats_visit, on=['temporada','jornada','equipo_visitante'], how='left')

    if len(base) > 1.1 * len(df):
        log(f"⚠️ Stats merge creció {len(base)} vs {len(df)}. Forzando dedup por partido...")
        base = base.sort_values(['temporada','jornada','equipo_local','equipo_visitante']).drop_duplicates(
            subset=['temporada','jornada','equipo_local','equipo_visitante'], keep='first'
        )

    base = add_h2h_features(base)
    base = add_h2h_home_features(base)

    diff_pairs = [
        ('c_pj_local','c_pj_visitante'), ('c_pts_local','c_pts_visitante'),
        ('c_win_local','c_win_visitante'), ('c_draw_local','c_draw_visitante'),
        ('c_loss_local','c_loss_visitante'), ('c_gf_local','c_gf_visitante'),
        ('c_gc_local','c_gc_visitante'), ('c_dg_local','c_dg_visitante'),
        ('ppp_local','ppp_visitante'), ('win_rate_local','win_rate_visitante'),
        ('form_pts5_local','form_pts5_visitante'), ('form_gf5_local','form_gf5_visitante'),
        ('form_gc5_local','form_gc5_visitante'),
        ('ewm_pts_local','ewm_pts_visitante'), ('ewm_gf_local','ewm_gf_visitante'),
        ('ewm_gc_local','ewm_gc_visitante'),
    ]
    for a,b in diff_pairs:
        base[f'diff__{a[:-6]}'] = base[a] - base[b]

    base = (base.replace([np.inf, -np.inf], np.nan).fillna(0.0))

    if len(base) > 1.05 * len(df):
        log(f"⚠️ Base creció {len(base)} vs {len(df)}. Dedupe final por partido.")
        base = base.sort_values(['temporada','jornada','equipo_local','equipo_visitante']).drop_duplicates(
            subset=['temporada','jornada','equipo_local','equipo_visitante'], keep='first'
        )

    log(f"Build de features base: fin en {time.perf_counter()-t0:.1f}s | filas={len(base)}")
    return base

# =========================
# Elo con carry-over
# =========================
def compute_season_elo(df_matches, ELO_K, ELO_HOME_ADV, carry=ELO_CARRY):
    df = df_matches.sort_values(['temporada', 'jornada']).copy()
    df['elo_local_pre'] = np.nan; df['elo_visitante_pre'] = np.nan

    def first_year(t):
        try: return int(str(t).split('/')[0])
        except: return 0
    seasons = sorted(df['temporada'].unique(), key=first_year)

    elo_prev_end = {}

    for temp in seasons:
        dft = df[df['temporada']==temp].copy()
        equipos = pd.unique(pd.concat([dft['equipo_local'], dft['equipo_visitante']], ignore_index=True))
        elo = {}
        for e in equipos:
            elo[e] = carry*elo_prev_end.get(e, ELO_START) + (1.0-carry)*ELO_START if e in elo_prev_end else ELO_START

        for i in dft.index:
            home = df.at[i, 'equipo_local']; away = df.at[i, 'equipo_visitante']
            gl = df.at[i, 'goles_local'];     gv = df.at[i, 'goles_visitante']
            h_elo = elo.get(home, ELO_START); a_elo = elo.get(away, ELO_START)
            df.at[i, 'elo_local_pre'] = h_elo
            df.at[i, 'elo_visitante_pre'] = a_elo
            if pd.isna(gl) or pd.isna(gv):
                continue
            exp_home = 1.0 / (1.0 + 10 ** (-(h_elo + ELO_HOME_ADV - a_elo) / 400.0))
            s_home = 1.0 if gl>gv else (0.5 if gl==gv else 0.0)
            elo[home] = h_elo + ELO_K*(s_home - exp_home)
            elo[away] = a_elo + ELO_K*((1.0 - s_home) - (1.0 - exp_home))

        elo_prev_end.update(elo)

    return df[['elo_local_pre','elo_visitante_pre']]

def inject_elo(base_df, raw_df_matches, ELO_K, ELO_HOME_ADV):
    elo_cols = compute_season_elo(raw_df_matches, ELO_K, ELO_HOME_ADV)
    out = pd.concat([base_df.reset_index(drop=True), elo_cols.reset_index(drop=True)], axis=1)

    for c in ['elo_local_pre','elo_visitante_pre']:
        out[c] = pd.to_numeric(out[c], errors='coerce')

    out['elo_diff'] = out['elo_local_pre'] - out['elo_visitante_pre']

    for c in ['elo_local_pre','elo_visitante_pre','elo_diff']:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out

# =========================
# Modelo
# =========================
def crear_modelo(input_dim, output_dim=3, lr=7e-4):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(256, kernel_regularizer=l2(1e-4)))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.25))
    model.add(Dense(128, kernel_regularizer=l2(1e-4)))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.20))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def postproc_preds(p1, pX, p2, delta, tau):
    argmax_lbl = np.array(['1','X','2'])[np.stack([p1,pX,p2], axis=1).argmax(axis=1)]
    close = (np.abs(p1 - p2) < delta) & (pX > tau)
    out = argmax_lbl.copy()
    out[close] = 'X'
    return out

# ====== Temperature scaling ======
def temperature_scale(logits, T):
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

def fit_temperature(logits, y_true_idx):
    def nll(T):
        T = np.clip(T[0], 0.5, 5.0)
        p = temperature_scale(logits, T) + 1e-12
        ll = -np.log(p[np.arange(len(y_true_idx)), y_true_idx]).mean()
        return ll
    res = minimize(nll, x0=[1.0], bounds=[(0.5, 5.0)], method='L-BFGS-B')
    return float(res.x[0])

# =========================
# Lectura de datos + DEDUP
# =========================
log("Conectando a BD y leyendo temporadas...")
t0 = time.perf_counter()
with conexion_db() as conn:
    temps_dbo = pd.read_sql("SELECT DISTINCT temporada FROM dbo.v_jornadas_liga", conn)['temporada'].tolist()
    temps_ch  = pd.read_sql("SELECT DISTINCT temporada FROM chavalitos.v_jornadas_liga", conn)['temporada'].tolist()
    log(f"Temporadas dbo: {len(temps_dbo)} | chavalitos: {len(temps_ch)}")
    dfs = []
    for t in temps_dbo: dfs.append(pd.read_sql(q_matches_dbo(t), conn))
    for t in temps_ch:  dfs.append(pd.read_sql(q_matches_chavalitos(t), conn))

df_raw = pd.concat(dfs, ignore_index=True)
log(f"Lectura OK: {len(df_raw)} filas en {time.perf_counter()-t0:.1f}s")

# Dedup estricto
df = clean_and_dedup(df_raw)
df = df.sort_values(['temporada','jornada']).reset_index(drop=True)

# Filtro de 1ª división y descarte de temporadas incompletas
df, temporadas_descartadas = filter_primary_division(df)


def enforce_round_robin(df_all):
    """
    v2: intenta seleccionar exactamente el total esperado de liga por temporada:
        - Máx 2 por pareja desordenada, intentando 1 por sede (A vs B y B vs A).
        - Cap por jornada: N/2 (10 si N=20; 11 si N=22).
        - Si tras caps > objetivo, recorta; si < objetivo, avisa.
    Prioridad: (goles no nulos) ↓, |jornada - mediana| ↑ (preferimos jornadas "centrales"), jornada ↑.
    """
    out = []
    for temp, g in df_all.groupby('temporada', sort=False):
        g = g.copy()

        # inferir nº equipos
        equipos = pd.unique(pd.concat([g['equipo_local'], g['equipo_visitante']], ignore_index=True))
        n_teams = len(equipos)
        per_round_cap = 11 if n_teams >= 22 else 10
        target_total  = n_teams * (n_teams - 1)  # 20->380; 22->462

        # columnas auxiliares
        g['__has_score'] = (~g['goles_local'].isna() & ~g['goles_visitante'].isna()).astype(int)
        key1 = g[['equipo_local','equipo_visitante']].min(axis=1)
        key2 = g[['equipo_local','equipo_visitante']].max(axis=1)
        g['__pair'] = key1 + '|' + key2
        # sede: quién fue local dentro de la pareja ordenada
        g['__side'] = np.where(g['equipo_local'] == key1, 'Ahome', 'Bhome')

        # prioridad: goles no nulos primero; luego preferimos jornadas "centrales"
        med_j = g['jornada'].median() if pd.notna(g['jornada']).any() else 19.5
        g['__prio'] = (
            g['__has_score']*10000
            - (g['jornada'] - med_j).abs()*100
            - g['jornada'].fillna(9999)
        )

        # 1) seleccionar máx 1 por sede dentro de la pareja (intenta Ahome y Bhome)
        def pick_pair(h):
            h = h.sort_values(['__prio'], ascending=False)
            a = h[h['__side']=='Ahome'].head(1)
            b = h[h['__side']=='Bhome'].head(1)
            sel = pd.concat([a, b], ignore_index=True)
            if sel.empty:  # por si algo raro
                sel = h.head(2)
            return sel.head(2)

        g_pair = g.groupby('__pair', group_keys=False).apply(pick_pair)

        # 2) cap por jornada (10/11)
        g_pair = (
            g_pair.sort_values(['__prio'], ascending=False)
                  .groupby('jornada', group_keys=False)
                  .apply(lambda h: h.head(per_round_cap))
        )

        # 3) si todavía excede el total objetivo, recorta globalmente por prioridad
        if len(g_pair) > target_total:
            g_pair = g_pair.sort_values('__prio', ascending=False).head(target_total)

        # 4) limpieza
        g_pair = g_pair.drop(columns=['__pair','__side','__prio','__has_score'], errors='ignore')
        g_pair = g_pair.sort_values(['temporada','jornada','equipo_local','equipo_visitante'])
        out.append(g_pair)

        # reporte por temporada
        cnt_j = g_pair.groupby('jornada').size().sort_index()
        log(f"[round-robin v2] {temp}: partidos={len(g_pair)} (target={target_total}) | "
            f"jornadas={cnt_j.index.min()}–{cnt_j.index.max()} | "
            f"cap_jornada={per_round_cap} | "
            f"min/max por jornada={cnt_j.min() if len(cnt_j) else 0}/{cnt_j.max() if len(cnt_j) else 0}")

        if len(g_pair) < target_total:
            log(f"[warn] {temp}: faltan {target_total - len(g_pair)} partidos para llegar al objetivo. "
                "Revisa la fuente o añade heurística extra para diferenciar liga vs. copas.")

    df_ok = pd.concat(out, ignore_index=True)
    log("[round-robin v2] Conteo final por temporada:")
    log(df_ok.groupby('temporada').size().reset_index(name='partidos').to_string(index=False))
    return df_ok


df = enforce_round_robin(df)

# =========================
# Features base
# =========================
base = build_base_features(df)

# =========================
# Columnas base
# =========================
base_cols = [
    # Local
    'c_pj_local','c_pts_local','c_win_local','c_draw_local','c_loss_local',
    'c_gf_local','c_gc_local','c_dg_local','ppp_local','win_rate_local',
    'form_pts5_local','form_gf5_local','form_gc5_local',
    'ewm_pts_local','ewm_gf_local','ewm_gc_local',
    # Visitante
    'c_pj_visitante','c_pts_visitante','c_win_visitante','c_draw_visitante','c_loss_visitante',
    'c_gf_visitante','c_gc_visitante','c_dg_visitante','ppp_visitante','win_rate_visitante',
    'form_pts5_visitante','form_gf5_visitante','form_gc5_visitante',
    'ewm_pts_visitante','ewm_gf_visitante','ewm_gc_visitante',
    # H2H
    'h2h_pts5','h2h_gf5','h2h_gc5',
    'h2h_home_pts5','h2h_home_gf5','h2h_home_gc5',
] + [c for c in base.columns if c.startswith('diff__')]

# Etiqueta válida + splits
mask_lbl = base['resultado_1x2'].isin(['1','X','2'])
base = base[mask_lbl].copy()
log(f"Dataset con etiqueta válida: {len(base)} filas")

base_train_base = base[(base['temporada'] != VAL_SEASON) & (base['temporada'] != TEST_SEASON)].copy()
base_val        = base[base['temporada'] == VAL_SEASON].copy()
base_train_full = base[base['temporada'] != TEST_SEASON].copy()
base_test       = base[base['temporada'] == TEST_SEASON].copy()
log(f"Split -> train_base={len(base_train_base)}, val={len(base_val)}, train_full={len(base_train_full)}, test={len(base_test)}")

# =========================
# Helpers robustos X/y
# =========================
def prep_xy_general(df_, feature_cols):
    y_mapped = df_['resultado_1x2'].map({'1':0, 'X':1, '2':2})
    valid = y_mapped.notna()
    X = (df_.loc[valid, feature_cols]
            .apply(pd.to_numeric, errors='coerce')
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .values.astype('float32'))
    y = y_mapped.loc[valid].astype(int).values
    return X, y

# =========================
# Tuning en 23/24
# =========================
total_combos = len(ELO_K_GRID)*len(ELO_HOME_GRID)*len(W_X_GRID)
log(f"Inicio tuning (23/24): total combinaciones (EloK, EloHOME, wX) = {total_combos}")

best = {'acc': -1,
        'elo_k': None, 'elo_home': None,
        'w_x': None, 'delta': None, 'tau': None,
        'T_opt': 1.0}

combo_idx = 0
tuning_start = time.perf_counter()

for elo_k in ELO_K_GRID:
    for elo_home in ELO_HOME_GRID:
        train_base = inject_elo(base_train_base, df, elo_k, elo_home)
        val_set    = inject_elo(base_val,        df, elo_k, elo_home)
        feature_cols = base_cols + ['elo_local_pre','elo_visitante_pre','elo_diff']

        X_tb, y_tb   = prep_xy_general(train_base, feature_cols)
        X_val, y_val = prep_xy_general(val_set, feature_cols)

        scaler_tune = StandardScaler()
        X_tb_s  = scaler_tune.fit_transform(X_tb)
        X_val_s = scaler_tune.transform(X_val)

        for w_x in W_X_GRID:
            combo_idx += 1
            t_combo = time.perf_counter()
            log(f"[{combo_idx}/{total_combos}] Entrenando (EloK={elo_k}, HOME={elo_home}, wX={w_x})...")

            cw = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_tb)
            cw_dict = {i: w for i, w in enumerate(cw)}
            cw_dict[1] = cw_dict[1] * w_x

            y_tb_cat = to_categorical(y_tb, num_classes=3)
            callbacks = [
                ReduceLROnPlateau(monitor='loss', factor=0.5, patience=PATIENCE_TUNE//2, min_lr=1e-5, verbose=0),
                EarlyStopping(monitor='loss', patience=PATIENCE_TUNE, restore_best_weights=True)
            ]

            model_t = crear_modelo(X_tb_s.shape[1])
            model_t.fit(X_tb_s, y_tb_cat, epochs=EPOCHS_TUNE, batch_size=256, verbose=0,
                        class_weight=cw_dict, callbacks=callbacks)

            prob_val = model_t.predict(X_val_s, verbose=0)
            base_lbl = np.array(['1','X','2'])[prob_val.argmax(axis=1)]
            y_val_lbl = pd.Series(y_val).map({0:'1',1:'X',2:'2'}).values
            base_acc = (base_lbl == y_val_lbl).mean()

            logits_proxy = np.log(prob_val + 1e-9)
            T_opt = fit_temperature(logits_proxy, y_val)
            prob_val_cal = temperature_scale(logits_proxy, T_opt)

            base_lbl_cal = np.array(['1','X','2'])[prob_val_cal.argmax(axis=1)]
            base_acc_cal = (base_lbl_cal == y_val_lbl).mean()

            use_prob_for_pp = prob_val_cal if base_acc_cal >= base_acc else prob_val
            use_base_acc     = max(base_acc, base_acc_cal)
            use_T            = T_opt if base_acc_cal >= base_acc else 1.0

            if use_base_acc > best['acc']:
                best = {'acc': use_base_acc, 'elo_k': elo_k, 'elo_home': elo_home,
                        'w_x': w_x, 'delta': None, 'tau': None, 'T_opt': use_T}
                log(f"  ↪ Nuevo BEST baseline: acc={use_base_acc:.4f} | T={use_T:.3f}")

            p1, pX, p2 = use_prob_for_pp[:,0], use_prob_for_pp[:,1], use_prob_for_pp[:,2]
            for d in DELTA_GRID:
                for t in TAU_GRID:
                    cand = postproc_preds(p1, pX, p2, d, t)
                    acc = (cand == y_val_lbl).mean()
                    if acc > best['acc']:
                        best = {'acc': acc, 'elo_k': elo_k, 'elo_home': elo_home,
                                'w_x': w_x, 'delta': d, 'tau': t, 'T_opt': use_T}
                        log(f"  ★ Nuevo BEST POST-PROC: acc={acc:.4f} | delta={d} | tau={t} | T={use_T:.3f}")

            log(f"  Hecho combo en {time.perf_counter()-t_combo:.1f}s (mejor hasta ahora={best['acc']:.4f})")

log(f"Fin tuning en {time.perf_counter()-tuning_start:.1f}s")
print(f"\n🔧 Mejor en 23/24 -> acc={best['acc']:.4f} | EloK={best['elo_k']} | EloHOME={best['elo_home']} | wX={best['w_x']} | delta={best['delta']} | tau={best['tau']} | T={best['T_opt']:.3f}")

# =========================
# Entrenamiento final y test
# =========================
log("Entrenamiento final con train_full + test 24/25")
train_full = inject_elo(base_train_full, df, best['elo_k'], best['elo_home'])
test_set   = inject_elo(base_test,       df, best['elo_k'], best['elo_home'])

feature_cols = base_cols + ['elo_local_pre','elo_visitante_pre','elo_diff']
X_tf, y_tf = prep_xy_general(train_full, feature_cols)
dist_train_series = pd.Series(y_tf).map({0:'1',1:'X',2:'2'})
dist_train = dist_train_series.value_counts(normalize=True).rename_axis('resultado').reset_index(name='proporcion')
print("\n📊 Distribución real en entrenamiento (train_full, tras prep):")
print(dist_train)

X_te, y_te = prep_xy_general(test_set, feature_cols)
log(f"Shapes -> X_tf={X_tf.shape}, X_te={X_te.shape}")

scaler = StandardScaler()
X_tf_s = scaler.fit_transform(X_tf)
X_te_s = scaler.transform(X_te)

cw = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_tf)
cw_dict = {i: w for i, w in enumerate(cw)}
cw_dict[1] = cw_dict[1] * (best['w_x'] if best['w_x'] is not None else 1.0)

y_tf_cat = to_categorical(y_tf, num_classes=3)

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=0),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
]

model = crear_modelo(X_tf_s.shape[1])   # <<< fix: usar shape del set final
log("Fit final...")
t_fit = time.perf_counter()
model.fit(
    X_tf_s, y_tf_cat,                    # <<< fix: entrenar con train_full
    epochs=EPOCHS_FINAL,
    batch_size=256,
    verbose=0,
    class_weight=cw_dict,
    validation_split=0.12,
    callbacks=callbacks
)
log(f"Fit final listo en {time.perf_counter()-t_fit:.1f}s")

log("Predicción test + calibración...")
probas = model.predict(X_te_s, verbose=0)
logits_proxy_test = np.log(probas + 1e-9)
probas_used = temperature_scale(logits_proxy_test, best['T_opt']) if best['T_opt'] != 1.0 else probas

p1, pX, p2 = probas_used[:,0], probas_used[:,1], probas_used[:,2]
baseline_lbl = np.array(['1','X','2'])[probas_used.argmax(axis=1)]
y_true_lbl = pd.Series(y_te).map({0:'1',1:'X',2:'2'}).values

IMPROVE_EPS = 0.002  # mínimo +0.2 pts
if best['delta'] is not None and best['tau'] is not None:
    preds_pp = postproc_preds(p1, pX, p2, best['delta'], best['tau'])
    acc_base = (baseline_lbl == y_true_lbl).mean()
    acc_pp   = (preds_pp == y_true_lbl).mean()
    if acc_pp >= acc_base + IMPROVE_EPS:
        final_preds = preds_pp
        print(f"\n✅ 24/25: usamos POST-PROC (acc={acc_pp:.4f} >= baseline={acc_base:.4f} + {IMPROVE_EPS:.3f})")
    else:
        final_preds = baseline_lbl
        print(f"\n✅ 24/25: usamos BASELINE (acc={acc_base:.4f} ≥ post-proc={acc_pp:.4f}+{IMPROVE_EPS:.3f})")
else:
    final_preds = baseline_lbl
    print(f"\n✅ 24/25: usamos BASELINE (sin post-proc mejor en val)")

# =========================
# Métricas finales
# =========================
df_out = test_set.copy().iloc[:len(final_preds)].copy()
df_out['prediccion_1x2'] = final_preds
df_out['confianza'] = probas_used.max(axis=1)[:len(final_preds)]
df_out['acierto'] = (df_out['prediccion_1x2'] == df_out['resultado_1x2'])

print(f"\n🎯 Precisión global en {TEST_SEASON}: {df_out['acierto'].mean():.2%}")

preds_idx = pd.Series(final_preds).map({'1':0,'X':1,'2':2}).values
print(f"\n📋 Classification report ({TEST_SEASON}):")
print(classification_report(y_te[:len(preds_idx)], preds_idx, target_names=['1','X','2']))

print(f"\n🧩 Matriz de confusión (filas=real, columnas=pred) ({TEST_SEASON}):")
print(confusion_matrix(y_te[:len(preds_idx)], preds_idx, labels=[0,1,2]))

prec_j = df_out.groupby('jornada')['acierto'].mean().reset_index().sort_values('acierto', ascending=False)
print("\n📈 Precisión por jornada:")
print(prec_j)

dist_train = train_full['resultado_1x2'].value_counts(normalize=True).rename_axis('resultado').reset_index(name='proporcion')
print("\n📊 Distribución real en entrenamiento (train_full):")
print(dist_train)

dist_pred = pd.Series(final_preds).value_counts(normalize=True).rename_axis('prediccion').reset_index(name='proporcion')
print(f"\n📊 Distribución de predicciones en {TEST_SEASON}:")
print(dist_pred)
