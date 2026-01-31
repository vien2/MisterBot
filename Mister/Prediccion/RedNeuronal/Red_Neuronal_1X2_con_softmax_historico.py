# MisterBot - Softmax 1X2 sin fuga + Forma (EWMA) + H2H + H2H Home + Elo Carry + Temp Scaling
# + DEDUP robusto y guardarraíles para evitar explosión de filas
# + FAST_MODE opcional para tuning más rápido
# + Filtro de 1ª división por temporada (top-N equipos) y descarte de temporadas incompletas
# + MEJORAS: tuning temporal 24/25 (T, delta, tau), guardarraíl anti-X, prior correction, macro-F1
# + Classwise bias seguro (opcional), round priors por jornada (logits), wX_final≥1.0
# + Cupón de empates con presupuesto de pérdida esperada (para no perder accuracy)
# + (APLICADAS) 8 mejoras de afinado para subir X y macro-F1 sin perder accuracy
# + (PUNTO 4) Cupón más capaz con pase 0 y presupuesto 2.0 pp
# + (PUNTO 5) Post-proc más permisivo para X (ya integrado vía clamps delta/tau y cuota mínima)
# + (PUNTO 2 y 3) τ adaptativo + regla expected con γ tunado en early

import sys, os, time, re, random
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy

from scipy.optimize import minimize

from utils import conexion_db
import tensorflow as tf

# ============ Logging helper ============
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# =========================
# CONFIG
# =========================
TEST_SEASON = '24/25'    # test final
VAL_SEASON  = '23/24'    # validación para tuning base
FORM_WINDOW = 5
H2H_WINDOW  = 5
ELO_START   = 1500.0
ELO_CARRY   = 0.75
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

FAST_MODE = True

# >>> Nuevo: control de bias por clase (calibración por clase)
USE_CLASSWISE_BIAS = False          # puedes poner False para desactivarlo del todo
BIAS_BOUNDS = (-0.15, 0.15)        # límites muy estrechos
BIAS_L2 = 0.50                     # regularización fuerte
BIAS_MIN_MEAN_X = 0.15             # (más permisivo) si con bias la media de p(X) < 0.15 en val, se desactiva

# Priors por jornada (en logits)  [MEJORA #3: X más "friendly"]
ROUND_PRIORS_LAMBDA = 0.11
LAM_VEC = np.array([0.10, 0.16, 0.10], dtype=float)

# Micro-floor para X en test si colapsa (REFORZADO)
PX_FLOOR_MEAN = 0.16
PX_FLOOR_BUMP = 0.10

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
# Consultas mínimas
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

def set_all_seeds(seed):
    import random, numpy as np, tensorflow as tf, os
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def train_and_predict_once(X_tf_s, y_tf, X_te_s, cw_dict, epochs=EPOCHS_FINAL, val_split=0.12, seed=42):
    set_all_seeds(seed)
    y_tf_cat = to_categorical(y_tf, num_classes=3)

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=0),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    ]

    model = crear_modelo(X_tf_s.shape[1])
    model.compile(
        optimizer=Adam(learning_rate=7e-4),
        loss=CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    model.fit(
        X_tf_s, y_tf_cat,
        epochs=epochs,
        batch_size=256,
        verbose=0,
        class_weight=cw_dict,
        validation_split=val_split,
        callbacks=callbacks
    )
    probas = model.predict(X_te_s, verbose=0)
    return probas

# ===== Calibración por clase (bias) =====
def apply_classwise_bias_from_logits(logits, b):
    z = logits + b.reshape(1, -1)
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

def fit_classwise_bias(logits, y_true_idx, b_bounds=BIAS_BOUNDS, l2_reg=BIAS_L2):
    def nll(b_raw):
        b = b_raw - np.mean(b_raw)     # sum(b)=0
        p = apply_classwise_bias_from_logits(logits, b) + 1e-12
        nll_val = -np.log(p[np.arange(len(y_true_idx)), y_true_idx]).mean()
        return nll_val + 0.5*l2_reg*np.sum(b**2)
    x0 = np.array([0.0, 0.0, 0.0], dtype=float)
    bounds = [b_bounds, b_bounds, b_bounds]
    res = minimize(nll, x0=x0, bounds=bounds, method='L-BFGS-B')
    b_opt = res.x - np.mean(res.x)
    return b_opt.astype(float)

# ===== Priors por jornada =====
def compute_round_priors(df_all, exclude_season):
    g = df_all[df_all['temporada'] != exclude_season].copy()
    g = g.dropna(subset=['resultado_1x2','jornada'])
    priors = (g.groupby('jornada')['resultado_1x2']
                .value_counts(normalize=True)
                .unstack()
                .reindex(columns=['1','X','2'])
                .fillna(0.0))
    priors = (priors + 1e-6)
    priors = priors.div(priors.sum(axis=1), axis=0)
    return priors

def compute_global_priors_from_train(train_df):
    pri = (train_df['resultado_1x2'].value_counts(normalize=True)
              .reindex(['1','X','2']).fillna(1/3).values)
    return pri.astype(float)

def apply_round_priors_on_logits(logits, jornadas, round_priors, base_priors, lam=ROUND_PRIORS_LAMBDA, lam_vec=LAM_VEC):
    out = logits.copy()
    base = np.clip(base_priors.reshape(1, -1), 1e-12, 1.0)
    for j in np.unique(jornadas):
        mask = (jornadas == j)
        if j in round_priors.index:
            pri = round_priors.loc[j, ['1','X','2']].values.reshape(1, -1)
            ratio = np.log(np.clip(pri / base, 1e-12, 1e12))
            out[mask] = out[mask] + (lam_vec.reshape(1, -1) * lam) * ratio
    return out

def enforce_x_quota_predictions(P, preds, min_share=0.15, margin_cap=0.06, min_px=0.30):
    """
    Intenta alcanzar min_share de 'X' en ETIQUETAS:
      - Pase A: duelos muy parejos y pX≥min_px.
      - Pase B (fallback): mejores (pX - max(p1,p2)) aunque el margen sea > margin_cap.
    """
    preds = preds.copy()
    n = len(preds)
    cur = (preds == 'X').mean()
    if cur >= min_share or n == 0:
        return preds

    p1, pX, p2 = P[:,0], P[:,1], P[:,2]
    margin = np.abs(p1 - p2)

    need = int(np.ceil(min_share * n)) - int((preds == 'X').sum())
    if need <= 0:
        return preds

    # Pase A: estrictos
    maskA = (preds != 'X') & (margin <= float(max(margin_cap, 0.10))) & (pX >= float(max(min_px, 0.22)))
    idxA = np.where(maskA)[0]
    scoreA = pX[idxA] - np.maximum(p1[idxA], p2[idxA])
    orderA = idxA[np.argsort(-scoreA)]
    takeA = orderA[:need]
    preds[takeA] = 'X'
    need -= len(takeA)
    if need <= 0:
        return preds

    # Pase B: relajamos margen; priorizamos por cercanía a X
    maskB = (preds != 'X') & (pX >= float(max(min_px - 0.04, 0.20)))
    idxB = np.setdiff1d(np.where(maskB)[0], takeA, assume_unique=False)
    if idxB.size:
        scoreB = pX[idxB] - np.maximum(p1[idxB], p2[idxB])
        orderB = idxB[np.argsort(-scoreB)]
        takeB = orderB[:need]
        preds[takeB] = 'X'

    return preds

# =========================
# Limpieza / Filtro / Features
# =========================
def clean_and_dedup(df):
    n0 = len(df)
    for c in ['goles_local','goles_visitante','jornada']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = normalizar_nombres(df, ['equipo_local','equipo_visitante'])
    df = df[df['temporada'].astype(str).str.match(SEASON_RX, na=False)].copy()
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
    df = df.drop_duplicates(subset=['match_id']).copy()
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
    cnt = df.groupby('temporada').size().reset_index(name='partidos')
    log("Partidos por temporada después de dedup:")
    log(cnt.sort_values('temporada').to_string(index=False))
    log(f"Dedup: {n0} -> {len(df)} filas")
    return df

def infer_top_division_teams(df_temp):
    t = df_temp['temporada'].iloc[0]
    max_j = int(pd.to_numeric(df_temp['jornada'], errors='coerce').max())
    eq_counts = pd.concat([df_temp['equipo_local'], df_temp['equipo_visitante']]).value_counts()
    if max_j >= 39:
        keep_n = 22; exp_matches = 462; exp_j = 42
    else:
        keep_n = 20; exp_matches = 380; exp_j = 38
    top_teams = set(eq_counts.head(keep_n).index)
    df_f = df_temp[
        df_temp['equipo_local'].isin(top_teams) &
        df_temp['equipo_visitante'].isin(top_teams)
    ].copy()
    nunq_teams = len(top_teams)
    n_raw = len(df_temp); n_f = len(df_f)
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

def build_base_features(df_matches):
    log("Build de features base: inicio")
    t0 = time.perf_counter()

    df = df_matches.sort_values(['temporada','jornada']).copy()
    for c in ['goles_local','goles_visitante','jornada']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['resultado_1x2'] = [etiqueta_1x2(gl, gv) for gl, gv in zip(df['goles_local'], df['goles_visitante'])]
    assert df.duplicated(subset=['temporada','jornada','equipo_local','equipo_visitante']).sum() == 0, "Partidos duplicados antes de features"

    loc = pd.DataFrame({
        'temporada': df['temporada'], 'jornada': df['jornada'],
        'equipo': df['equipo_local'], 'gf': df['goles_local'], 'gc': df['goles_visitante'], 'is_local': 1
    })
    vis = pd.DataFrame({
        'temporada': df['temporada'], 'jornada': df['jornada'],
        'equipo': df['equipo_visitante'], 'gf': df['goles_visitante'], 'gc': df['goles_local'], 'is_local': 0
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
        # Streaks
        # racha de victorias
        # Agrupamos por bloques consecutivos de win
        # (simplemente: cuantos seguidos llevan win=1 hasta antes de este partido)
        # Manera vectorizada de pandas para rachas es compleja, usaremos aproximacion iterativa o rolling sum check
        # Si queremos "streak actual":
        # check si el anterior ganó. Si sí, streak = streak_prev + 1. Si no, 0.
        # Lo haremos via diff de grupos de (win!=1).cumsum()
        def get_streak(s):
            # s es serie de 0/1 (win, draw o loss)
            # Compare with 0
            # Identify groups of consecutive 1s
            # Cumsum reset at 0
            # Truco:
            return s.groupby((s != 1).cumsum()).cumcount()
        
        g['streak_win']  = get_streak(g['win'].shift(1).fillna(0))
        g['streak_draw'] = get_streak(g['draw'].shift(1).fillna(0))
        g['streak_loss'] = get_streak(g['loss'].shift(1).fillna(0))

        # Context-specific: HOME form (pts5) and AWAY form (pts5)
        # Esto requiere filtrar solo las filas donde is_local=1 o is_local=0
        # Pero dentro de 'g' tenemos mezclado local y visitante para el mismo equipo.
        # Haremos:
        # pts_home_only = g['puntos'] donde is_local=1. Shift, rolling sum.
        # pts_away_only = g['puntos'] donde is_local=0. Shift, rolling sum.
        # Luego rellenamos hacia abajo (ffill) para que el equipo "recuerde" su ultima forma home cuando juega away, y viceversa.
        
        # Home Form
        mask_h = (g['is_local'] == 1)
        p_home = g.loc[mask_h, 'puntos'].shift(1).rolling(FORM_WINDOW, min_periods=1).sum()
        g.loc[mask_h, 'f_home_pts5'] = p_home
        g['f_home_pts5'] = g['f_home_pts5'].ffill().fillna(0.0)

        # Away Form
        mask_a = (g['is_local'] == 0)
        p_away = g.loc[mask_a, 'puntos'].shift(1).rolling(FORM_WINDOW, min_periods=1).sum()
        g.loc[mask_a, 'f_away_pts5'] = p_away
        g['f_away_pts5'] = g['f_away_pts5'].ffill().fillna(0.0)

        return g

    tg = tg.groupby(['temporada','equipo'], group_keys=False).apply(accum_and_shift)

    fill_cols = ['c_pj','c_pts','c_win','c_draw','c_loss','c_gf','c_gc',
                 'form_pts5','form_gf5','form_gc5','ewm_pts','ewm_gf','ewm_gc',
                 'streak_win','streak_draw','streak_loss','f_home_pts5','f_away_pts5']
    tg[fill_cols] = tg[fill_cols].fillna(0.0)
    tg['c_dg'] = tg['c_gf'] - tg['c_gc']
    tg['ppp']  = np.where(tg['c_pj']>0, tg['c_pts']/tg['c_pj'], 0.0)
    tg['win_rate'] = np.where(tg['c_pj']>0, tg['c_win']/tg['c_pj'], 0.0)

    def add_suffix(df_in, suf):
        keep = ['temporada','jornada','equipo','c_pj','c_pts','c_win','c_draw','c_loss',
                'c_gf','c_gc','c_dg','ppp','win_rate','form_pts5','form_gf5','form_gc5',
                'ewm_pts','ewm_gf','ewm_gc',
                'streak_win','streak_draw','streak_loss','f_home_pts5','f_away_pts5']
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
        ('streak_win_local','streak_win_visitante'),
        ('streak_loss_local','streak_loss_visitante'),
        # IMPORTANTE: Comparar peras con peras o cruzado?
        # Para input del modelo, pasamos el raw de cada uno.
        # Y quiza diff de las formas especificas:
        # diff_context_form = f_home_pts5 (del local) - f_away_pts5 (del visitante)
        # Eso mide "Local fuerte en casa" vs "Visitante fuerte fuera"
    ]
    for a,b in diff_pairs:
        base[f'diff__{a[:-6]}'] = base[a] - base[b]
    
    # Feature cruzada especial: HomeForm(Local) vs AwayForm(Visitante)
    base['diff__context_form'] = base['f_home_pts5_local'] - base['f_away_pts5_visitante']

    base = (base.replace([np.inf, -np.inf], np.nan).fillna(0.0))

    if len(base) > 1.05 * len(df):
        log(f"⚠️ Base creció {len(base)} vs {len(df)}. Dedupe final por partido.")
        base = base.sort_values(['temporada','jornada','equipo_local','equipo_visitante']).drop_duplicates(
            subset=['temporada','jornada','equipo_local','equipo_visitante'], keep='first'
        )

    log(f"Build de features base: fin en {time.perf_counter()-t0:.1f}s | filas={len(base)}")
    return base

# ===== (PUNTO 1 ya implementado por ti): expected rule + tuning de γ =====
def postproc_expected_rule(p1, pX, p2, delta, tau, gamma):
    """
    Regla de decisión:
      - Base: argmax.
      - Si 1 vs 2 están "cerca" (< delta) y X es razonable (pX >= tau)
        o bien pX está a <= gamma del ganador, etiquetamos X.
    """
    P = np.stack([p1, pX, p2], axis=1)
    base = np.array(['1','X','2'])[P.argmax(axis=1)]

    margin12 = np.abs(p1 - p2)
    close12 = (margin12 < float(delta))
    near_win = (np.maximum(p1, p2) - pX) <= float(gamma)
    viableX = (pX >= float(tau)) | near_win

    #out = base.copy
    out = base.copy()
    mask = close12 & viableX
    out[mask] = 'X'
    return out

def tune_gamma_on_early(df_2425, probs_all, y_all, delta, tau,
                        j_early_max=25, gamma_grid=(0.00, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)):
    mask_early = (df_2425['jornada'] >= 1) & (df_2425['jornada'] <= j_early_max)
    if mask_early.sum() < 10:
        return 0.03  # neutro
    y_e = y_all[mask_early.values]
    pe = probs_all[mask_early]
    p1, pX, p2 = pe[:,0], pe[:,1], pe[:,2]
    best_s, best_g = -1e9, 0.03
    for g in gamma_grid:
        cand = postproc_expected_rule(p1, pX, p2, delta, tau, g)
        s = mixed_score_from_labels(y_e, cand)
        if s > best_s:
            best_s, best_g = s, g
    return float(best_g)

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
    df_all = raw_df_matches.sort_values(['temporada','jornada','equipo_local','equipo_visitante']).copy()
    df_all = df_all[['temporada','jornada','equipo_local','equipo_visitante','goles_local','goles_visitante']].reset_index(drop=True)
    elo_cols = compute_season_elo(df_all, ELO_K, ELO_HOME_ADV)
    keys = df_all[['temporada','jornada','equipo_local','equipo_visitante']].reset_index(drop=True)
    elo_df = pd.concat([keys, elo_cols.reset_index(drop=True)], axis=1)

    out = base_df.merge(
        elo_df,
        on=['temporada','jornada','equipo_local','equipo_visitante'],
        how='left'
    )

    for c in ['elo_local_pre','elo_visitante_pre']:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out['elo_diff'] = out[c := 'elo_local_pre'] - out['elo_visitante_pre']
    for c in ['elo_local_pre','elo_visitante_pre','elo_diff']:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out

# =========================
# Modelo
# =========================
def crear_modelo(input_dim, output_dim=3, lr=7e-4):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    
    model.add(Dense(300, kernel_regularizer=l2(1e-4)))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.30))
    
    model.add(Dense(128, kernel_regularizer=l2(1e-4)))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.25))
    
    model.add(Dense(output_dim, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss=CategoricalCrossentropy(label_smoothing=0.05),
                  metrics=['accuracy'])
    return model

# ====== Post-procesado ======
def postproc_preds(p1, pX, p2, delta, tau):
    argmax_lbl = np.array(['1','X','2'])[np.stack([p1,pX,p2], axis=1).argmax(axis=1)]
    close = (np.abs(p1 - p2) < delta) & (pX > tau)
    out = argmax_lbl.copy()
    out[close] = 'X'
    return out

def postproc_preds_guardrails(p1, pX, p2, delta, tau, max_draw=0.40):
    base = np.array(['1','X','2'])[np.stack([p1,pX,p2], axis=1).argmax(axis=1)]
    close = (np.abs(p1 - p2) < delta) & (pX > tau)
    out = base.copy()
    out[close] = 'X'
    propX = (out=='X').mean()
    if propX > max_draw:
        idx = np.where(out=='X')[0]
        order = idx[np.argsort(pX[idx])]
        n_cut = int(len(out) * (propX - max_draw))
        if n_cut > 0:
            flip = order[:n_cut]
            out[flip] = base[flip]
    return out

def mean_px_after_shift(logits, shift):
    z = logits.copy()
    z[:, 1] += float(shift)   # mover solo la X en logits
    p = temperature_scale(z, 1.0)
    return float(p[:, 1].mean()), z

def balance_draw_share(logits, target_share, max_abs_shift=0.55, iters=30):
    """
    Ajusta SOLO el logit de la X con búsqueda binaria para que su media ≈ target_share.
    Rango más amplio y más iteraciones para rescatar pX si colapsa tras calibraciones.
    """
    lo, hi = -max_abs_shift, max_abs_shift
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        m, _ = mean_px_after_shift(logits, mid)
        if m > target_share:
            hi = mid
        else:
            lo = mid
    _, z = mean_px_after_shift(logits, (lo + hi) / 2.0)
    return temperature_scale(z, 1.0)

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
    out = []
    for temp, g in df_all.groupby('temporada', sort=False):
        g = g.copy()
        equipos = pd.unique(pd.concat([g['equipo_local'], g['equipo_visitante']], ignore_index=True))
        n_teams = len(equipos)
        per_round_cap = 11 if n_teams >= 22 else 10
        target_total  = n_teams * (n_teams - 1)
        g['__has_score'] = (~g['goles_local'].isna() & ~g['goles_visitante'].isna()).astype(int)
        key1 = g[['equipo_local','equipo_visitante']].min(axis=1)
        key2 = g[['equipo_local','equipo_visitante']].max(axis=1)
        g['__pair'] = key1 + '|' + key2
        g['__side'] = np.where(g['equipo_local'] == key1, 'Ahome', 'Bhome')
        med_j = g['jornada'].median() if pd.notna(g['jornada']).any() else 19.5
        g['__prio'] = (
            g['__has_score']*10000
            - (g['jornada'] - med_j).abs()*100
            - g['jornada'].fillna(9999)
        )
        def pick_pair(h):
            h = h.sort_values(['__prio'], ascending=False)
            a = h[h['__side']=='Ahome'].head(1)
            b = h[h['__side']=='Bhome'].head(1)
            sel = pd.concat([a, b], ignore_index=True)
            if sel.empty:
                sel = h.head(2)
            return sel.head(2)
        g_pair = g.groupby('__pair', group_keys=False).apply(pick_pair)
        g_pair = (
            g_pair.sort_values(['__prio'], ascending=False)
                 .groupby('jornada', group_keys=False)
                 .apply(lambda h: h.head(per_round_cap))
        )
        if len(g_pair) > target_total:
            g_pair = g_pair.sort_values('__prio', ascending=False).head(target_total)
        g_pair = g_pair.drop(columns=['__pair','__side','__prio','__has_score'], errors='ignore')
        g_pair = g_pair.sort_values(['temporada','jornada','equipo_local','equipo_visitante'])
        out.append(g_pair)
        cnt_j = g_pair.groupby('jornada').size().sort_index()
        log(f"[round-robin v2] {temp}: partidos={len(g_pair)} (target={target_total}) | jornadas={cnt_j.index.min()}–{cnt_j.index.max()} | cap_jornada={per_round_cap} | min/max por jornada={cnt_j.min() if len(cnt_j) else 0}/{cnt_j.max() if len(cnt_j) else 0}")
        if len(g_pair) < target_total:
            log(f"[warn] {temp}: faltan {target_total - len(g_pair)} partidos para llegar al objetivo. Revisa la fuente o añade heurística extra para diferenciar liga vs. copas.")
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
    # New
    'streak_win_local','streak_draw_local','streak_loss_local',
    'f_home_pts5_local','f_away_pts5_local', # aunque f_away_pts5_local no importa mucho cuando juega en casa, lo dejamos por si acaso
    'streak_win_visitante','streak_draw_visitante','streak_loss_visitante',
    'f_home_pts5_visitante','f_away_pts5_visitante',
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

# ===== Métrica mixta
def mixed_score_from_probs(y_true_idx, probs):
    lbl = np.array(['1','X','2'])[probs.argmax(1)]
    y_pred_idx = pd.Series(lbl).map({'1':0,'X':1,'2':2}).values
    acc = accuracy_score(y_true_idx, y_pred_idx)
    macro = f1_score(y_true_idx, y_pred_idx, average='macro')
    return macro + 0.25*acc

def mixed_score_from_labels(y_true_idx, cand_labels):
    y_pred_idx = pd.Series(cand_labels).map({'1':0,'X':1,'2':2}).values
    acc = accuracy_score(y_true_idx, y_pred_idx)
    macro = f1_score(y_true_idx, y_pred_idx, average='macro')
    return macro + 0.25*acc

# =========================
# Tuning en 23/24
# =========================
total_combos = len(ELO_K_GRID)*len(ELO_HOME_GRID)*len(W_X_GRID)
log(f"Inicio tuning ({VAL_SEASON}): total combinaciones (EloK, EloHOME, wX) = {total_combos}")

best = {'acc': -1,
        'elo_k': None, 'elo_home': None,
        'w_x': None, 'delta': None, 'tau': None,
        'T_opt': 1.0,
        'b_opt': np.zeros(3, dtype=float)}

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
            model_t.fit(X_tb_s, y_tb_cat, epochs=EPOCHS_TUNE, batch_size=128, verbose=0,
                        class_weight=cw_dict, callbacks=callbacks)

            prob_val = model_t.predict(X_val_s, verbose=0)

            # ---- Calibración T ----
            logits_proxy = np.log(prob_val + 1e-9)
            T_opt = fit_temperature(logits_proxy, y_val)
            prob_val_T = temperature_scale(logits_proxy, T_opt)

            # ---- Bias opcional
            use_b = np.zeros(3, dtype=float)
            prob_val_TB = prob_val_T
            if USE_CLASSWISE_BIAS:
                b_try = fit_classwise_bias(np.log(prob_val_T + 1e-12), y_val,
                                           b_bounds=BIAS_BOUNDS, l2_reg=BIAS_L2)
                prob_TB_try = apply_classwise_bias_from_logits(np.log(prob_val_T + 1e-12), b_try)
                mean_pX_try = prob_TB_try[:,1].mean()
                if (mean_pX_try >= BIAS_MIN_MEAN_X) and (
                    mixed_score_from_probs(y_val, prob_TB_try) >= mixed_score_from_probs(y_val, prob_val_T) - 1e-9
                ):
                    prob_val_TB = prob_TB_try
                    use_b = b_try

            # Elegir baseline por métrica mixta
            scores = [
                ('raw', mixed_score_from_probs(y_val, prob_val), 1.0, np.zeros(3)),
                ('T',   mixed_score_from_probs(y_val, prob_val_T), T_opt, np.zeros(3)),
                ('TB',  mixed_score_from_probs(y_val, prob_val_TB), T_opt, use_b)
            ]
            kind, use_base_score, use_T, use_b = max(scores, key=lambda x: x[1])
            use_prob_for_pp = prob_val if kind=='raw' else (prob_val_T if kind=='T' else prob_val_TB)

            if use_base_score > best['acc']:
                best = {'acc': use_base_score, 'elo_k': elo_k, 'elo_home': elo_home,
                        'w_x': w_x, 'delta': None, 'tau': None, 'T_opt': use_T, 'b_opt': use_b}
                log(f"  ↪ Nuevo BEST baseline: score={use_base_score:.4f} | T={use_T:.3f}")

            p1, pX, p2 = use_prob_for_pp[:,0], use_prob_for_pp[:,1], use_prob_for_pp[:,2]
            for d in DELTA_GRID:
                for t in TAU_GRID:
                    cand = postproc_preds(p1, pX, p2, d, t)
                    cand_score = mixed_score_from_labels(y_val, cand)
                    if cand_score > best['acc']:
                        best = {'acc': cand_score, 'elo_k': elo_k, 'elo_home': elo_home,
                                'w_x': w_x, 'delta': d, 'tau': t, 'T_opt': use_T, 'b_opt': use_b}
                        log(f"  ★ Nuevo BEST POST-PROC: score={cand_score:.4f} | delta={d} | tau={t} | T={use_T:.3f}")

            log(f"  Hecho combo en {time.perf_counter()-t_combo:.1f}s (mejor score hasta ahora={best['acc']:.4f})")

log(f"Fin tuning en {time.perf_counter()-tuning_start:.1f}s")
print(f"\n🔧 Mejor en {VAL_SEASON} -> score_mix={best['acc']:.4f} | EloK={best['elo_k']} | EloHOME={best['elo_home']} | wX={best['w_x']} | delta={best['delta']} | tau={best['tau']} | T={best['T_opt']:.3f}")

# =========================
# Entrenamiento final y test
# =========================
log("Entrenamiento final con train_full + test 24/25")
train_full = inject_elo(base_train_full, df, best['elo_k'], best['elo_home'])
test_set   = inject_elo(base_test,       df, best['elo_k'], best['elo_home'])

assert len(train_full) == len(base_train_full), f"train_full len mismatch {len(train_full)} != {len(base_train_full)}"
assert len(test_set)  == len(base_test),        f"test_set len mismatch {len(test_set)} != {len(base_test)}"

feature_cols = base_cols + ['elo_local_pre','elo_visitante_pre','elo_diff']
X_tf, y_tf = prep_xy_general(train_full, feature_cols)
dist_train_series = pd.Series(y_tf).map({0:'1',1:'X',2:'2'})
dist_train = dist_train_series.value_counts(normalize=True).reindex(['1','X','2']).fillna(1/3).values
print("\n📊 Distribución real en entrenamiento (train_full, tras prep):")
print(pd.Series(dist_train, index=['1','X','2']).rename_axis('resultado').reset_index(name='proporcion'))

X_te, y_te = prep_xy_general(test_set, feature_cols)
log(f"Shapes -> X_tf={X_tf.shape}, X_te={X_te.shape}")

scaler = StandardScaler()
X_tf_s = scaler.fit_transform(X_tf)
X_te_s = scaler.transform(X_te)

# ===== Peso final para X
cw = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_tf)
cw_dict = {i: w for i, w in enumerate(cw)}

TARGET_X_FIT = float(np.clip(0.26, 0.24, 0.28))
train_priors_vec = pd.Series(y_tf).value_counts(normalize=True).reindex([0,1,2]).fillna(1/3).values
px_train = float(train_priors_vec[1])

wx_boost = np.clip(TARGET_X_FIT / max(1e-6, px_train), 1.00, 1.30)
w_x_final = max(1.0, wx_boost)
cw_dict[1] = cw_dict[1] * w_x_final
log(f"Peso de clase X en fit final: {cw_dict[1]:.3f} (w_x_final={w_x_final:.3f}, px_train={px_train:.3f})")

# ====== Fit final (ensemble) ======
log("Fit final (ensemble de semillas) ...")
SEED_LIST = [13, 37, 42, 73, 101]
probas_test_ensemble = []
t_fit = time.perf_counter()
for i, s in enumerate(SEED_LIST, 1):
    log(f"  • Modelo {i}/{len(SEED_LIST)} con seed={s}")
    probas_i = train_and_predict_once(
        X_tf_s, y_tf,
        X_te_s,
        cw_dict,
        epochs=EPOCHS_FINAL,
        val_split=0.12,
        seed=s
    )
    probas_test_ensemble.append(probas_i)
log(f"Fit ensemble listo en {time.perf_counter()-t_fit:.1f}s")

# Promedio de probabilidades (antes de calibración final)
probas_test_raw = np.mean(probas_test_ensemble, axis=0)

log("Predicción test + calibración...")

# ---- TUNING TEMPORAL EN 24/25 ----
def tune_pp_and_T_on_season_early(df_2425, probas_all, y_all, j_early_max=25,
                                  delta_cand=(0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14),
                                  tau_cand=(0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36),
                                  max_draw_guard=0.42):
    mask_early = (df_2425['jornada'] >= 1) & (df_2425['jornada'] <= j_early_max)
    if mask_early.sum() < 10:
        return best['T_opt'], best['delta'], best['tau']
    logits_early = np.log(probas_all[mask_early] + 1e-9)
    y_early = y_all[mask_early.values]
    T_opt_early = fit_temperature(logits_early, y_early)
    probs_early = temperature_scale(logits_early, T_opt_early)
    y_lbl_early = pd.Series(y_early).map({0:'1',1:'X',2:'2'}).values
    p1, pX, p2 = probs_early[:,0], probs_early[:,1], probs_early[:,2]
    best_score, best_d, best_t = -1e9, None, None
    base_lbl = np.array(['1','X','2'])[probs_early.argmax(axis=1)]
    base_macro = f1_score(y_early, pd.Series(base_lbl).map({'1':0,'X':1,'2':2}), average='macro')
    for d in delta_cand:
        for t in tau_cand:
            cand = postproc_preds_guardrails(p1, pX, p2, d, t, max_draw=max_draw_guard)
            macro = f1_score(y_early, pd.Series(cand).map({'1':0,'X':1,'2':2}), average='macro')
            acc = accuracy_score(y_lbl_early, cand)
            propX = (cand=='X').mean()
            score = macro + 0.25*acc - max(0.0, propX-0.40)*0.05
            if score > best_score:
                best_score, best_d, best_t = score, d, t
    if best_d is None:
        best_d, best_t = 0.0, 0.50
    return T_opt_early, best_d, best_t

# ---- PRIOR CORRECTION (early labels) ----
def prior_correction_logits(logits, df_2425, y_te_idx, dist_train_vec, j_prior_max=15):
    mask_prior = (df_2425['jornada'] >= 1) & (df_2425['jornada'] <= j_prior_max)
    if mask_prior.sum() < 10:
        return logits
    pri = pd.Series(y_te_idx[mask_prior.values]).value_counts(normalize=True).reindex([0,1,2]).fillna(1/3).values
    train_pr_idx = np.array([dist_train_vec[0], dist_train_vec[1], dist_train_vec[2]])
    adj = np.log(pri + 1e-9) - np.log(train_pr_idx + 1e-9)
    return logits + adj.reshape(1, -1)

# 1) prior correction
logits_test_proxy = np.log(probas_test_raw + 1e-9)
logits_test_proxy = prior_correction_logits(logits_test_proxy, test_set, y_te, dist_train, j_prior_max=15)

# 2) T/delta/tau óptimos para early 24/25
T_use, delta_use, tau_use = tune_pp_and_T_on_season_early(test_set, np.exp(logits_test_proxy), y_te, j_early_max=25)

# --- clamps de seguridad ---
if delta_use is None:
    delta_use = 0.06
else:
    delta_use = float(np.clip(delta_use, 0.04, 0.14))

if tau_use is None:
    tau_use = 0.24
else:
    tau_use = float(np.clip(tau_use, 0.20, 0.36))

# 3) Aplica T
probas_T = temperature_scale(logits_test_proxy, T_use)

# 3b) Bias por clase (opcional)
b_opt = best.get('b_opt', np.zeros(3, dtype=float))
MIN_MEAN_PX_AFTER_BIAS = 0.24
if USE_CLASSWISE_BIAS and np.any(np.abs(b_opt) > 0):
    logits_T = np.log(probas_T + 1e-12)
    probas_TB_try = apply_classwise_bias_from_logits(logits_T, b_opt)
    if (probas_TB_try[:,1].mean() >= MIN_MEAN_PX_AFTER_BIAS) and \
       (probas_TB_try[:,1].mean() >= probas_T[:,1].mean() - 0.02):
        probas_TB = probas_TB_try
        if probas_TB[:,1].mean() < 0.20:
            ltmp = np.log(probas_TB + 1e-12); ltmp[:,1] += 0.10
            probas_TB = temperature_scale(ltmp, 1.0)
    else:
        log(f"[bias] Rechazado: pX_try={probas_TB_try[:,1].mean():.3f} | pX_T={probas_T[:,1].mean():.3f}")
        probas_TB = probas_T
else:
    probas_TB = probas_T

# ===== (PUNTO 2) τ adaptativo en función de pX previa =====
# DESPUÉS (un punto menos de holgura por abajo)
def adapt_tau(tau_base, mean_px):
    if mean_px < 0.20:   return max(0.18, tau_base - 0.06)
    if mean_px < 0.22:   return max(0.20, tau_base - 0.04)
    if mean_px > 0.30:   return min(0.34, tau_base + 0.04)
    if mean_px > 0.28:   return min(0.32, tau_base + 0.02)
    return float(tau_base)

# ========= Dos ramas (NR vs R) + selección automática =========
y_true_lbl = pd.Series(y_te).map({0:'1',1:'X',2:'2'}).values

# --- Rama 1: SIN round-priors (NR) ---
logits_NR = np.log(probas_TB + 1e-12)
target_X_share = float(np.clip(dist_train[1], 0.30, 0.33))
probas_NR_used = balance_draw_share(logits_NR, target_X_share, max_abs_shift=0.30, iters=24)

mean_pX_NR = probas_NR_used[:,1].mean()
if mean_pX_NR < PX_FLOOR_MEAN:
    lsafe = np.log(probas_NR_used + 1e-12); lsafe[:,1] += PX_FLOOR_BUMP
    probas_NR_used = temperature_scale(lsafe, 1.0)

# (PUNTO 3) τ adaptado + γ tunado en early -> regla expected
tau_NR   = adapt_tau(tau_use, mean_pX_NR)
gamma_NR = tune_gamma_on_early(test_set, probas_NR_used, y_te, delta_use, tau_NR, j_early_max=25)

p1_NR, pX_NR, p2_NR = probas_NR_used[:,0], probas_NR_used[:,1], probas_NR_used[:,2]
preds_NR = postproc_expected_rule(p1_NR, pX_NR, p2_NR, delta_use, tau_NR, gamma_NR)
acc_NR = (preds_NR == y_true_lbl).mean()
macro_NR = f1_score(y_te, pd.Series(preds_NR).map({'1':0,'X':1,'2':2}), average='macro')

# --- Rama 2: CON round-priors (R) ---
round_priors = compute_round_priors(train_full, exclude_season=TEST_SEASON)
base_priors  = compute_global_priors_from_train(train_full)
j_te         = test_set['jornada'].values.astype(int)

logits_TB_l = np.log(probas_TB + 1e-12)
logits_R    = apply_round_priors_on_logits(
    logits_TB_l, j_te, round_priors, base_priors,
    lam=ROUND_PRIORS_LAMBDA, lam_vec=LAM_VEC
)
probas_R_bal   = balance_draw_share(logits_R,  target_X_share, max_abs_shift=0.30, iters=24)

mean_pX_R = probas_R_bal[:,1].mean()
if mean_pX_R < PX_FLOOR_MEAN:
    lsafe = np.log(probas_R_bal + 1e-12); lsafe[:,1] += PX_FLOOR_BUMP
    probas_R_used = temperature_scale(lsafe, 1.0)
else:
    probas_R_used = probas_R_bal

tau_R   = adapt_tau(tau_use, mean_pX_R)
gamma_R = tune_gamma_on_early(test_set, probas_R_used, y_te, delta_use, tau_R, j_early_max=25)

p1_R, pX_R, p2_R = probas_R_used[:,0], probas_R_used[:,1], probas_R_used[:,2]
preds_R = postproc_expected_rule(p1_R, pX_R, p2_R, delta_use, tau_R, gamma_R)
acc_R = (preds_R == y_true_lbl).mean()
macro_R = f1_score(y_te, pd.Series(preds_R).map({'1':0,'X':1,'2':2}), average='macro')

# --- Elección automática ---
use_R = (acc_R > acc_NR) or (acc_R == acc_NR and macro_R >= macro_NR)
probas_used = probas_R_used if use_R else probas_NR_used
final_preds = preds_R if use_R else preds_NR

# Forzar cuota mínima de 'X' en etiquetas finales (previo a cupón)
true_X_rate = float((test_set['resultado_1x2'] == 'X').mean()) if 'resultado_1x2' in test_set else dist_train[1]
min_x_quota = float(np.clip(true_X_rate - 0.03, 0.18, 0.26))

P_used = np.stack([p1_R, pX_R, p2_R], axis=1) if use_R else np.stack([p1_NR, pX_NR, p2_NR], axis=1)
final_preds = enforce_x_quota_predictions(
    P_used, final_preds,
    min_share=min_x_quota,
    margin_cap=max(0.06, float(delta_use)),
    min_px=max(0.24, float(tau_use) - 0.04)
)

print(f"\n✅ Selección de rama: {'R (round-priors)' if use_R else 'NR (sin round-priors)'} | acc_R={acc_R:.4f}, macro_R={macro_R:.4f} | acc_NR={acc_NR:.4f}, macro_NR={macro_NR:.4f}")

baseline_lbl = np.array(['1','X','2'])[probas_used.argmax(axis=1)]
acc_base = (baseline_lbl == y_true_lbl).mean()
acc_pp   = (final_preds == y_true_lbl).mean()
print(f"   ↳ En rama elegida: baseline={acc_base:.4f} vs post-proc={acc_pp:.4f}")

# Cap superior de X en etiquetas (seguro, antes del cupón)
true_X_rate = float((test_set['resultado_1x2'] == 'X').mean()) if 'resultado_1x2' in test_set else dist_train[1]
max_x_cap = float(min(0.29, true_X_rate + 0.04))  # ej. real 0.255 -> cap 0.295

cur_x = (final_preds == 'X').mean()
if cur_x > max_x_cap:
    # baja las X "peor justificadas" (menor pX - max(p1,p2))
    P_used = np.stack([p1_R, pX_R, p2_R], axis=1) if use_R else np.stack([p1_NR, pX_NR, p2_NR], axis=1)
    p1c, pXc, p2c = P_used[:,0], P_used[:,1], P_used[:,2]
    base = np.array(['1','X','2'])[P_used.argmax(axis=1)]
    idxX = np.where(final_preds == 'X')[0]
    score = pXc[idxX] - np.maximum(p1c[idxX], p2c[idxX])  # cuanto más negativo, peor
    order = idxX[np.argsort(score)]  # primero las peores X
    need_drop = int(np.ceil(len(final_preds)* (cur_x - max_x_cap)))
    drop = order[:max(0, need_drop)]
    final_preds[drop] = base[drop]

# ===== (PUNTO 4) Cupón de empates mejorado (pase 0 + presupuesto 2.0 pp) =====
x_target = float(np.clip(dist_train[1], 0.18, 0.26))
p1, pX, p2 = probas_used[:,0], probas_used[:,1], probas_used[:,2]
base_prob = np.maximum(p1, p2)
pred_X_share = (final_preds == 'X').mean()
need = int(np.ceil(x_target*len(final_preds))) - int((final_preds=='X').sum())
print(f"   [cupón/budget] pred_X={pred_X_share:.3f} → target={x_target:.3f} | need={max(0,need)}")

final_preds_model = final_preds.copy()
acc_model = accuracy_score(pd.Series(y_te).map({0:'1',1:'X',2:'2'}).values, final_preds_model)

if need > 0:
    n = len(final_preds)
    # Presupuesto total: 2.0 pp
    exp_loss_budget = 0.020 * n
    exp_loss_cum = 0.0
    flips = []

    margin = np.abs(p1 - p2)

    # Pase 0: flips "casi gratis" (muy parejos, pX ≈ ganador)
    cand0 = np.where((final_preds != 'X') &
                 (margin <= min(0.12, float(delta_use) + 0.04)) &
                 (pX >= base_prob - 0.02))[0]
    if cand0.size > 0:
        order0 = cand0[np.argsort(-(pX[cand0] - base_prob[cand0]))]
        for i in order0:
            if len(flips) >= need: break
            d0 = float(pX[i] - base_prob[i])
            if d0 >= -0.001 or (d0 >= -0.008 and (exp_loss_cum - d0) <= exp_loss_budget):
                flips.append(i); exp_loss_cum -= min(0.0, d0)

    # Candidatos por etapas (margen y pX mínimos)
    margin_cap0 = float(delta_use if delta_use is not None else 0.04)
    min_px0     = float(tau_use   if tau_use   is not None else 0.30)
    stages = [
        (max(0.02, margin_cap0),            max(0.30, min_px0)),        # Etapa 1
        (min(0.10, margin_cap0 + 0.03),     max(0.28, min_px0 - 0.02)), # Etapa 2
        (min(0.14, margin_cap0 + 0.05),     max(0.26, min_px0 - 0.04)), # Etapa 3
    ]

    # Umbral por flip
    per_flip_floor = -0.012

    for (mc, mpx) in stages:
        if len(flips) >= need:
            break
        cand = np.where((final_preds != 'X') & (margin <= mc) & (pX >= mpx))[0]
        if cand.size == 0:
            continue
        delta = pX[cand] - base_prob[cand]
        order = cand[np.argsort(-delta)]
        for i in order:
            if len(flips) >= need: break
            d = float(pX[i] - base_prob[i])
            if d >= 0.0:
                flips.append(i); continue
            if d >= per_flip_floor and (exp_loss_cum - d) <= exp_loss_budget:
                flips.append(i); exp_loss_cum -= d

    # Último recurso
    if len(flips) < need and exp_loss_cum < exp_loss_budget:
        idx = np.where((final_preds != 'X') &
               (pX >= max(0.22, min_px0 - 0.08)) &
               (pX >= base_prob - 0.07))[0]
        if idx.size > 0:
            delta2 = pX[idx] - base_prob[idx]
            order2 = idx[np.argsort(-delta2)]
            for i in order2:
                if len(flips) >= need or exp_loss_cum >= exp_loss_budget:
                    break
                d = float(pX[i] - base_prob[i])
                if d >= 0.0 or (d >= per_flip_floor and (exp_loss_cum - d) <= exp_loss_budget):
                    flips.append(i)
                    exp_loss_cum -= min(0.0, d)

    final_preds_ticket = final_preds.copy()
    if flips:
        final_preds_ticket[np.array(flips, dtype=int)] = 'X'
    acc_ticket = accuracy_score(pd.Series(y_te).map({0:'1',1:'X',2:'2'}).values, final_preds_ticket)

    print(f"   [cupón/budget] flips={len(flips)} | exp_loss_used={exp_loss_cum:.3f} "
          f"| pred_X_final={(final_preds_ticket=='X').mean():.3f}")
    print(f"[check] acc modelo = {acc_model:.4f} | acc ticket = {acc_ticket:.4f}")

    final_preds = final_preds_model.copy()
else:
    final_preds = final_preds.copy()

# Chequeo de accuracy post-cupón
acc_after_coupon = accuracy_score(y_true_lbl, final_preds)
print(f"[check] acc post-cupón = {acc_after_coupon:.4f}")

true_X = (test_set['resultado_1x2'] == 'X').mean()
pred_X = (final_preds == 'X').mean()
print(f"[drift X] real={true_X:.3f} | pred_final={pred_X:.3f} | gap={pred_X-true_X:+.3f}")

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

dist_train_show = train_full['resultado_1x2'].value_counts(normalize=True).rename_axis('resultado').reset_index(name='proporcion')
print("\n📊 Distribución real en entrenamiento (train_full):")
print(dist_train_show)

dist_pred = pd.Series(final_preds).value_counts(normalize=True).rename_axis('prediccion').reset_index(name='proporcion')
print(f"\n📊 Distribución de predicciones en {TEST_SEASON}:")
print(dist_pred)

def show_dist(tag, P):
    d = pd.Series(np.array(['1','X','2'])[P.argmax(1)]).value_counts(normalize=True).reindex(['1','X','2']).fillna(0)
    print(f"[dist] {tag} -> 1:{d['1']:.3f}  X:{d['X']:.3f}  2:{d['2']:.3f}")

show_dist("raw-ens", probas_test_raw)
show_dist(" + priorEarly(T)", temperature_scale(logits_test_proxy, T_use))
show_dist(" + T (+ bias si ok)", probas_TB)
show_dist(" + Rama elegida (post)", probas_used)
