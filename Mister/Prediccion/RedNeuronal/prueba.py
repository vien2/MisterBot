# -*- coding: utf-8 -*-
"""
MisterBot 1X2 — Versión con 5 mejoras:
1) Eval honesta (calibra early, reporta late)
2) Vector Scaling (calibración por clase)
3) Meta-decisor para la X
4) Ensamble GBDT (CatBoost / LGBM / XGB) + NN
5) Features baratas (home-adv por equipo, gaps de jornada)

Requisitos: numpy, pandas, scikit-learn, tensorflow (igual que antes).
Opcionales: catboost, lightgbm o xgboost (si no, se omiten).
"""

import sys, os, time, re, random, warnings
from datetime import datetime
import numpy as np
import pandas as pd
from utils import conexion_db

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy

# Opcionales (GBDT). Se cargan si están disponibles.
HAVE_CATBOOST = False
HAVE_LGBM = False
HAVE_XGB = False
try:
    from catboost import CatBoostClassifier
    HAVE_CATBOOST = True
except Exception:
    pass
try:
    import lightgbm as lgb
    HAVE_LGBM = True
except Exception:
    pass
try:
    import xgboost as xgb
    HAVE_XGB = True
except Exception:
    pass

# === Configuración general ===
TEST_SEASON = '24/25'
VAL_SEASON  = '23/24'
FORM_WINDOW = 5
H2H_WINDOW  = 5
ELO_START   = 1500.0
ELO_CARRY   = 0.75
SEED = 42
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)
FAST_MODE = True

# Activadores de las 5 mejoras
USE_HONEST_EVAL = True
USE_VECTOR_SCALING = True
USE_META_X = True
USE_GBDT_ENSEMBLE = True  # se omite si no hay librerías
ADD_CHEAP_FEATURES = True

# === Utilidades básicas ===
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

SEASON_RX = re.compile(r"^\d{2}/\d{2}$")

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

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z); return ez / np.clip(ez.sum(axis=1, keepdims=True), 1e-12, None)

def set_all_seeds(seed):
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# =========================
# Conexión BD (adapta a tus utilidades)
# =========================


def q_matches(table_schema, temporada):
    return f"""
    SELECT
        jornada::integer AS jornada,
        equipo_local,
        equipo_visitante,
        goles_local,
        goles_visitante,
        temporada
    FROM {table_schema}.v_jornadas_liga
    WHERE temporada = '{temporada}'
    ORDER BY jornada ASC
    """

# ============== Limpieza + dedup (igual base, resumido) ==============
map_nombres = {
    "Ath Bilbao": "Athletic Club", "Ath Madrid": "Atlético", "Vallecano": "Rayo Vallecano",
    "Espanol": "Espanyol", "Villareal": "Villarreal", "Alaves": "Alavés", "La Coruna": "Deportivo",
    "Sp Gijon": "Sporting Gijón", "Sociedad": "Real Sociedad", "Cadiz": "Cádiz", "Almeria": "Almería",
}

def normalizar_nombres(df, cols):
    for c in cols:
        df[c] = df[c].astype(str).str.strip().replace(map_nombres)
    return df

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
    df = df.drop_duplicates(subset=['match_id']).copy()
    df = df.sort_values(
        ['temporada','jornada','equipo_local','equipo_visitante']
    ).drop_duplicates(
        subset=['temporada','jornada','equipo_local','equipo_visitante'], keep='first'
    )
    return df.drop(columns=['match_id'])

# ============== Filtro 1ª + round-robin (resumido de tu versión buena) ==============
def infer_top_division_teams(df_temp):
    max_j = int(pd.to_numeric(df_temp['jornada'], errors='coerce').max())
    eq_counts = pd.concat([df_temp['equipo_local'], df_temp['equipo_visitante']]).value_counts()
    if max_j >= 39: keep_n, exp_matches, exp_j = 22, 462, 42
    else:          keep_n, exp_matches, exp_j = 20, 380, 38
    top_teams = set(eq_counts.head(keep_n).index)
    df_f = df_temp[df_temp['equipo_local'].isin(top_teams) & df_temp['equipo_visitante'].isin(top_teams)].copy()
    return df_f, exp_matches, exp_j

def filter_primary_division(df_all):
    out = []
    for _, g in df_all.groupby('temporada', sort=False):
        g = g.copy()
        g['jornada'] = pd.to_numeric(g['jornada'], errors='coerce').astype('Int64')
        g = g.dropna(subset=['jornada']).copy()
        g['jornada'] = g['jornada'].astype(int)
        g_f, exp_matches, exp_j = infer_top_division_teams(g)
        if (len(g_f) >= 0.90*exp_matches) and (g_f['jornada'].nunique() >= int(0.90*exp_j)):
            out.append(g_f)
    if not out: return df_all.copy()
    return pd.concat(out, ignore_index=True).sort_values(['temporada','jornada'])

def enforce_round_robin(df_all):
    # (Usa el tuyo; aquí hacemos paso identidad por brevedad si ya está limpio)
    return df_all

# ============== Features base (tu pipeline + añadidos “baratos”) ==============
def add_cheap_features(base_df):
    """Añade:
       - home_adv_team: ventaja casa histórica por equipo (hasta jornada-1, por temporada y acumulado global)
       - gap_jornada_local/visitante: diferencia de jornadas respecto al último partido de ese equipo (proxy congestión)
    """
    df = base_df.copy().sort_values(['temporada','jornada'])

    # 1) Ventaja casa histórica por equipo (acumulada hasta el partido)
    # puntuación en casa vs fuera (rolling)
    # Construimos por equipo todas sus apariciones como local/visitante
    loc = df[['temporada','jornada','equipo_local','goles_local','goles_visitante']].rename(
        columns={'equipo_local':'equipo','goles_local':'gf','goles_visitante':'gc'}
    ); loc['is_home'] = 1
    vis = df[['temporada','jornada','equipo_visitante','goles_visitante','goles_local']].rename(
        columns={'equipo_visitante':'equipo','goles_visitante':'gf','goles_local':'gc'}
    ); vis['is_home'] = 0
    tg = pd.concat([loc, vis], ignore_index=True).sort_values(['equipo','temporada','jornada'])
    tg['pts'] = tg.apply(lambda r: 3 if r.gf>r.gc else (1 if r.gf==r.gc else 0), axis=1)

    def cum_home_adv(g):
        g = g.copy()
        g['cum_pts_home'] = (g['pts']*(g['is_home']==1)).cumsum().shift(1).fillna(0.0)
        g['cum_pts_away'] = (g['pts']*(g['is_home']==0)).cumsum().shift(1).fillna(0.0)
        g['cum_pj_home']  = ((g['is_home']==1).astype(int)).cumsum().shift(1).fillna(0.0)
        g['cum_pj_away']  = ((g['is_home']==0).astype(int)).cumsum().shift(1).fillna(0.0)
        # ventaja esperada por partido en casa vs fuera
        home_rate = np.where(g['cum_pj_home']>0, g['cum_pts_home']/g['cum_pj_home'], 0.0)
        away_rate = np.where(g['cum_pj_away']>0, g['cum_pts_away']/g['cum_pj_away'], 0.0)
        g['home_adv_team'] = home_rate - away_rate
        return g

    tg = tg.groupby(['equipo','temporada'], group_keys=False).apply(cum_home_adv)

    # gap de jornadas por equipo
    def gaps(g):
        g = g.copy()
        g['j_gap'] = g['jornada'].diff().shift(0).fillna(1.0)  # suele ser 1; si salta más, descanso largo
        return g
    tg = tg.groupby(['equipo','temporada'], group_keys=False).apply(gaps)

    # Reincorporamos
    loc_feats = tg[tg['is_home']==1][['temporada','jornada','equipo','home_adv_team','j_gap']].rename(
        columns={'equipo':'equipo_local','home_adv_team':'home_adv_team_local','j_gap':'gap_jornada_local'})
    vis_feats = tg[tg['is_home']==0][['temporada','jornada','equipo','home_adv_team','j_gap']].rename(
        columns={'equipo':'equipo_visitante','home_adv_team':'home_adv_team_visit','j_gap':'gap_jornada_visit'})
    out = (df
           .merge(loc_feats, on=['temporada','jornada','equipo_local'], how='left')
           .merge(vis_feats, on=['temporada','jornada','equipo_visitante'], how='left'))
    for c in ['home_adv_team_local','home_adv_team_visit','gap_jornada_local','gap_jornada_visit']:
        out[c] = out[c].fillna(0.0)
    out['home_adv_team_diff'] = out['home_adv_team_local'] - out['home_adv_team_visit']
    out['gap_jornada_diff']   = out['gap_jornada_local'] - out['gap_jornada_visit']
    return out

def build_base_features(df_matches):
    # (Tu versión condensada: acumulados + rolling + ewm + h2h)
    df = df_matches.sort_values(['temporada','jornada']).copy()
    for c in ['goles_local','goles_visitante','jornada']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['resultado_1x2'] = [etiqueta_1x2(gl, gv) for gl, gv in zip(df['goles_local'], df['goles_visitante'])]

    loc = pd.DataFrame({'temporada':df['temporada'],'jornada':df['jornada'],
                        'equipo':df['equipo_local'],'gf':df['goles_local'],'gc':df['goles_visitante'],'is_local':1})
    vis = pd.DataFrame({'temporada':df['temporada'],'jornada':df['jornada'],
                        'equipo':df['equipo_visitante'],'gf':df['goles_visitante'],'gc':df['goles_local'],'is_local':0})
    tg = pd.concat([loc,vis], ignore_index=True).sort_values(['temporada','equipo','jornada']).drop_duplicates(['temporada','equipo','jornada'])

    tg['puntos'] = [puntos_row(gf,gc) for gf,gc in zip(tg['gf'],tg['gc'])]
    tg['win']=(tg['gf']>tg['gc']).astype(float); tg['draw']=(tg['gf']==tg['gc']).astype(float)
    tg['loss']=(tg['gf']<tg['gc']).astype(float); tg['pj']=1.0

    def accum_and_shift(g):
        g=g.copy()
        g['c_pj']=g['pj'].cumsum().shift(1); g['c_pts']=g['puntos'].cumsum().shift(1)
        g['c_win']=g['win'].cumsum().shift(1); g['c_draw']=g['draw'].cumsum().shift(1); g['c_loss']=g['loss'].cumsum().shift(1)
        g['c_gf']=g['gf'].cumsum().shift(1); g['c_gc']=g['gc'].cumsum().shift(1)
        g['form_pts5']=g['puntos'].rolling(FORM_WINDOW,min_periods=1).sum().shift(1)
        g['form_gf5']=g['gf'].rolling(FORM_WINDOW,min_periods=1).sum().shift(1)
        g['form_gc5']=g['gc'].rolling(FORM_WINDOW,min_periods=1).sum().shift(1)
        g['ewm_pts']=g['puntos'].shift(1).ewm(alpha=0.3,min_periods=1).mean()
        g['ewm_gf']=g['gf'].shift(1).ewm(alpha=0.3,min_periods=1).mean()
        g['ewm_gc']=g['gc'].shift(1).ewm(alpha=0.3,min_periods=1).mean()
        return g

    tg = tg.groupby(['temporada','equipo'], group_keys=False).apply(accum_and_shift)
    fill_cols = ['c_pj','c_pts','c_win','c_draw','c_loss','c_gf','c_gc','form_pts5','form_gf5','form_gc5','ewm_pts','ewm_gf','ewm_gc']
    tg[fill_cols]=tg[fill_cols].fillna(0.0)
    tg['c_dg']=tg['c_gf']-tg['c_gc']; tg['ppp']=np.where(tg['c_pj']>0, tg['c_pts']/tg['c_pj'], 0.0)
    tg['win_rate']=np.where(tg['c_pj']>0, tg['c_win']/tg['c_pj'], 0.0)

    def add_suffix(df_in, suf):
        keep=['temporada','jornada','equipo','c_pj','c_pts','c_win','c_draw','c_loss','c_gf','c_gc','c_dg','ppp','win_rate',
              'form_pts5','form_gf5','form_gc5','ewm_pts','ewm_gf','ewm_gc']
        out=df_in[keep].copy(); ren={c:f"{c}_{suf}" for c in keep if c not in ['temporada','jornada','equipo']}; ren['equipo']=f"equipo_{suf}"
        return out.rename(columns=ren)

    stats_local = add_suffix(tg[tg['is_local']==1], 'local').drop_duplicates(['temporada','jornada','equipo_local'])
    stats_visit = add_suffix(tg[tg['is_local']==0], 'visitante').drop_duplicates(['temporada','jornada','equipo_visitante'])
    base = (df.merge(stats_local, on=['temporada','jornada','equipo_local'], how='left')
              .merge(stats_visit, on=['temporada','jornada','equipo_visitante'], how='left'))

    # H2H compacta (acumulada 5)
    def h2h_feats(df_in):
        h = []
        for _,r in df_in.iterrows():
            gl,gv=r['goles_local'],r['goles_visitante']
            pts_home = (3 if gl>gv else (1 if gl==gv else 0)) if (not pd.isna(gl) and not pd.isna(gv)) else np.nan
            h.append((r['temporada'],r['jornada'],r['equipo_local'],r['equipo_visitante'],gl,gv,pts_home))
        h=pd.DataFrame(h, columns=['temporada','jornada','home','away','gl','gv','pts_home']).sort_values(['temporada','jornada'])
        h=h.drop_duplicates(['temporada','jornada','home','away'])
        for col in ['h2h_pts5','h2h_gf5','h2h_gc5','h2h_home_pts5','h2h_home_gf5','h2h_home_gc5']: h[col]=0.0
        # par ordenado
        key1 = h['home'].where(h['home']<h['away'], h['away'])
        key2 = h['away'].where(h['home']<h['away'], h['home'])
        for _,g in h.groupby([key1,key2], sort=False):
            g=g.sort_values(['temporada','jornada'])
            pts_hist=[]; gf_hist=[]; gc_hist=[]
            pts5=[]; gf5=[]; gc5=[]
            for _,rr in g.iterrows():
                pts5.append(np.nansum(pts_hist[-H2H_WINDOW:]) if pts_hist else 0.0)
                gf5.append(np.nansum(gf_hist[-H2H_WINDOW:]) if gf_hist else 0.0)
                gc5.append(np.nansum(gc_hist[-H2H_WINDOW:]) if gc_hist else 0.0)
                pts_hist.append(rr['pts_home'] if not pd.isna(rr['pts_home']) else 0.0)
                gf_hist.append(rr['gl'] if not pd.isna(rr['gl']) else 0.0)
                gc_hist.append(rr['gv'] if not pd.isna(rr['gv']) else 0.0)
            h.loc[g.index, 'h2h_pts5']=pts5; h.loc[g.index, 'h2h_gf5']=gf5; h.loc[g.index,'h2h_gc5']=gc5
        # home-specific
        for (home,away),g in h.groupby(['home','away'], sort=False):
            g=g.sort_values(['temporada','jornada'])
            pts_hist=[]; gf_hist=[]; gc_hist=[]
            pts5=[]; gf5=[]; gc5=[]
            for _,rr in g.iterrows():
                pts5.append(np.nansum(pts_hist[-H2H_WINDOW:]) if pts_hist else 0.0)
                gf5.append(np.nansum(gf_hist[-H2H_WINDOW:]) if gf_hist else 0.0)
                gc5.append(np.nansum(gc_hist[-H2H_WINDOW:]) if gc_hist else 0.0)
                pts_hist.append(rr['pts_home'] if not pd.isna(rr['pts_home']) else 0.0)
                gf_hist.append(rr['gl'] if not pd.isna(rr['gl']) else 0.0)
                gc_hist.append(rr['gv'] if not pd.isna(rr['gv']) else 0.0)
            h.loc[g.index,'h2h_home_pts5']=pts5; h.loc[g.index,'h2h_home_gf5']=gf5; h.loc[g.index,'h2h_home_gc5']=gc5
        return df_in.merge(h[['temporada','jornada','home','away','h2h_pts5','h2h_gf5','h2h_gc5','h2h_home_pts5','h2h_home_gf5','h2h_home_gc5']],
                           left_on=['temporada','jornada','equipo_local','equipo_visitante'],
                           right_on=['temporada','jornada','home','away'], how='left').drop(columns=['home','away'])

    base = h2h_feats(base)
    # diffs
    diff_pairs = [
        ('c_pj_local','c_pj_visitante'),('c_pts_local','c_pts_visitante'),('c_win_local','c_win_visitante'),
        ('c_draw_local','c_draw_visitante'),('c_loss_local','c_loss_visitante'),
        ('c_gf_local','c_gf_visitante'),('c_gc_local','c_gc_visitante'),('c_dg_local','c_dg_visitante'),
        ('ppp_local','ppp_visitante'),('win_rate_local','win_rate_visitante'),
        ('form_pts5_local','form_pts5_visitante'),('form_gf5_local','form_gf5_visitante'),
        ('form_gc5_local','form_gc5_visitante'),
        ('ewm_pts_local','ewm_pts_visitante'),('ewm_gf_local','ewm_gf_visitante'),('ewm_gc_local','ewm_gc_visitante'),
    ]
    for a,b in diff_pairs: base[f'diff__{a[:-6]}'] = base[a] - base[b]
    base = base.replace([np.inf,-np.inf], np.nan).fillna(0.0)
    if ADD_CHEAP_FEATURES:
        base = add_cheap_features(base)
    return base

# ============== Elo con carry y diff ==============
def compute_season_elo(df_matches, ELO_K, ELO_HOME_ADV, carry=ELO_CARRY):
    df = df_matches.sort_values(['temporada', 'jornada']).copy()
    df['elo_local_pre']=np.nan; df['elo_visitante_pre']=np.nan
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
            home=df.at[i,'equipo_local']; away=df.at[i,'equipo_visitante']
            gl=df.at[i,'goles_local']; gv=df.at[i,'goles_visitante']
            h_elo=elo.get(home,ELO_START); a_elo=elo.get(away,ELO_START)
            df.at[i,'elo_local_pre']=h_elo; df.at[i,'elo_visitante_pre']=a_elo
            if pd.isna(gl) or pd.isna(gv): continue
            exp_home = 1.0/(1.0+10**(-(h_elo+ELO_HOME_ADV-a_elo)/400.0))
            s_home = 1.0 if gl>gv else (0.5 if gl==gv else 0.0)
            elo[home] = h_elo + ELO_K*(s_home-exp_home)
            elo[away] = a_elo + ELO_K*((1.0-s_home)-(1.0-exp_home))
        elo_prev_end.update(elo)
    return df[['elo_local_pre','elo_visitante_pre']]

def inject_elo(base_df, raw_df_matches, ELO_K, ELO_HOME_ADV):
    df_all = raw_df_matches.sort_values(['temporada','jornada','equipo_local','equipo_visitante']).copy()
    df_all = df_all[['temporada','jornada','equipo_local','equipo_visitante','goles_local','goles_visitante']].reset_index(drop=True)
    elo_cols = compute_season_elo(df_all, ELO_K, ELO_HOME_ADV)
    keys = df_all[['temporada','jornada','equipo_local','equipo_visitante']].reset_index(drop=True)
    elo_df = pd.concat([keys, elo_cols.reset_index(drop=True)], axis=1)
    out = base_df.merge(elo_df, on=['temporada','jornada','equipo_local','equipo_visitante'], how='left')
    for c in ['elo_local_pre','elo_visitante_pre']: out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0.0)
    out['elo_diff'] = out['elo_local_pre'] - out['elo_visitante_pre']
    return out

# ============== NN ==============
def crear_modelo(input_dim, output_dim=3, lr=7e-4):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(256, kernel_regularizer=l2(1e-4))); model.add(LeakyReLU(0.1)); model.add(Dropout(0.30))
    model.add(Dense(128, kernel_regularizer=l2(1e-4))); model.add(LeakyReLU(0.1)); model.add(Dropout(0.25))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=lr), loss=CategoricalCrossentropy(label_smoothing=0.05), metrics=['accuracy'])
    return model

def train_and_predict_once(X_tf_s, y_tf, X_te_s, cw_dict, epochs=120, val_split=0.12, seed=42):
    set_all_seeds(seed)
    y_tf_cat = to_categorical(y_tf, num_classes=3)
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=0),
                 EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)]
    model = crear_modelo(X_tf_s.shape[1])
    model.fit(X_tf_s, y_tf_cat, epochs=epochs, batch_size=256, verbose=0, class_weight=cw_dict, validation_split=val_split, callbacks=callbacks)
    return model.predict(X_te_s, verbose=0)

# ============== Vector Scaling (calibración) ==============
def fit_vector_scaling(logits, y_idx, l2=1e-2):
    K = logits.shape[1]
    x0 = np.zeros(2*K)  # [s0..sK-1, b0..bK-1]
    bounds = [(-3,3)]*K + [(-1.5,1.5)]*K
    from scipy.optimize import minimize
    def nll(theta):
        s = theta[:K]; b = theta[K:]
        z = logits * s.reshape(1,-1) + b.reshape(1,-1)
        p = softmax(z) + 1e-12
        ll = -np.log(p[np.arange(len(y_idx)), y_idx]).mean()
        reg = l2*(np.sum(s**2) + 0.25*np.sum(b**2))
        return ll + reg
    res = minimize(nll, x0, method='L-BFGS-B', bounds=bounds)
    s = res.x[:K]; b = res.x[K:]
    return s.astype(float), b.astype(float)

def apply_vector_scaling(logits, s, b):
    return softmax(logits * s.reshape(1,-1) + b.reshape(1,-1))

# ============== Meta-decisor X ==============
def build_meta_X_features(P, df):
    p1,pX,p2 = P[:,0],P[:,1],P[:,2]
    margin12 = np.abs(p1 - p2)
    cols = []
    def safe_col(name):
        return np.array(df[name].values, dtype=float) if name in df.columns else np.zeros(len(df), dtype=float)

    feats = np.column_stack([
        p1, pX, p2, margin12,
        np.abs(safe_col('elo_diff')),
        safe_col('diff__ppp'),
        safe_col('diff__win_rate'),
        safe_col('home_adv_team_diff'),
        safe_col('gap_jornada_diff'),
    ])
    return np.nan_to_num(feats, copy=False)

def tune_threshold_for_metaX(p_scores, base_labels, y_true_idx, grid=None):
    """Elige umbral que maximiza macro-F1 (o tu métrica mixta) sobre validación."""
    if grid is None: grid = np.linspace(0.2, 0.7, 26)
    best_thr, best_score = 0.5, -1e9
    for thr in grid:
        cand = base_labels.copy()
        flip_mask = (cand!='X') & (p_scores>=thr)
        cand[flip_mask] = 'X'
        y_pred_idx = pd.Series(cand).map({'1':0,'X':1,'2':2}).values
        macro = f1_score(y_true_idx, y_pred_idx, average='macro')
        acc   = accuracy_score(y_true_idx, y_pred_idx)
        score = macro + 0.25*acc
        if score > best_score:
            best_score, best_thr = score, float(thr)
    return best_thr

# ============== Lectura datos ==============
log("Leyendo datos...")
with conexion_db() as conn:
    temps_dbo = pd.read_sql("SELECT DISTINCT temporada FROM dbo.v_jornadas_liga", conn)['temporada'].tolist()
    temps_ch  = pd.read_sql("SELECT DISTINCT temporada FROM chavalitos.v_jornadas_liga", conn)['temporada'].tolist()
    dfs = []
    for t in temps_dbo: dfs.append(pd.read_sql(q_matches('dbo', t), conn))
    for t in temps_ch:  dfs.append(pd.read_sql(q_matches('chavalitos', t), conn))
df_raw = pd.concat(dfs, ignore_index=True)
df = clean_and_dedup(df_raw)
df = filter_primary_division(df)
df = enforce_round_robin(df)

# ============== Features base + split ==============
base = build_base_features(df)

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
    'h2h_pts5','h2h_gf5','h2h_gc5','h2h_home_pts5','h2h_home_gf5','h2h_home_gc5',
] + [c for c in base.columns if c.startswith('diff__')]

if ADD_CHEAP_FEATURES:
    base_cols += ['home_adv_team_local','home_adv_team_visit','home_adv_team_diff',
                  'gap_jornada_local','gap_jornada_visit','gap_jornada_diff']

mask_lbl = base['resultado_1x2'].isin(['1','X','2'])
base = base[mask_lbl].copy()

base_train_base = base[(base['temporada'] != VAL_SEASON) & (base['temporada'] != TEST_SEASON)].copy()
base_val        = base[base['temporada'] == VAL_SEASON].copy()
base_train_full = base[base['temporada'] != TEST_SEASON].copy()
base_test       = base[base['temporada'] == TEST_SEASON].copy()

# ============== Tuning (muy compacto) ==============
if FAST_MODE:
    ELO_K_GRID    = [22.0, 26.0]
    ELO_HOME_GRID = [60.0, 80.0]
    W_X_GRID      = [1.00, 1.20]
else:
    ELO_K_GRID    = [22.0, 24.0, 26.0]
    ELO_HOME_GRID = [60.0, 70.0, 80.0]
    W_X_GRID      = [1.00, 1.10, 1.20, 1.30]

best = {'score':-1e9, 'elo_k':None, 'elo_home':None, 'w_x':1.0}
def prep_xy(df_, cols):
    y = df_['resultado_1x2'].map({'1':0,'X':1,'2':2}).astype(int).values
    X = (df_[cols].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan).fillna(0.0).values.astype('float32'))
    return X, y

for elo_k in ELO_K_GRID:
    for elo_home in ELO_HOME_GRID:
        trb = inject_elo(base_train_base, df, elo_k, elo_home)
        val = inject_elo(base_val, df, elo_k, elo_home)
        fcols = base_cols + ['elo_local_pre','elo_visitante_pre','elo_diff']
        X_tb, y_tb = prep_xy(trb, fcols); X_val, y_val = prep_xy(val, fcols)
        ss = StandardScaler().fit(X_tb); X_tb_s = ss.transform(X_tb); X_val_s = ss.transform(X_val)
        cw = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_tb)
        cw = {i:w for i,w in enumerate(cw)}

        for w_x in W_X_GRID:
            cw_use = cw.copy(); cw_use[1] = cw_use[1]*w_x
            # pequeño NN rápido
            prob_val = train_and_predict_once(X_tb_s, y_tb, X_val_s, cw_use, epochs=60 if FAST_MODE else 120, val_split=0.12, seed=37)
            # calibración: VS u original
            logits = np.log(np.clip(prob_val, 1e-9, 1.0))
            if USE_VECTOR_SCALING:
                s_vs,b_vs = fit_vector_scaling(logits, y_val, l2=1e-2)
                prob_val_c = apply_vector_scaling(logits, s_vs, b_vs)
            else:
                prob_val_c = prob_val
            # score mixto
            lbl = np.array(['1','X','2'])[prob_val_c.argmax(1)]
            y_pred_idx = pd.Series(lbl).map({'1':0,'X':1,'2':2}).values
            macro = f1_score(y_val, y_pred_idx, average='macro'); acc = accuracy_score(y_val, y_pred_idx)
            score = macro + 0.25*acc
            if score > best['score']:
                best = {'score':score, 'elo_k':elo_k, 'elo_home':elo_home, 'w_x':w_x}
                log(f"Best so far -> score={score:.4f} | EloK={elo_k} Home={elo_home} wX={w_x}")

log(f"🔧 Mejor VAL {VAL_SEASON}: score_mix={best['score']:.4f} | EloK={best['elo_k']} | EloHOME={best['elo_home']} | wX={best['w_x']}")

# ============== Entrenamiento final + Test ==============
train_full = inject_elo(base_train_full, df, best['elo_k'], best['elo_home'])
test_set   = inject_elo(base_test,       df, best['elo_k'], best['elo_home'])
fcols = base_cols + ['elo_local_pre','elo_visitante_pre','elo_diff']
X_tf, y_tf = prep_xy(train_full, fcols)
X_te, y_te = prep_xy(test_set, fcols)

# scaler
ss = StandardScaler().fit(X_tf); X_tf_s = ss.transform(X_tf); X_te_s = ss.transform(X_te)

# pesos clases con boost de X hacia prior target ~0.26 (como tenías)
cw = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_tf)
cw = {i:w for i,w in enumerate(cw)}
train_priors = pd.Series(y_tf).value_counts(normalize=True).reindex([0,1,2]).fillna(1/3).values
px_train = float(train_priors[1])
TARGET_X_FIT = 0.26
w_x_final = float(np.clip(TARGET_X_FIT / max(px_train,1e-6), 1.0, 1.30))
cw[1] = cw[1]*w_x_final
log(f"Peso clase X final: {cw[1]:.3f} (w_x_final={w_x_final:.3f}, px_train={px_train:.3f})")

# Ensamble NN
SEED_LIST=[13,37,42,73,101]
probs_nn=[]
for i,s in enumerate(SEED_LIST,1):
    log(f"NN {i}/{len(SEED_LIST)} seed={s}")
    probs_nn.append(train_and_predict_once(X_tf_s, y_tf, X_te_s, cw, epochs=120, val_split=0.12, seed=s))
probs_nn = np.mean(probs_nn, axis=0)

# ===== GBDT opcional =====
def fit_predict_gbdt_block(name, Xtr, ytr, Xte):
    if name=='cat' and HAVE_CATBOOST:
        model = CatBoostClassifier(
            iterations=1000, depth=7, learning_rate=0.05, loss_function='MultiClass',
            random_seed=SEED, l2_leaf_reg=3.0, verbose=False
        )
        model.fit(Xtr, ytr)
        return model.predict_proba(Xte)
    if name=='lgbm' and HAVE_LGBM:
        params = dict(objective='multiclass', num_class=3, learning_rate=0.05, num_leaves=63,
                      min_data_in_leaf=30, feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                      verbosity=-1, seed=SEED)
        lgb_tr = lgb.Dataset(Xtr, label=ytr)
        model = lgb.train(params, lgb_tr, num_boost_round=1200)
        return model.predict(Xte)
    if name=='xgb' and HAVE_XGB:
        dtr = xgb.DMatrix(Xtr, label=ytr); dte = xgb.DMatrix(Xte)
        params = dict(objective='multi:softprob', num_class=3, eta=0.05, max_depth=8, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, seed=SEED)
        model = xgb.train(params, dtr, num_boost_round=1200)
        return model.predict(dte)
    return None

probs_blocks = [probs_nn]
gb_names = []
if USE_GBDT_ENSEMBLE:
    for name in ['cat','lgbm','xgb']:
        P = fit_predict_gbdt_block(name, X_tf, y_tf, X_te)
        if P is not None:
            probs_blocks.append(np.array(P))
            gb_names.append(name.upper())
            log(f"GBDT {name.upper()} añadido al ensamble")

# Blend simple (pesos afinados en val idealmente; aquí 0.6 para NN y resto equitativo)
if len(probs_blocks)==1:
    probs_blend = probs_blocks[0]
else:
    w_nn = 0.6
    w_others = (1.0 - w_nn) / (len(probs_blocks)-1)
    weights = [w_nn] + [w_others]*(len(probs_blocks)-1)
    probs_blend = np.zeros_like(probs_blocks[0])
    for w,P in zip(weights, probs_blocks):
        probs_blend += w*np.clip(P, 1e-9, 1.0)

# ===== Calibración con Vector Scaling (MEJORA #2)
logits_te = np.log(np.clip(probs_blend, 1e-9, 1.0))

# HONEST EVAL: calibramos VS usando SOLO early (j≤25) y reportamos métricas en late
j = test_set['jornada'].astype(int).values
mask_early = (j <= 25)
mask_late  = (j > 25)

if USE_HONEST_EVAL:
    s_vs, b_vs = fit_vector_scaling(logits_te[mask_early], y_te[mask_early], l2=1e-2) if USE_VECTOR_SCALING else (np.ones(3), np.zeros(3))
else:
    s_vs, b_vs = fit_vector_scaling(logits_te, y_te, l2=1e-2) if USE_VECTOR_SCALING else (np.ones(3), np.zeros(3))

probs_cal = apply_vector_scaling(logits_te, s_vs, b_vs) if USE_VECTOR_SCALING else probs_blend
probs_final_for_decision = probs_cal.copy()

# ===== Meta-decisor X (MEJORA #3)
base_labels = np.array(['1','X','2'])[probs_final_for_decision.argmax(1)]
if USE_META_X:
    # Entrenamos meta con VALIDACIÓN (23/24) y mismo calibrador VS aprendido en early test? => mejor: entrenar meta con val sin VS o con VS de val.
    # Para simplicidad: entrenamos meta en val con calibración VS propia de val.
    valX = inject_elo(base_val, df, best['elo_k'], best['elo_home'])
    Xv, yv = prep_xy(valX, fcols)
    Xv_s = ss.transform(Xv)

    # Re-entrenamos un NN rápido y calibramos VS en val para obtener P_val bien calibradas:
    prob_val_meta = train_and_predict_once(ss.transform(prep_xy(inject_elo(base_train_base, df, best['elo_k'], best['elo_home']), fcols)[0]),
                                           prep_xy(inject_elo(base_train_base, df, best['elo_k'], best['elo_home']), fcols)[1],
                                           Xv_s, cw, epochs=60, val_split=0.12, seed=13)
    logits_val = np.log(np.clip(prob_val_meta, 1e-9, 1.0))
    if USE_VECTOR_SCALING:
        s_v, b_v = fit_vector_scaling(logits_val, yv, l2=1e-2)
        P_val_c = apply_vector_scaling(logits_val, s_v, b_v)
    else:
        P_val_c = prob_val_meta

    # Meta features
    feats_val = build_meta_X_features(P_val_c, base_val)
    y_meta = (yv==1).astype(int)
    meta = LogisticRegression(max_iter=300, class_weight='balanced').fit(feats_val, y_meta)
    # Umbral óptimo en val
    base_lbl_val = np.array(['1','X','2'])[P_val_c.argmax(1)]
    p_flip_val = meta.predict_proba(feats_val)[:,1]
    thr_meta = tune_threshold_for_metaX(p_flip_val, base_lbl_val, yv)

    # Aplicamos en test
    feats_te = build_meta_X_features(probs_final_for_decision, test_set)
    p_flip_te = meta.predict_proba(feats_te)[:,1]
    preds = base_labels.copy()
    mask_flip = (preds!='X') & (p_flip_te >= thr_meta)
    preds[mask_flip] = 'X'
else:
    preds = base_labels

# ===== Reporte (HONEST: late only) =====
y_true_lbl = pd.Series(y_te).map({0:'1',1:'X',2:'2'}).values

def report_block(tag, mask, preds_use):
    y_true_m = y_true_lbl[mask]
    preds_m  = preds_use[mask]
    acc = float((preds_m == y_true_m).mean())
    macro = f1_score(pd.Series(y_true_m).map({'1':0,'X':1,'2':2}), pd.Series(preds_m).map({'1':0,'X':1,'2':2}), average='macro')
    print(f"\n[{tag}] acc={acc:.4f} | macro-F1={macro:.4f} | n={mask.sum()}")
    print(classification_report(pd.Series(y_true_m).map({'1':0,'X':1,'2':2}), pd.Series(preds_m).map({'1':0,'X':1,'2':2}), target_names=['1','X','2']))
    print(confusion_matrix(pd.Series(y_true_m).map({'1':0,'X':1,'2':2}), pd.Series(preds_m).map({'1':0,'X':1,'2':2}), labels=[0,1,2]))

# Global (todo test 24/25)
report_block(f"TEST {TEST_SEASON} (GLOBAL)", np.ones(len(preds), dtype=bool), preds)

if USE_HONEST_EVAL:
    report_block(f"TEST {TEST_SEASON} EARLY (j<=25)", mask_early, preds)
    report_block(f"TEST {TEST_SEASON} LATE  (j>25)",  mask_late, preds)

# Distribuciones
dist_pred = pd.Series(preds).value_counts(normalize=True).reindex(['1','X','2']).fillna(0.0)
print("\n📊 Distribución de predicciones (test):")
print(dist_pred.reset_index().rename(columns={'index':'prediccion',0:'proporcion'}))

# Por jornada
df_out = test_set.copy()
df_out['prediccion_1x2'] = preds
df_out['acierto'] = (df_out['prediccion_1x2'] == df_out['resultado_1x2'])
prec_j = df_out.groupby('jornada')['acierto'].mean().reset_index().sort_values('jornada')
print("\n📈 Precisión por jornada:")
print(prec_j)
