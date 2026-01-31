# -*- coding: utf-8 -*-
# MisterBot - Postproceso: Modelo Neuronal 1X2 con distribución de clases

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical

from utils import log
from patterns.df_postgresql import cargar_dataframe_postgresql

# ------------------------ #
# CONFIGURACIÓN
# ------------------------ #
TEMPORADAS_HIST = ['23/24', '24/25', '25/26']
TEMPORADA_OBJETIVO = '25/26'

# ------------------------ #
# CONSULTAS
# ------------------------ #
def _query_partidos(temporada: str) -> str:
    return f"""
    SELECT
        jl.jornada,
        jl.equipo_local,
        jl.equipo_visitante,
        el.posicion AS posicion_local,
        el.puntos AS puntos_local,
        el.partidos_jugados AS partidos_jugados_local,
        el.partidos_ganados AS ganados_local,
        el.partidos_empatados AS empatados_local,
        el.partidos_perdidos AS perdidos_local,
        el.diferencia_goles AS dif_goles_local,
        ev.posicion AS posicion_visitante,
        ev.puntos AS puntos_visitante,
        ev.partidos_jugados AS partidos_jugados_visitante,
        ev.partidos_ganados AS ganados_visitante,
        ev.partidos_empatados AS empatados_visitante,
        ev.partidos_perdidos AS perdidos_visitante,
        ev.diferencia_goles AS dif_goles_visitante,
        jl.goles_local,
        jl.goles_visitante,
        CASE
            WHEN jl.goles_local > jl.goles_visitante THEN '1'
            WHEN jl.goles_local = jl.goles_visitante THEN 'X'
            ELSE '2'
        END AS resultado_1x2,
        jl.temporada
    FROM chavalitos.v_jornadas_liga jl
    LEFT JOIN chavalitos.v_datos_laliga el ON el.equipo = jl.equipo_local AND el.temporada = jl.temporada
    LEFT JOIN chavalitos.v_datos_laliga ev ON ev.equipo = jl.equipo_visitante AND ev.temporada = jl.temporada
    WHERE jl.temporada = '{temporada}'
    ORDER BY jl.jornada ASC
    """

def _query_proxima_jornada(temporada_objetivo: str) -> str:
    # 1. Buscamos la jornada MINIMA donde falten más de 2 partidos.
    # Esto salta la jornada 16 (solo falta 1) y salta la 17 (faltan 0).
    return rf"""
    WITH jornadas_status AS (
        SELECT jornada::integer,
               COUNT(*) AS partidos_totales,
               SUM(CASE WHEN resultado ~ '^\d+\s*[-·]\s*\d+$' THEN 1 ELSE 0 END) AS jugados
        FROM chavalitos.jornadas_liga
        WHERE temporada = '{temporada_objetivo}'
        GROUP BY jornada
    )
    SELECT MIN(jornada) AS proxima_jornada
    FROM jornadas_status
    WHERE (partidos_totales - jugados) > 2; 
    """

# ------------------------ #
# MODELO
# ------------------------ #
def _crear_modelo(input_dim: int, output_dim: int = 3):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(128))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.3))
    model.add(Dense(64))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0007),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ------------------------ #
# ENTRYPOINT POSTPROCESO
# ------------------------ #
def post_neuronal_1x2(conn, schema=None):
    _ = schema
    log("post_neuronal_1x2: inicio")

    # --- LECTURA DE DATOS ---
    log(f"Lectura histórica temporadas: {TEMPORADAS_HIST} | objetivo: {TEMPORADA_OBJETIVO}")
    dfs = [pd.read_sql(_query_partidos(temp), conn) for temp in TEMPORADAS_HIST]
    dfs = [d for d in dfs if d is not None and not d.empty]
    
    if not dfs:
        log("⚠️ No hay datos históricos. Salgo.")
        return
    df_partidos = pd.concat(dfs, ignore_index=True, sort=False)

    # --- OBTENER PROXIMA JORNADA REAL ---
    try:
        proxima_jornada = pd.read_sql(_query_proxima_jornada(TEMPORADA_OBJETIVO), conn).iloc[0]['proxima_jornada']
    except Exception:
        proxima_jornada = None

    if pd.isna(proxima_jornada):
        log("⚠️ No se detectó ninguna jornada futura pendiente. Salgo.")
        return
    
    log(f"📅 Próxima jornada OBJETIVO: {int(proxima_jornada)}")

    # --- FEATURES ---
    df = df_partidos.copy()
    
    cols_num = ['partidos_jugados_local', 'partidos_jugados_visitante', 
                'ganados_local', 'ganados_visitante', 'puntos_local', 'puntos_visitante']
    for c in cols_num:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    df['ratio_victorias_local'] = df['ganados_local'] / df['partidos_jugados_local'].replace(0, 1)
    df['ratio_victorias_visitante'] = df['ganados_visitante'] / df['partidos_jugados_visitante'].replace(0, 1)
    df['ppp_local'] = df['puntos_local'] / df['partidos_jugados_local'].replace(0, 1)
    df['ppp_visitante'] = df['puntos_visitante'] / df['partidos_jugados_visitante'].replace(0, 1)
    df['abs_dif_goles_local'] = df['dif_goles_local'].abs()
    df['abs_dif_goles_visitante'] = df['dif_goles_visitante'].abs()

    features = [c for c in df.columns if any(x in c for x in
               ['posicion_', 'puntos_', 'ganados_', 'empatados_', 'perdidos_', 'dif_goles_',
                'ratio_', 'ppp_', 'abs_dif_'])]

    # --- SPLIT ---
    # Entrenamiento: Todo lo ANTERIOR a la jornada objetivo (incluye la 16 y 17 si ya se jugaron)
    df_entrenamiento = df[
        (df['temporada'] != TEMPORADA_OBJETIVO) | 
        (df['jornada'] < proxima_jornada)
    ].copy()
    
    # Predicción: SOLO la jornada objetivo estricta (==)
    df_pred = df[
        (df['temporada'] == TEMPORADA_OBJETIVO) &
        (df['jornada'] == proxima_jornada) & 
        (df['goles_local'].isna())
    ].copy()

    if df_pred.empty:
        log(f"ℹ️ No hay partidos pendientes EXACTAMENTE en la jornada {proxima_jornada}. Salgo.")
        return

    X_train = df_entrenamiento[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test  = df_pred[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y_train = df_entrenamiento['resultado_1x2'].map({'1':0,'X':1,'2':2}).fillna(1).astype(int)
    y_train_cat = to_categorical(y_train, num_classes=3)

    # --- SCALER ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # --- WEIGHTS ---
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_train)
    cw_dict = {i: w for i, w in enumerate(class_weights)}

    temporada_arr = df_entrenamiento['temporada'].to_numpy()
    y_arr = y_train.to_numpy()

    boost_temporada = np.where(temporada_arr == TEMPORADA_OBJETIVO, 4.0, 1.0).astype(np.float32)
    peso_clase = np.array([cw_dict[y] for y in y_arr]).astype(np.float32)
    sample_weights = boost_temporada * peso_clase

    # --- FIT & PREDICT ---
    model = _crear_modelo(X_train.shape[1])
    model.fit(X_train_scaled, y_train_cat, epochs=40, batch_size=16, verbose=0, sample_weight=sample_weights)

    probas = model.predict(X_test_scaled, verbose=0)
    preds = probas.argmax(axis=1)
    mapa = {0:'1', 1:'X', 2:'2'}

    df_pred = df_pred.copy()
    df_pred['prediccion_1x2'] = [mapa.get(p, 'X') for p in preds]
    df_pred['confianza'] = probas.max(axis=1)

    log(f"🔮 Predicciones para Jornada {int(proxima_jornada)}: {len(df_pred)} partidos")
    for _, row in df_pred.iterrows():
        log(f" - {row['equipo_local']} vs {row['equipo_visitante']}: {row['prediccion_1x2']} ({row['confianza']:.2%})")

    # --- UPSERT ---
    df_output = df_pred[['temporada', 'jornada', 'equipo_local', 'equipo_visitante']].copy()
    df_output['modelo'] = 'Neuronal_1X2_Softmax'
    df_output['prediccion_1x2'] = df_pred['prediccion_1x2']
    df_output['confianza'] = df_pred['confianza']

    cargar_dataframe_postgresql(
        df_output,
        schema='dbo',
        tabla='predicciones_jornada',
        clave_conflicto=['temporada', 'jornada', 'equipo_local', 'equipo_visitante', 'modelo']
    )

    log("post_neuronal_1x2: fin OK")