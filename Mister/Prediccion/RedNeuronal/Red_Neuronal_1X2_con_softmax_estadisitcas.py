# -*- coding: utf-8 -*-
# MisterBot - Modelo Neuronal 1X2 (mejorado, con validación jornada a jornada)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical
from utils import conexion_db
from patterns.df_postgresql import cargar_dataframe_postgresql  # 🔴 Comentado

# ------------------------ #
# CONFIGURACIÓN
# ------------------------ #
temporadas_hist = ['23/24', '24/25', '25/26']
temporada_objetivo = '25/26'

# ------------------------ #
# CONSULTAS
# ------------------------ #
def query_partidos(temporada):
    return f"""
    SELECT
        jl.jornada::integer AS jornada,
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
    LEFT JOIN chavalitos.v_datos_laliga el
        ON el.equipo = jl.equipo_local AND el.temporada = jl.temporada
    LEFT JOIN chavalitos.v_datos_laliga ev
        ON ev.equipo = jl.equipo_visitante AND ev.temporada = jl.temporada
    WHERE jl.temporada = '{temporada}'
    ORDER BY jornada ASC
    """

def query_jugadores(temporada):
    return f"""
    SELECT
        j.jornada::integer AS jornada,
        j.id_jugador,
        j.goles_esperados,
        j.asistencias_esperadas,
        j.pases_clave,
        j.tiros_fuera,
        j.tiros_a_puerta,
        j.paradas,
        j.intercepciones,
        j.entradas_totales,
        j.faltas_cometidas,
        dj.equipo,
        j.temporada
    FROM chavalitos.v_datos_jornadas j
    LEFT JOIN chavalitos.v_datos_jugador dj
        ON dj.id_jugador = j.id_jugador AND dj.temporada = j.temporada
    WHERE j.temporada = '{temporada}'
    """

query_proxima_jornada = f"""
WITH jornadas AS (
    SELECT jornada::integer AS jornada,
           COUNT(*) AS partidos,
           SUM(CASE WHEN resultado ~ '^\d+\\s*[-·]\\s*\\d+$' THEN 1 ELSE 0 END) AS jugados
    FROM chavalitos.jornadas_liga
    WHERE temporada = '{temporada_objetivo}'
    GROUP BY jornada
)
SELECT MIN(jornada) AS proxima_jornada
FROM jornadas
WHERE jugados < partidos;
"""

# ------------------------ #
# LECTURA DE DATOS
# ------------------------ #
with conexion_db() as conn:
    dfs = [pd.read_sql(query_partidos(temp), conn) for temp in temporadas_hist]
    df_partidos = pd.concat(dfs, ignore_index=True)

    dfj = [pd.read_sql(query_jugadores(temp), conn) for temp in temporadas_hist]
    df_jugadores = pd.concat(dfj, ignore_index=True)

    proxima_jornada = pd.read_sql(query_proxima_jornada, conn).iloc[0]['proxima_jornada']

if pd.isna(proxima_jornada):
    print("⚠️ No hay jornada futura para predecir.")
    raise SystemExit

# ------------------------ #
# AGREGACIÓN JUGADORES → EQUIPO/JORNADA
# ------------------------ #
agg_stats = df_jugadores.groupby(['temporada', 'jornada', 'equipo']).agg({
    'goles_esperados': 'sum',
    'asistencias_esperadas': 'sum',
    'pases_clave': 'sum',
    'tiros_fuera': 'sum',
    'tiros_a_puerta': 'sum',
    'paradas': 'sum',
    'intercepciones': 'sum',
    'entradas_totales': 'sum',
    'faltas_cometidas': 'sum'
}).reset_index()

stats_local = agg_stats.copy()
stats_local.columns = ['temporada', 'jornada', 'equipo_local'] + [f"{c}_local" for c in stats_local.columns[3:]]

stats_visit = agg_stats.copy()
stats_visit.columns = ['temporada', 'jornada', 'equipo_visitante'] + [f"{c}_visitante" for c in stats_visit.columns[3:]]

df_full = df_partidos.merge(stats_local, on=['temporada', 'jornada', 'equipo_local'], how='left')
df_full = df_full.merge(stats_visit, on=['temporada', 'jornada', 'equipo_visitante'], how='left')

# ------------------------ #
# FEATURES
# ------------------------ #
df_full['ratio_victorias_local'] = df_full['ganados_local'] / df_full['partidos_jugados_local']
df_full['ratio_victorias_visitante'] = df_full['ganados_visitante'] / df_full['partidos_jugados_visitante']
df_full['ppp_local'] = df_full['puntos_local'] / df_full['partidos_jugados_local']
df_full['ppp_visitante'] = df_full['puntos_visitante'] / df_full['partidos_jugados_visitante']
df_full['abs_dif_goles_local'] = df_full['dif_goles_local'].abs()
df_full['abs_dif_goles_visitante'] = df_full['dif_goles_visitante'].abs()

# diferencias locales - visitantes en features
df_full['dif_xg'] = df_full['goles_esperados_local'] - df_full['goles_esperados_visitante']
df_full['dif_tiros_puerta'] = df_full['tiros_a_puerta_local'] - df_full['tiros_a_puerta_visitante']
df_full['dif_paradas'] = df_full['paradas_local'] - df_full['paradas_visitante']

features = []
for col in df_full.columns:
    if any(x in col for x in [
        'posicion_', 'puntos_', 'ganados_', 'empatados_', 'perdidos_', 'dif_goles_',
        'ratio_', 'ppp_', 'abs_dif_', 
        'goles_esperados_', 'asistencias_esperadas_', 'pases_clave_',
        'tiros_', 'paradas_', 'intercepciones_', 'entradas_', 'faltas_cometidas_',
        'dif_xg', 'dif_tiros_puerta', 'dif_paradas'
    ]):
        features.append(col)

# ------------------------ #
# VALIDACIÓN JORNADA A JORNADA
# ------------------------ #
resultados = []

for jornada in sorted(df_full['jornada'].unique()):
    df_train = df_full[df_full['jornada'] < jornada]
    df_test = df_full[df_full['jornada'] == jornada]

    if df_train.empty or df_test.empty:
        continue

    X_train = df_train[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = df_test[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    y_train = df_train['resultado_1x2'].map({'1':0,'X':1,'2':2}).astype(int)
    y_test = df_test['resultado_1x2']
    y_train_cat = to_categorical(y_train, num_classes=3)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    class_weights = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_train)
    cw_dict = {i: w for i, w in enumerate(class_weights)}

    sample_weights = np.ones(len(y_train))
    sample_weights[df_train['temporada'] == temporada_objetivo] *= 2.0

    def crear_modelo(input_dim, output_dim=3):
        model = Sequential()
        model.add(Input(shape=(input_dim,)))
        model.add(Dense(128))
        model.add(LeakyReLU(0.1))
        model.add(Dropout(0.2))
        model.add(Dense(64))
        model.add(LeakyReLU(0.1))
        model.add(Dropout(0.15))
        model.add(Dense(32))
        model.add(LeakyReLU(0.1))
        model.add(Dropout(0.1))
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=0.0005),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    model = crear_modelo(X_train.shape[1])
    model.fit(X_train_scaled, y_train_cat,
              epochs=30, batch_size=64, verbose=0,
              class_weight=cw_dict, sample_weight=sample_weights)

    probas = model.predict(X_test_scaled, verbose=0)
    preds = probas.argmax(axis=1)
    mapa = {0:'1',1:'X',2:'2'}

    df_pred = df_test.copy()
    df_pred['prediccion_1x2'] = [mapa[p] for p in preds]
    df_pred['acierto'] = df_pred['prediccion_1x2'] == y_test

    resultados.append(df_pred)

df_all_preds = pd.concat(resultados, ignore_index=True)
print(f"\n✅ Precisión total simulada: {df_all_preds['acierto'].mean():.2%}")

precision_jornada = df_all_preds.groupby('jornada')['acierto'].mean().reset_index()
print("\n📈 Precisión por jornada:")
print(precision_jornada.sort_values(by='acierto', ascending=False))

# ------------------------ #
# PREDICCIÓN JORNADA FUTURA
# ------------------------ #
df_entrenamiento = df_full[(df_full['temporada'] != temporada_objetivo) |
                           (df_full['jornada'] < proxima_jornada)].copy()
df_pred = df_full[(df_full['temporada'] == temporada_objetivo) &
                  (df_full['jornada'] == proxima_jornada) &
                  df_full['goles_local'].isna()].copy()

X_train = df_entrenamiento[features].replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = df_pred[features].replace([np.inf, -np.inf], np.nan).fillna(0)

y_train = df_entrenamiento['resultado_1x2'].map({'1':0,'X':1,'2':2}).astype(int)
y_train_cat = to_categorical(y_train, num_classes=3)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class_weights = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_train)
cw_dict = {i: w for i, w in enumerate(class_weights)}

sample_weights = np.ones(len(y_train))
sample_weights[df_entrenamiento['temporada'] == temporada_objetivo] *= 2.0

model = crear_modelo(X_train.shape[1])
model.fit(X_train_scaled, y_train_cat,
          epochs=40, batch_size=64, verbose=0,
          class_weight=cw_dict, sample_weight=sample_weights)

probas = model.predict(X_test_scaled, verbose=0)
preds = probas.argmax(axis=1)
mapa = {0:'1',1:'X',2:'2'}

df_pred['prediccion_1x2'] = [mapa[p] for p in preds]
df_pred['confianza'] = probas.max(axis=1)

print(f"\n🔮 Predicciones jornada {proxima_jornada} ({temporada_objetivo}):")
for _, row in df_pred.iterrows():
    print(f" - {row['equipo_local']} vs {row['equipo_visitante']}: "
          f"{row['prediccion_1x2']} (confianza: {row['confianza']:.2%})")

df_output = df_pred[['temporada', 'jornada', 'equipo_local', 'equipo_visitante']].copy()
df_output['modelo'] = 'Neuronal_1X2_Softmax_Mejorado'
df_output['prediccion_1x2'] = df_pred['prediccion_1x2']
df_output['confianza'] = df_pred['confianza']

#Guardado en PostgreSQL comentado
cargar_dataframe_postgresql(
    df_output,
    schema='dbo',
    tabla='predicciones_jornada',
    clave_conflicto=['temporada', 'jornada', 'equipo_local', 'equipo_visitante', 'modelo']
)
