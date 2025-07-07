import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from xgboost import XGBClassifier
from utils import conexion_db
from patterns.df_postgresql import cargar_dataframe_postgresql

# ------------------------ #
# CONFIGURACIÓN
# ------------------------ #

temporada_objetivo = '25/26'
mejor_params = {
    'n_estimators': 200,
    'max_depth': 6,
    'min_child_weight': 3,
    'learning_rate': 0.1,
    'use_label_encoder': False,
    'eval_metric': 'mlogloss',
    'random_state': 42
}

# ------------------------ #
# 1. Consultar datos base
# ------------------------ #

query_partidos_jugados = f"""
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
    END AS resultado_1x2
FROM chavalitos.v_jornadas_liga jl
LEFT JOIN chavalitos.v_datos_laliga el ON el.equipo = jl.equipo_local AND el.temporada = jl.temporada
LEFT JOIN chavalitos.v_datos_laliga ev ON ev.equipo = jl.equipo_visitante AND ev.temporada = jl.temporada
WHERE jl.goles_local IS NOT NULL
  AND jl.goles_visitante IS NOT NULL
  AND jl.temporada = '{temporada_objetivo}'
ORDER BY jl.jornada ASC
"""

query_partidos_pendientes = f"""
SELECT *
FROM chavalitos.v_jornadas_liga
WHERE goles_local IS NULL AND goles_visitante IS NULL AND temporada = '{temporada_objetivo}'
ORDER BY jornada ASC
LIMIT 1
"""

query_jugadores = f"SELECT * FROM chavalitos.v_datos_jornadas WHERE temporada = '{temporada_objetivo}'"
query_equipos = f"SELECT DISTINCT id_jugador, equipo FROM chavalitos.v_datos_jugador WHERE temporada = '{temporada_objetivo}'"

# ------------------------ #
# 2. Cargar datos
# ------------------------ #

with conexion_db() as conn:
    df_partidos = pd.read_sql(query_partidos_jugados, conn)
    df_pred_jornada = pd.read_sql(query_partidos_pendientes, conn)
    df_jugadores = pd.read_sql(query_jugadores, conn)
    df_equipos = pd.read_sql(query_equipos, conn)

if df_pred_jornada.empty:
    print("⚠️ No hay jornadas futuras disponibles para predecir.")
    exit()

jornada_a_predecir = df_pred_jornada['jornada'].iloc[0]
print(f"\n🔮 Prediciendo jornada {jornada_a_predecir}...")

# ------------------------ #
# 3. Enriquecer datos con estadísticas
# ------------------------ #

def extraer_evento(eventos, palabra):
    if pd.isna(eventos):
        return 0
    return int(palabra in eventos.lower())

eventos_binarios = ['gol', 'asistencia', 'penalti', 'penalti fallado', 'penalti parado',
                    'autogol', 'tarjeta amarilla', 'tarjeta roja', 'doble amarilla', 'entrada', 'salida']

for evento in eventos_binarios:
    df_jugadores[evento.replace(' ', '_')] = df_jugadores['eventos'].apply(lambda x: extraer_evento(x, evento))

df_jugadores = df_jugadores.merge(df_equipos, on='id_jugador', how='left')

agg_stats = df_jugadores.groupby(['jornada', 'equipo']).agg({
    'goles_esperados': 'mean',
    'asistencias_esperadas': 'mean',
    'pases_clave': 'mean',
    'tiros_fuera': 'mean',
    'tiros_a_puerta': 'mean',
    'paradas': 'sum',
    'duelos_ganados': 'mean',
    'intercepciones': 'mean',
    'faltas_cometidas': 'mean',
    'tarjeta_amarilla': 'sum',
    'tarjeta_roja': 'sum',
    'gol': 'sum',
    'asistencia': 'sum'
}).reset_index()

# Preparamos local y visitante
stats_local = agg_stats.copy()
stats_local.columns = ['jornada'] + [f"{col}_local" for col in stats_local.columns[1:]]

stats_visit = agg_stats.copy()
stats_visit.columns = ['jornada'] + [f"{col}_visitante" for col in stats_visit.columns[1:]]

df_partidos = df_partidos.merge(stats_local, left_on=['jornada', 'equipo_local'], right_on=['jornada', 'equipo_local'], how='left')
df_partidos = df_partidos.merge(stats_visit, left_on=['jornada', 'equipo_visitante'], right_on=['jornada', 'equipo_visitante'], how='left')

df_pred_jornada = df_pred_jornada.merge(stats_local, on=['jornada', 'equipo_local'], how='left')
df_pred_jornada = df_pred_jornada.merge(stats_visit, on=['jornada', 'equipo_visitante'], how='left')

# ------------------------ #
# 4. Variables adicionales
# ------------------------ #

def add_vars(df):
    df['ratio_victorias_local'] = df['ganados_local'] / df['partidos_jugados_local']
    df['ratio_victorias_visitante'] = df['ganados_visitante'] / df['partidos_jugados_visitante']
    df['ppp_local'] = df['puntos_local'] / df['partidos_jugados_local']
    df['ppp_visitante'] = df['puntos_visitante'] / df['partidos_jugados_visitante']
    df['abs_dif_goles_local'] = df['dif_goles_local'].abs()
    df['abs_dif_goles_visitante'] = df['dif_goles_visitante'].abs()
    return df

df_partidos = add_vars(df_partidos)
df_pred_jornada = add_vars(df_pred_jornada)

# ------------------------ #
# 5. Features y predicción
# ------------------------ #

features = [col for col in df_partidos.columns if any(metric in col for metric in [
    'posicion_', 'puntos_', 'ganados_', 'empatados_', 'perdidos_', 'dif_goles_', 'ratio_victorias_', 'ppp_', 'abs_dif_goles_',
    'goles_esperados_', 'asistencias_esperadas_', 'pases_clave_', 'tiros_fuera_', 'tiros_a_puerta_', 'paradas_', 'tarjeta_',
    'gol_', 'asistencia_', 'duelos_ganados_', 'intercepciones_', 'faltas_cometidas_'
])]

X_train = df_partidos[features]
y_train = df_partidos['resultado_1x2'].map({'1': 0, 'X': 1, '2': 2})
X_test = df_pred_jornada[features]

model = XGBClassifier(**mejor_params)
model.fit(X_train, y_train)

y_pred_encoded = model.predict(X_test)
y_proba = model.predict_proba(X_test).max(axis=1)
inv_mapa = {0: '1', 1: 'X', 2: '2'}

df_pred_jornada['prediccion_1x2'] = pd.Series(y_pred_encoded).map(inv_mapa)
df_pred_jornada['confianza'] = y_proba

# ------------------------ #
# 6. Mostrar resultados
# ------------------------ #

print(f"\n📊 Predicciones jornada {jornada_a_predecir} ({temporada_objetivo}):\n")
for _, row in df_pred_jornada.iterrows():
    print(f" - {row['equipo_local']} vs {row['equipo_visitante']}: {row['prediccion_1x2']} "
          f"(confianza: {row['confianza']:.2%})")

# ------------------------ #
# 7. Guardar en PostgreSQL
# ------------------------ #

df_output = df_pred_jornada[['jornada', 'equipo_local', 'equipo_visitante', 'prediccion_1x2', 'confianza']].copy()
df_output['modelo'] = 'XGBoost_best'
df_output['temporada'] = temporada_objetivo

# Reordenar columnas por claridad
df_output = df_output[['temporada', 'jornada', 'equipo_local', 'equipo_visitante', 'modelo', 'prediccion_1x2', 'confianza']]

cargar_dataframe_postgresql(
    df_output,
    schema='dbo',
    tabla='predicciones_jornada',
    clave_conflicto=['temporada', 'jornada', 'equipo_local', 'equipo_visitante', 'modelo']
)

