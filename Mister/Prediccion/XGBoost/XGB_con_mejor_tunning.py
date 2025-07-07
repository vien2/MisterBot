import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from xgboost import XGBClassifier
from utils import conexion_db
from psycopg2.extras import execute_values
import matplotlib.pyplot as plt
from patterns.df_postgresql import cargar_dataframe_postgresql

# ------------------------ #
# 1. Carga de datos
# ------------------------ #

query_partidos = """
SELECT
    jl.temporada,
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
WHERE jl.goles_local IS NOT NULL AND jl.goles_visitante IS NOT NULL AND jl.temporada = '24/25'
ORDER BY jl.jornada ASC
"""

query_jugadores = "SELECT * FROM chavalitos.v_datos_jornadas WHERE temporada = '24/25'"
query_equipos = "SELECT DISTINCT id_jugador, equipo FROM chavalitos.v_datos_jugador WHERE temporada = '24/25'"

with conexion_db() as conn:
    df_partidos = pd.read_sql(query_partidos, conn)
    df_jugadores = pd.read_sql(query_jugadores, conn)
    df_equipos = pd.read_sql(query_equipos, conn)

# ------------------------ #
# 2. Agregar estadísticas de jugadores por equipo
# ------------------------ #

def extraer_evento(eventos, palabra):
    if pd.isna(eventos):
        return 0
    return int(palabra in eventos.lower())

eventos_binarios = [
    'gol', 'asistencia', 'penalti', 'penalti fallado', 'penalti parado',
    'autogol', 'tarjeta amarilla', 'tarjeta roja', 'doble amarilla', 'entrada', 'salida'
]

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

stats_local = agg_stats.copy()
stats_local.columns = ['jornada'] + [f"{col}_local" for col in stats_local.columns[1:]]

stats_visit = agg_stats.copy()
stats_visit.columns = ['jornada'] + [f"{col}_visitante" for col in stats_visit.columns[1:]]

df = df_partidos.copy()
df = df.merge(stats_local, on=['jornada', 'equipo_local'], how='left')
df = df.merge(stats_visit, on=['jornada', 'equipo_visitante'], how='left')

# ------------------------ #
# 3. Variables adicionales
# ------------------------ #

df['ratio_victorias_local'] = df['ganados_local'] / df['partidos_jugados_local']
df['ratio_victorias_visitante'] = df['ganados_visitante'] / df['partidos_jugados_visitante']
df['ppp_local'] = df['puntos_local'] / df['partidos_jugados_local']
df['ppp_visitante'] = df['puntos_visitante'] / df['partidos_jugados_visitante']
df['abs_dif_goles_local'] = df['dif_goles_local'].abs()
df['abs_dif_goles_visitante'] = df['dif_goles_visitante'].abs()

# ------------------------ #
# 4. Simulación jornada a jornada
# ------------------------ #

features = [col for col in df.columns if any(x in col for x in [
    'posicion_', 'puntos_', 'ganados_', 'empatados_', 'perdidos_', 'dif_goles_', 'ratio_', 'ppp_', 'abs_dif_',
    'goles_esperados_', 'asistencias_esperadas_', 'pases_clave_', 'tiros_', 'paradas_', 'tarjeta_', 'gol_',
    'asistencia_', 'duelos_ganados_', 'intercepciones_', 'faltas_cometidas_'
])]

params_mejores = {
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_child_weight': 3,
    'n_estimators': 200
}

resultados = []

for jornada in sorted(df['jornada'].unique()):
    df_train = df[df['jornada'] < jornada].copy()
    df_test = df[df['jornada'] == jornada].copy()
    if df_train.empty or df_test.empty:
        continue

    X_train = df_train[features]
    y_train = df_train['resultado_1x2'].map({'1': 0, 'X': 1, '2': 2})
    X_test = df_test[features]
    y_test = df_test['resultado_1x2'].map({'1': 0, 'X': 1, '2': 2})

    model = XGBClassifier(**params_mejores, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)

    df_test['prediccion_1x2'] = model.predict(X_test)
    df_test['confianza'] = model.predict_proba(X_test).max(axis=1)
    df_test['acierto'] = df_test['prediccion_1x2'] == y_test

    resultados.append(df_test)

df_preds = pd.concat(resultados, ignore_index=True)

# ------------------------ #
# 5. Evaluación
# ------------------------ #

precision_total = df_preds['acierto'].mean()
print(f"\n✅ Precisión total XGBoost con mejor modelo: {precision_total:.2%}")

precision_por_jornada = df_preds.groupby('jornada')['acierto'].mean()
print("\n📊 Precisión por jornada:")
print(precision_por_jornada)

plt.figure(figsize=(12, 6))
plt.plot(precision_por_jornada.index, precision_por_jornada.values, marker='o')
plt.xlabel("Jornada")
plt.ylabel("Precisión")
plt.title("Evolución precisión jornada a jornada (XGBoost mejor modelo)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------ #
# 6. Guardar en PostgreSQL
# ------------------------ #

# Preparar para guardar
df_output = df_preds[[
    'temporada', 'jornada', 'equipo_local', 'equipo_visitante'
]].copy()
df_output['modelo'] = 'XGBoost_best'
df_output['prediccion_1x2'] = df_preds['prediccion_1x2'].map({0: '1', 1: 'X', 2: '2'})
df_output['confianza'] = df_preds['confianza']

cargar_dataframe_postgresql(
    df_output,
    schema='dbo',
    tabla='predicciones_jornada',
    clave_conflicto=['temporada', 'jornada', 'equipo_local', 'equipo_visitante', 'modelo']
)
