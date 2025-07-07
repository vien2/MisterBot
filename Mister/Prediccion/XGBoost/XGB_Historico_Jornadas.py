import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from xgboost import XGBClassifier
from utils import conexion_db
import matplotlib.pyplot as plt

# ------------------------ #
# 1. Carga de datos
# ------------------------ #

query_partidos = """
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
  AND jl.temporada = '24/25'
ORDER BY jl.jornada ASC
"""

query_jugadores = "SELECT * FROM chavalitos.v_datos_jornadas WHERE temporada = '24/25'"
query_equipos_jugador = "SELECT DISTINCT id_jugador, equipo FROM chavalitos.v_datos_jugador WHERE temporada = '24/25'"

with conexion_db() as conn:
    df_partidos = pd.read_sql(query_partidos, conn)
    df_jugadores = pd.read_sql(query_jugadores, conn)
    df_equipos = pd.read_sql(query_equipos_jugador, conn)

# ------------------------ #
# 2. Procesar eventos y agregar por equipo
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
df = df.merge(stats_local, left_on=['jornada', 'equipo_local'], right_on=['jornada', 'equipo_local'], how='left')
df = df.merge(stats_visit, left_on=['jornada', 'equipo_visitante'], right_on=['jornada', 'equipo_visitante'], how='left')

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
# 4. Features
# ------------------------ #

features = [
    'posicion_local', 'puntos_local', 'ganados_local', 'empatados_local', 'perdidos_local', 'dif_goles_local',
    'posicion_visitante', 'puntos_visitante', 'ganados_visitante', 'empatados_visitante', 'perdidos_visitante', 'dif_goles_visitante',
    'ratio_victorias_local', 'ratio_victorias_visitante', 'ppp_local', 'ppp_visitante',
    'abs_dif_goles_local', 'abs_dif_goles_visitante',
    'goles_esperados_local', 'goles_esperados_visitante',
    'asistencias_esperadas_local', 'asistencias_esperadas_visitante',
    'pases_clave_local', 'pases_clave_visitante',
    'tiros_fuera_local', 'tiros_fuera_visitante',
    'tiros_a_puerta_local', 'tiros_a_puerta_visitante',
    'paradas_local', 'paradas_visitante',
    'tarjeta_amarilla_local', 'tarjeta_amarilla_visitante',
    'tarjeta_roja_local', 'tarjeta_roja_visitante',
    'gol_local', 'gol_visitante',
    'asistencia_local', 'asistencia_visitante',
    'duelos_ganados_local', 'duelos_ganados_visitante',
    'intercepciones_local', 'intercepciones_visitante',
    'faltas_cometidas_local', 'faltas_cometidas_visitante'
]

# ------------------------ #
# 5. Simulacion jornada a jornada
# ------------------------ #

mapa_clases = {'1': 0, 'X': 1, '2': 2}
inv_mapa = {v: k for k, v in mapa_clases.items()}

resultados = []

for jornada in sorted(df['jornada'].unique()):
    df_train = df[df['jornada'] < jornada].copy()
    df_test = df[df['jornada'] == jornada].copy()

    if df_train.empty or df_test.empty:
        continue

    X_train = df_train[features]
    y_train = df_train['resultado_1x2'].map(mapa_clases)
    X_test = df_test[features]

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)

    df_test['prediccion_1x2'] = model.predict(X_test)
    df_test['prediccion_1x2'] = df_test['prediccion_1x2'].map(inv_mapa)
    df_test['confianza'] = model.predict_proba(X_test).max(axis=1)
    df_test['acierto'] = df_test['prediccion_1x2'] == df_test['resultado_1x2']

    resultados.append(df_test)

# ------------------------ #
# 6. Evaluación
# ------------------------ #

df_preds = pd.concat(resultados, ignore_index=True)

precision_total = df_preds['acierto'].mean()
print(f"\n✅ Precisión total XGBoost en simulación: {precision_total:.2%}")

precision_por_jornada = df_preds.groupby('jornada')['acierto'].mean()
print("\n📊 Precisión por jornada:")
print(precision_por_jornada)

plt.figure(figsize=(12, 6))
plt.plot(precision_por_jornada.index, precision_por_jornada.values, marker='o')
plt.xlabel("Jornada")
plt.ylabel("Precisión")
plt.title("📊 Evolución precisión jornada a jornada (XGBoost)")
plt.grid(True)
plt.tight_layout()
plt.show()
