import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils import conexion_db

# ------------------------ #
# 1. Consultas a la BD
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

query_equipos_jugador = """
SELECT DISTINCT id_jugador, equipo
FROM chavalitos.v_datos_jugador
WHERE temporada = '24/25'
"""

# ------------------------ #
# 2. Conexión y lectura
# ------------------------ #

with conexion_db() as conn:
    df_partidos = pd.read_sql(query_partidos, conn)
    df_jugadores = pd.read_sql("SELECT * FROM chavalitos.v_datos_jornadas WHERE temporada = '24/25'", conn)
    df_equipos = pd.read_sql(query_equipos_jugador, conn)

# ------------------------ #
# 3. Procesar eventos
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

# ------------------------ #
# 4. Unir equipo y agrupar
# ------------------------ #

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

# ------------------------ #
# 5. Renombrar y unir
# ------------------------ #

stats_local = agg_stats.copy()
stats_local.columns = ['jornada'] + [f"{col}_local" for col in stats_local.columns[1:]]
stats_visit = agg_stats.copy()
stats_visit.columns = ['jornada'] + [f"{col}_visitante" for col in stats_visit.columns[1:]]

df_full = df_partidos.copy()
df_full = df_full.merge(stats_local, left_on=['jornada', 'equipo_local'], right_on=['jornada', 'equipo_local'], how='left')
df_full = df_full.merge(stats_visit, left_on=['jornada', 'equipo_visitante'], right_on=['jornada', 'equipo_visitante'], how='left')

# ------------------------ #
# 6. Variables adicionales
# ------------------------ #

df_full['ratio_victorias_local'] = df_full['ganados_local'] / df_full['partidos_jugados_local']
df_full['ratio_victorias_visitante'] = df_full['ganados_visitante'] / df_full['partidos_jugados_visitante']
df_full['ppp_local'] = df_full['puntos_local'] / df_full['partidos_jugados_local']
df_full['ppp_visitante'] = df_full['puntos_visitante'] / df_full['partidos_jugados_visitante']
df_full['abs_dif_goles_local'] = df_full['dif_goles_local'].abs()
df_full['abs_dif_goles_visitante'] = df_full['dif_goles_visitante'].abs()

# ------------------------ #
# 7. Variables del modelo
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
# 8. Predicción por jornada
# ------------------------ #

resultados = []

for jornada in sorted(df_full['jornada'].unique()):
    df_train = df_full[df_full['jornada'] < jornada]
    df_test = df_full[df_full['jornada'] == jornada]
    if df_train.empty or df_test.empty:
        continue

    X_train = df_train[features]
    y_train = df_train['resultado_1x2']
    X_test = df_test[features]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    df_pred = df_test.copy()
    df_pred['prediccion_1x2'] = y_pred
    resultados.append(df_pred)

df_all_preds = pd.concat(resultados, ignore_index=True)
df_all_preds['acierto'] = df_all_preds['prediccion_1x2'] == df_all_preds['resultado_1x2']

# ------------------------ #
# 9. Resultados y evaluación
# ------------------------ #

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("\n📊 Predicciones completas jornada a jornada:")
print(df_all_preds[['jornada', 'equipo_local', 'equipo_visitante', 'resultado_1x2', 'prediccion_1x2', 'acierto']])

precision_global = df_all_preds['acierto'].mean()
print(f"\n✅ Precisión global del modelo: {precision_global:.2%}")

precision_por_jornada = df_all_preds.groupby('jornada')['acierto'].mean().reset_index()
precision_por_jornada = precision_por_jornada.sort_values(by='acierto', ascending=False)

print("\n📈 Precisión por jornada:")
print(precision_por_jornada)

# ------------------------ #
# 10. Importancia de variables (último modelo)
# ------------------------ #

importancias = model.feature_importances_
importancias_df = pd.DataFrame({'feature': features, 'importancia': importancias})
importancias_df = importancias_df.sort_values(by='importancia', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(importancias_df['feature'], importancias_df['importancia'])
plt.xlabel("Importancia")
plt.title("📊 Importancia de cada variable en el último modelo")
plt.tight_layout()
plt.show()
