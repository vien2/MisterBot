import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import conexion_db
import matplotlib.pyplot as plt

jornada_objetivo = 34

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

query_resultados = f"""
    SELECT equipo_local, equipo_visitante,
            CASE
                WHEN goles_local > goles_visitante THEN '1'
                WHEN goles_local = goles_visitante THEN 'X'
                ELSE '2'
            END AS resultado_real
    FROM chavalitos.v_jornadas_liga
    WHERE temporada = '24/25' AND jornada = {jornada_objetivo}
    """

query_jugadores = "SELECT * FROM chavalitos.v_datos_jornadas WHERE temporada = '24/25'"
query_equipos_jugador = "SELECT DISTINCT id_jugador, equipo FROM chavalitos.v_datos_jugador WHERE temporada = '24/25'"

with conexion_db() as conn:
    df = pd.read_sql(query_partidos, conn)
    df_resultados_reales = pd.read_sql(query_resultados, conn)
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

df = df.merge(stats_local, left_on=['jornada', 'equipo_local'], right_on=['jornada', 'equipo_local'], how='left')
df = df.merge(stats_visit, left_on=['jornada', 'equipo_visitante'], right_on=['jornada', 'equipo_visitante'], how='left')

# ------------------------ #
# 3. Crear variables adicionales
# ------------------------ #

df['ratio_victorias_local'] = df['ganados_local'] / df['partidos_jugados_local']
df['ratio_victorias_visitante'] = df['ganados_visitante'] / df['partidos_jugados_visitante']
df['ppp_local'] = df['puntos_local'] / df['partidos_jugados_local']
df['ppp_visitante'] = df['puntos_visitante'] / df['partidos_jugados_visitante']
df['abs_dif_goles_local'] = df['dif_goles_local'].abs()
df['abs_dif_goles_visitante'] = df['dif_goles_visitante'].abs()

# ------------------------ #
# 4. Definir variables del modelo
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
# 5. Separar train y test
# ------------------------ #

df_train = df[df['jornada'] < jornada_objetivo].copy()
df_test = df[df['jornada'] == jornada_objetivo].copy()

if df_test.empty:
    print(f"⚠️ No hay partidos registrados para la jornada {jornada_objetivo}.")
else:
    X_train = df_train[features]
    y_train = df_train['resultado_1x2']
    X_test = df_test[features]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    confianza_pred = y_proba.max(axis=1)

    df_test['prediccion_1x2'] = y_pred
    df_test['confianza'] = confianza_pred

    print(f"\n🔮 Predicción para jornada {jornada_objetivo} (con confianza):\n")
    for _, row in df_test.iterrows():
        print(f" - {row['equipo_local']} vs {row['equipo_visitante']}: {row['prediccion_1x2']} "
              f"(confianza: {row['confianza']:.2%})")

    # ------------------------ #
    # 6. Evaluar y graficar
    # ------------------------ #

    df_test = df_test.merge(df_resultados_reales, on=['equipo_local', 'equipo_visitante'], how='left')
    df_test['acierto'] = df_test['prediccion_1x2'] == df_test['resultado_real']

    partidos = df_test['equipo_local'] + ' vs ' + df_test['equipo_visitante']
    colores = df_test['acierto'].map({True: 'green', False: 'red'})

    plt.figure(figsize=(12, 6))
    bars = plt.bar(partidos, df_test['confianza'], color=colores)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.ylabel("Confianza")
    plt.title(f"Confianza y aciertos – Jornada {jornada_objetivo}")

    for bar, conf in zip(bars, df_test['confianza']):
        plt.text(bar.get_x() + bar.get_width() / 2, conf + 0.01, f"{conf:.0%}", ha='center', fontsize=8)

    plt.tight_layout()
    plt.show()

    precision = df_test['acierto'].mean()
    print(f"\n✅ Precisión del modelo en jornada {jornada_objetivo}: {precision:.2%}")
