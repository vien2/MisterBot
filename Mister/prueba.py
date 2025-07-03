import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from utils import conexion_db  # Ajusta si es necesario

# 1. Consulta SQL (ajustada sin columnas inexistentes)
query = """
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

# 2. Cargar datos
with conexion_db() as conn:
    df = pd.read_sql(query, conn)

# 3. Crear nuevas variables
df['ratio_victorias_local'] = df['ganados_local'] / df['partidos_jugados_local']
df['ratio_victorias_visitante'] = df['ganados_visitante'] / df['partidos_jugados_visitante']
df['ppp_local'] = df['puntos_local'] / df['partidos_jugados_local']
df['ppp_visitante'] = df['puntos_visitante'] / df['partidos_jugados_visitante']
df['abs_dif_goles_local'] = df['dif_goles_local'].abs()
df['abs_dif_goles_visitante'] = df['dif_goles_visitante'].abs()

# 4. Variables a usar
features = [
    'posicion_local', 'puntos_local', 'ganados_local', 'empatados_local', 'perdidos_local', 'dif_goles_local',
    'posicion_visitante', 'puntos_visitante', 'ganados_visitante', 'empatados_visitante', 'perdidos_visitante', 'dif_goles_visitante',
    'ratio_victorias_local', 'ratio_victorias_visitante',
    'ppp_local', 'ppp_visitante',
    'abs_dif_goles_local', 'abs_dif_goles_visitante'
]

# 5. Dividir entre entrenamiento y predicción
df_train = df[df['jornada'] < 37]
df_predict = df[df['jornada'].isin([37, 38])]

X_train = df_train[features]
y_train = df_train['resultado_1x2']
X_pred = df_predict[features]

# 6. Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Predicción
y_pred = model.predict(X_pred)
df_predict = df_predict.copy()
df_predict['prediccion_1x2'] = y_pred

# 8. Evaluación (si ya están jugados)
df_eval = df_predict[df_predict['goles_local'].notnull()].copy()
df_eval['acierto'] = df_eval['prediccion_1x2'] == df_eval['resultado_1x2']

# 9. Mostrar resultados
print("\n📊 Predicciones para Jornadas 37 y 38:\n")
print(df_predict[['jornada', 'equipo_local', 'equipo_visitante', 'prediccion_1x2']])

if not df_eval.empty:
    print("\n🎯 Evaluación de predicciones:")
    print(df_eval[['jornada', 'equipo_local', 'equipo_visitante', 'resultado_1x2', 'prediccion_1x2', 'acierto']])
    print(f"\n✅ Precisión del modelo sobre jornadas 37 y 38: {df_eval['acierto'].mean():.2%}")
else:
    print("\n⚠️ No se pueden evaluar aciertos porque los goles de jornadas 37 y 38 no están disponibles.")

# 10. 🔍 Importancia de variables
importancias = model.feature_importances_
importancias_df = pd.DataFrame({'feature': features, 'importancia': importancias})
importancias_df = importancias_df.sort_values(by='importancia', ascending=True)

# 11. 📊 Visualización
plt.figure(figsize=(10, 6))
plt.barh(importancias_df['feature'], importancias_df['importancia'])
plt.xlabel("Importancia")
plt.title("📊 Importancia de cada variable en el modelo")
plt.tight_layout()
plt.show()
