# MisterBot - Walk-forward predicción Portería a Cero (Visitante)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.calibration import calibration_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from utils import conexion_db
import matplotlib.pyplot as plt
import seaborn as sns
from patterns.df_postgresql import cargar_dataframe_postgresql

pd.set_option('display.max_rows', None)

# ------------------------ #
# CONFIGURACIÓN
# ------------------------ #
temporada_objetivo = '24/25'
umbral_prediccion = 0.70

# ------------------------ #
# Cargar datos
# ------------------------ #
query = f"""
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
    jl.goles_visitante
FROM chavalitos.v_jornadas_liga jl
LEFT JOIN chavalitos.v_datos_laliga el ON el.equipo = jl.equipo_local AND el.temporada = jl.temporada
LEFT JOIN chavalitos.v_datos_laliga ev ON ev.equipo = jl.equipo_visitante AND ev.temporada = jl.temporada
WHERE jl.goles_local IS NOT NULL AND jl.goles_visitante IS NOT NULL
  AND jl.temporada = '{temporada_objetivo}'
ORDER BY jl.jornada ASC
"""

query_jugadores = f"SELECT * FROM chavalitos.v_datos_jornadas WHERE temporada = '{temporada_objetivo}'"
query_equipos_jugador = f"SELECT DISTINCT id_jugador, equipo FROM chavalitos.v_datos_jugador WHERE temporada = '{temporada_objetivo}'"

with conexion_db() as conn:
    df = pd.read_sql(query, conn)
    df_jugadores = pd.read_sql(query_jugadores, conn)
    df_equipos = pd.read_sql(query_equipos_jugador, conn)

# ------------------------ #
# Agregados de jugadores
# ------------------------ #
def extraer_evento(eventos, palabra):
    if pd.isna(eventos):
        return 0
    return int(palabra in eventos.lower())

for evento in ['gol', 'asistencia', 'penalti', 'penalti fallado', 'penalti parado',
               'autogol', 'tarjeta amarilla', 'tarjeta roja', 'doble amarilla', 'entrada', 'salida']:
    df_jugadores[evento.replace(' ', '_')] = df_jugadores['eventos'].apply(lambda x: extraer_evento(x, evento))

agg_stats = (df_jugadores
    .merge(df_equipos, on='id_jugador', how='left')
    .groupby(['jornada', 'equipo'])
    .agg({
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
    }).reset_index())

stats_local = agg_stats.rename(columns={'equipo': 'equipo_local'})
stats_visit = agg_stats.rename(columns={'equipo': 'equipo_visitante'})

df = df.merge(stats_local, on=['jornada', 'equipo_local'], how='left')
df = df.merge(stats_visit, on=['jornada', 'equipo_visitante'], how='left')

# ------------------------ #
# Feature engineering
# ------------------------ #
df['ratio_victorias_local'] = df['ganados_local'] / df['partidos_jugados_local']
df['ratio_victorias_visitante'] = df['ganados_visitante'] / df['partidos_jugados_visitante']
df['ppp_local'] = df['puntos_local'] / df['partidos_jugados_local']
df['ppp_visitante'] = df['puntos_visitante'] / df['partidos_jugados_visitante']
df['abs_dif_goles_local'] = df['dif_goles_local'].abs()
df['abs_dif_goles_visitante'] = df['dif_goles_visitante'].abs()

features = [col for col in df.columns if any(x in col for x in [
    'posicion_', 'puntos_', 'ganados_', 'empatados_', 'perdidos_', 'dif_goles_', 'ratio_', 'ppp_', 'abs_dif_',
    'goles_esperados_', 'asistencias_esperadas_', 'pases_clave_', 'tiros_', 'paradas_', 'duelos_', 'intercepciones_', 
    'faltas_', 'tarjeta_', 'gol_', 'asistencia_'
])]

# ------------------------ #
# Modelo base
# ------------------------ #
def crear_modelo(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(128))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.3))
    model.add(Dense(64))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.0007), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ------------------------ #
# Entrenamiento walk-forward
# ------------------------ #
df['target_porteria_cero_visitante'] = (df['goles_local'] == 0).astype(int)
resultados = []

for jornada in sorted(df['jornada'].unique()):
    df_train = df[df['jornada'] < jornada]
    df_test = df[df['jornada'] == jornada]
    if df_train.empty or df_test.empty:
        continue

    X_train = df_train[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = df_test[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train = df_train['target_porteria_cero_visitante']
    model = crear_modelo(X_train.shape[1])

    if set(y_train.unique()) == {0, 1}:
        w = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
        class_weight = {0: w[0], 1: w[1]}
    else:
        class_weight = None

    model.fit(X_train_scaled, y_train, epochs=40, batch_size=16, verbose=0, class_weight=class_weight)
    proba = model.predict(X_test_scaled, verbose=0).flatten()
    pred = (proba >= umbral_prediccion).astype(int)

    df_test = df_test.copy()
    df_test['pred_porteria_cero_visitante'] = pred
    df_test['proba_porteria_cero_visitante'] = proba
    df_test['temporada'] = temporada_objetivo
    resultados.append(df_test)

# ------------------------ #
# Evaluación final
# ------------------------ #
df_final = pd.concat(resultados, ignore_index=True)
acc = accuracy_score(df_final['target_porteria_cero_visitante'], df_final['pred_porteria_cero_visitante'])
print(f"\n✅ Precisión global Portería Cero Visitante (walk-forward): {acc:.2%}")

print("\n📊 Clasificación:")
print(classification_report(df_final['target_porteria_cero_visitante'], df_final['pred_porteria_cero_visitante'], digits=4))

cm = confusion_matrix(df_final['target_porteria_cero_visitante'], df_final['pred_porteria_cero_visitante'], labels=[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='BuGn', xticklabels=['Encaja Gol', 'Portería 0'], yticklabels=['Encaja Gol', 'Portería 0'])
plt.title("Matriz de confusión final - Portería Cero (Visitante)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# ------------------------ #
# Curva de calibración
# ------------------------ #
print("\n📈 Curva de calibración - Portería Cero Visitante")
y_true = df_final['target_porteria_cero_visitante']
y_proba = df_final['proba_porteria_cero_visitante']

fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_proba, n_bins=10, strategy='uniform')

plt.figure(figsize=(6, 6))
plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label='Modelo')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfecto')
plt.xlabel('Probabilidad media predicha')
plt.ylabel('Fracción de positivos reales')
plt.title('Curva de calibración - Portería Cero Visitante')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------ #
# Recomendaciones de apuesta por alta confianza
# ------------------------ #
print("\n🎯 Partidos recomendados para apostar por portería a cero visitante (confianza >= 0.80):")
recomendados = df_final[(df_final['pred_porteria_cero_visitante'] == 1) & (df_final['proba_porteria_cero_visitante'] >= 0.8)]
print(recomendados[['jornada', 'equipo_local', 'equipo_visitante', 'proba_porteria_cero_visitante']].sort_values(by='proba_porteria_cero_visitante', ascending=False))

# ------------------------ #
# Guardar en PostgreSQL
# ------------------------ #

df_output = df_final[[
    'temporada', 'jornada', 'equipo_local', 'equipo_visitante'
]].copy()

df_output['modelo'] = 'Neuronal_PorteriaCero_Visitante'
df_output['prediccion'] = df_final['pred_porteria_cero_visitante']
df_output['probabilidad'] = df_final['proba_porteria_cero_visitante']
df_output['prediccion'] = df_output['prediccion'].astype(bool)

cargar_dataframe_postgresql(
    df_output,
    schema='dbo',
    tabla='predicciones_binarias',
    clave_conflicto=['temporada', 'jornada', 'equipo_local', 'equipo_visitante', 'modelo']
)
