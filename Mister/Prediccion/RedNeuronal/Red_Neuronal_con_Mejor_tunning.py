# MisterBot - Walk-forward sistema 1X2 con datos de jugadores

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from sklearn.calibration import calibration_curve
from utils import conexion_db
import matplotlib.pyplot as plt
import seaborn as sns
from patterns.df_postgresql import cargar_dataframe_postgresql

# ------------------------ #
# CONFIGURACIÓN
# ------------------------ #
temporada_objetivo = '24/25'
threshold_1 = 0.42
threshold_2 = 0.46

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
    jl.goles_visitante,
    CASE
        WHEN jl.goles_local > jl.goles_visitante THEN '1'
        WHEN jl.goles_local = jl.goles_visitante THEN 'X'
        ELSE '2'
    END AS resultado_1x2
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

    # Modelo 1 vs No1
    y_train_1 = (df_train['resultado_1x2'] == '1').astype(int)
    y_test_true = df_test['resultado_1x2']
    model_1 = crear_modelo(X_train.shape[1])
    w1 = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train_1)
    model_1.fit(X_train_scaled, y_train_1, epochs=40, batch_size=16, verbose=0, class_weight={0: w1[0], 1: w1[1]})
    proba_1 = model_1.predict(X_test_scaled, verbose=0).flatten()

    # Modelo X vs 2
    y_train_x2 = df_train[df_train['resultado_1x2'].isin(['X', '2'])].copy()
    y_train_x2['target_x2'] = (y_train_x2['resultado_1x2'] == '2').astype(int)
    model_2 = crear_modelo(X_train.shape[1])
    w2 = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train_x2['target_x2'])
    y_train_bin = df_train['resultado_1x2'].map({'X': 0, '2': 1}).fillna(0).astype(int)
    model_2.fit(X_train_scaled, y_train_bin, epochs=40, batch_size=16, verbose=0, class_weight={0: w2[0], 1: w2[1]})
    proba_2 = model_2.predict(X_test_scaled, verbose=0).flatten()

    pred = []
    for i in range(len(proba_1)):
        if proba_1[i] >= threshold_1:
            pred.append('1')
        else:
            pred.append('2' if proba_2[i] >= threshold_2 else 'X')

    df_test = df_test.copy()
    df_test['pred_1x2'] = pred
    df_test['proba_1'] = proba_1
    df_test['proba_2'] = proba_2
    df_test['temporada'] = temporada_objetivo
    resultados.append(df_test)

# ------------------------ #
# Evaluación final
# ------------------------ #
df_final = pd.concat(resultados, ignore_index=True)
acc = accuracy_score(df_final['resultado_1x2'], df_final['pred_1x2'])
print(f"\n✅ Precisión global final 1X2 (walk-forward): {acc:.2%}")

print("\n📊 Clasificación por clase:")
print(classification_report(df_final['resultado_1x2'], df_final['pred_1x2'], digits=4))

cm = confusion_matrix(df_final['resultado_1x2'], df_final['pred_1x2'], labels=['1', 'X', '2'])
sns.heatmap(cm, annot=True, fmt='d', cmap='PuBu', xticklabels=['1', 'X', '2'], yticklabels=['1', 'X', '2'])
plt.title("Matriz de confusión final (1X2) - Walk Forward")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# ------------------------ #
# 6. Guardar en PostgreSQL
# ------------------------ #

# Asegúrate de tener estas columnas disponibles en df_final
df_output = df_final[[
    'temporada', 'jornada', 'equipo_local', 'equipo_visitante'
]].copy()

df_output['modelo'] = 'Neuronal_1X2'
df_output['prediccion_1x2'] = df_final['pred_1x2']
df_output['confianza'] = df_final[['proba_1', 'proba_2']].max(axis=1)

cargar_dataframe_postgresql(
    df_output,
    schema='dbo',
    tabla='predicciones_jornada',
    clave_conflicto=['temporada', 'jornada', 'equipo_local', 'equipo_visitante', 'modelo']
)

# ------------------------ #
# Calibración de probabilidad (1X2)
# ------------------------ #
print("\n📈 Curva de calibración - Predicción '1'")
y_true = (df_final['resultado_1x2'] == '1').astype(int)
y_proba = df_final['proba_1']

fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_proba, n_bins=10, strategy='uniform')

plt.figure(figsize=(6, 6))
plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label='Modelo (1 vs No1)')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfecto')
plt.xlabel('Probabilidad media predicha (clase 1)')
plt.ylabel('Fracción de positivos reales')
plt.title('Curva de calibración - Clase 1 (1X2)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()