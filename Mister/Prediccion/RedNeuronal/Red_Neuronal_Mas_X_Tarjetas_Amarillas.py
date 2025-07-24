# MisterBot - Walk-forward predicción Over 4.5 Tarjetas Totales

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
umbral_prediccion = 0.65

# ------------------------ #
# Cargar datos
# ------------------------ #
query = f"""
SELECT
    jl.jornada,
    jl.equipo_local,
    jl.equipo_visitante
FROM chavalitos.v_jornadas_liga jl
WHERE jl.temporada = '{temporada_objetivo}'
ORDER BY jl.jornada ASC
"""

query_jugadores = f"SELECT * FROM chavalitos.v_datos_jornadas WHERE temporada = '{temporada_objetivo}'"
query_equipos_jugador = f"SELECT DISTINCT id_jugador, equipo FROM chavalitos.v_datos_jugador WHERE temporada = '{temporada_objetivo}'"

with conexion_db() as conn:
    df = pd.read_sql(query, conn)
    df_jugadores = pd.read_sql(query_jugadores, conn)
    df_equipos = pd.read_sql(query_equipos_jugador, conn)

# ------------------------ #
# Agregados de tarjetas amarillas
# ------------------------ #
def extraer_evento(eventos, palabra):
    if pd.isna(eventos):
        return 0
    return int(palabra in eventos.lower())

df_jugadores['tarjeta_amarilla'] = df_jugadores['eventos'].apply(lambda x: extraer_evento(x, 'tarjeta amarilla'))

agg_tarjetas = (df_jugadores
    .merge(df_equipos, on='id_jugador', how='left')
    .groupby(['jornada', 'equipo'])
    .agg({'tarjeta_amarilla': 'sum'})
    .reset_index())

local = agg_tarjetas.rename(columns={'equipo': 'equipo_local', 'tarjeta_amarilla': 'tarjetas_local'})
visit = agg_tarjetas.rename(columns={'equipo': 'equipo_visitante', 'tarjeta_amarilla': 'tarjetas_visitante'})

df = df.merge(local, on=['jornada', 'equipo_local'], how='left')
df = df.merge(visit, on=['jornada', 'equipo_visitante'], how='left')

# ------------------------ #
# Feature engineering
# ------------------------ #
df['tarjetas_totales'] = df['tarjetas_local'] + df['tarjetas_visitante']
df['target_over45_tarjetas'] = (df['tarjetas_totales'] > 4.5).astype(int)

features = ['tarjetas_local', 'tarjetas_visitante']

# ------------------------ #
# Modelo base
# ------------------------ #
def crear_modelo(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(64))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.3))
    model.add(Dense(32))
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

    X_train = df_train[features].fillna(0)
    X_test = df_test[features].fillna(0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train = df_train['target_over45_tarjetas']
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
    df_test['pred_over45_tarjetas'] = pred
    df_test['proba_over45_tarjetas'] = proba
    df_test['temporada'] = temporada_objetivo
    resultados.append(df_test)

# ------------------------ #
# Evaluación final
# ------------------------ #
df_final = pd.concat(resultados, ignore_index=True)
acc = accuracy_score(df_final['target_over45_tarjetas'], df_final['pred_over45_tarjetas'])
print(f"\n✅ Precisión global Over 4.5 Tarjetas (walk-forward): {acc:.2%}")

print("\n📊 Clasificación:")
print(classification_report(df_final['target_over45_tarjetas'], df_final['pred_over45_tarjetas'], digits=4))

cm = confusion_matrix(df_final['target_over45_tarjetas'], df_final['pred_over45_tarjetas'], labels=[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['<=4', '>4'], yticklabels=['<=4', '>4'])
plt.title("Matriz de confusión final - Over 4.5 Tarjetas")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# ------------------------ #
# Curva de calibración
# ------------------------ #
y_true = df_final['target_over45_tarjetas']
y_prob = df_final['proba_over45_tarjetas']
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

plt.plot(prob_pred, prob_true, marker='o', label='Modelo')
plt.plot([0, 1], [0, 1], linestyle='--', color='orange', label='Perfecto')
plt.title("Curva de calibración - Over 4.5 Tarjetas")
plt.xlabel("Probabilidad media predicha")
plt.ylabel("Fracción de positivos reales")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------ #
# Recomendaciones alta confianza
# ------------------------ #
print("\n🎯 Partidos recomendados para apostar por Over 4.5 tarjetas (confianza >= 0.80):")
recomendados = df_final[(df_final['pred_over45_tarjetas'] == 1) & (df_final['proba_over45_tarjetas'] >= 0.8)]
print(recomendados[['jornada', 'equipo_local', 'equipo_visitante', 'proba_over45_tarjetas']].sort_values(by='proba_over45_tarjetas', ascending=False))

# ------------------------ #
# Guardado en PostgreSQL
# ------------------------ #

df_output = df_final[[
    'temporada', 'jornada', 'equipo_local', 'equipo_visitante'
]].copy()

df_output['modelo'] = 'Neuronal_Over45_Tarjetas'
df_output['prediccion'] = df_final['pred_over45_tarjetas'].astype(bool)
df_output['probabilidad'] = df_final['proba_over45_tarjetas']

cargar_dataframe_postgresql(
    df_output,
    schema='dbo',
    tabla='predicciones_binarias',
    clave_conflicto=['temporada', 'jornada', 'equipo_local', 'equipo_visitante', 'modelo']
)
