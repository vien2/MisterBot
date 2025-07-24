# MisterBot - Walk-forward predicción Goleador Probable

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
# Cargar datos de jugadores
# ------------------------ #
query_jugadores = f"SELECT * FROM chavalitos.v_datos_jornadas WHERE temporada = '{temporada_objetivo}'"

with conexion_db() as conn:
    df_jugadores = pd.read_sql(query_jugadores, conn)

# ------------------------ #
# Etiquetado del objetivo
# ------------------------ #
def extraer_evento(eventos, palabra):
    if pd.isna(eventos):
        return 0
    return int(palabra in eventos.lower())

df_jugadores['target_goleador'] = df_jugadores['eventos'].apply(lambda x: extraer_evento(x, 'gol'))

# ------------------------ #
# Feature engineering
# ------------------------ #
features = [
    'goles_esperados', 'asistencias_esperadas', 'pases_clave', 'tiros_fuera', 'tiros_a_puerta',
    'duelos_ganados', 'intercepciones', 'faltas_cometidas', 'minutos_jugados'
]

df_jugadores[features] = df_jugadores[features].fillna(0)
df_jugadores = df_jugadores[df_jugadores[features].notnull().all(axis=1)]

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
# Entrenamiento walk-forward por jornada
# ------------------------ #
resultados = []
for jornada in sorted(df_jugadores['jornada'].unique()):
    df_train = df_jugadores[df_jugadores['jornada'] < jornada]
    df_test = df_jugadores[df_jugadores['jornada'] == jornada]
    if df_train.empty or df_test.empty:
        continue

    X_train = df_train[features].fillna(0)
    X_test = df_test[features].fillna(0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train = df_train['target_goleador']
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
    df_test['pred_goleador'] = pred
    df_test['proba_goleador'] = proba
    df_test['temporada'] = temporada_objetivo
    resultados.append(df_test)

# ------------------------ #
# Evaluación final
# ------------------------ #
df_final = pd.concat(resultados, ignore_index=True)
acc = accuracy_score(df_final['target_goleador'], df_final['pred_goleador'])
print(f"\n✅ Precisión global Goleador Probable (walk-forward): {acc:.2%}")

print("\n📊 Clasificación:")
print(classification_report(df_final['target_goleador'], df_final['pred_goleador'], digits=4))

cm = confusion_matrix(df_final['target_goleador'], df_final['pred_goleador'], labels=[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd', xticklabels=['No Gol', 'Gol'], yticklabels=['No Gol', 'Gol'])
plt.title("Matriz de confusión final - Goleador Probable")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# ------------------------ #
# Curva de calibración
# ------------------------ #
y_true = df_final['target_goleador']
y_prob = df_final['proba_goleador']
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

plt.plot(prob_pred, prob_true, marker='o', label='Modelo')
plt.plot([0, 1], [0, 1], linestyle='--', color='orange', label='Perfecto')
plt.title("Curva de calibración - Goleador Probable")
plt.xlabel("Probabilidad media predicha")
plt.ylabel("Fracción de goles reales")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------ #
# Recomendaciones alta confianza
# ------------------------ #
print("\n🎯 Jugadores recomendados (probabilidad gol >= 0.80):")
recomendados = df_final[(df_final['pred_goleador'] == 1) & (df_final['proba_goleador'] >= 0.8)]
print(recomendados[['jornada', 'nombre', 'apellido', 'proba_goleador']].sort_values(by='proba_goleador', ascending=False))

# ------------------------ #
# Guardado en PostgreSQL
# ------------------------ #

df_output = df_final[['temporada', 'jornada', 'id_jugador', 'nombre', 'apellido']].copy()
df_output['modelo'] = 'Neuronal_Goleador_Probable'
df_output['prediccion'] = df_final['pred_goleador'].astype(bool)
df_output['probabilidad'] = df_final['proba_goleador']

cargar_dataframe_postgresql(
    df_output,
    schema='dbo',
    tabla='predicciones_jugador',
    clave_conflicto=['temporada', 'jornada', 'id_jugador', 'modelo']
)
