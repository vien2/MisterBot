# MisterBot - Modelo Neuronal 1X2 con distribución de clases

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical
from utils import conexion_db
from patterns.df_postgresql import cargar_dataframe_postgresql

# ------------------------ #
# CONFIGURACIÓN
# ------------------------ #
temporadas_hist = ['23/24', '24/25', '25/26']
temporada_objetivo = '25/26'

# ------------------------ #
# CONSULTAS
# ------------------------ #
def query_partidos(temporada):
    return f"""
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
        END AS resultado_1x2,
        jl.temporada
    FROM chavalitos.v_jornadas_liga jl
    LEFT JOIN chavalitos.v_datos_laliga el ON el.equipo = jl.equipo_local AND el.temporada = jl.temporada
    LEFT JOIN chavalitos.v_datos_laliga ev ON ev.equipo = jl.equipo_visitante AND ev.temporada = jl.temporada
    WHERE jl.temporada = '{temporada}'
    ORDER BY jl.jornada ASC
    """

query_proxima_jornada = f"""
WITH jornadas AS (
    SELECT jornada::integer,
           COUNT(*) AS partidos,
           SUM(CASE WHEN resultado ~ '^\d+\s*[-·]\s*\d+$' THEN 1 ELSE 0 END) AS jugados
    FROM chavalitos.jornadas_liga
    WHERE temporada = '{temporada_objetivo}'
    GROUP BY jornada
)
SELECT MIN(jornada) AS proxima_jornada
FROM jornadas
WHERE jugados < partidos;
"""

# ------------------------ #
# LECTURA DE DATOS
# ------------------------ #
with conexion_db() as conn:
    dfs = []
    for temp in temporadas_hist:
        dfs.append(pd.read_sql(query_partidos(temp), conn))
    df_partidos = pd.concat(dfs, ignore_index=True)
    proxima_jornada = pd.read_sql(query_proxima_jornada, conn).iloc[0]['proxima_jornada']

if pd.isna(proxima_jornada):
    print("⚠️ No hay jornada futura para predecir.")
    exit()

# ------------------------ #
# FEATURES
# ------------------------ #
df_partidos['ratio_victorias_local'] = df_partidos['ganados_local'] / df_partidos['partidos_jugados_local']
df_partidos['ratio_victorias_visitante'] = df_partidos['ganados_visitante'] / df_partidos['partidos_jugados_visitante']
df_partidos['ppp_local'] = df_partidos['puntos_local'] / df_partidos['partidos_jugados_local']
df_partidos['ppp_visitante'] = df_partidos['puntos_visitante'] / df_partidos['partidos_jugados_visitante']
df_partidos['abs_dif_goles_local'] = df_partidos['dif_goles_local'].abs()
df_partidos['abs_dif_goles_visitante'] = df_partidos['dif_goles_visitante'].abs()

features = [col for col in df_partidos.columns if any(x in col for x in [
    'posicion_', 'puntos_', 'ganados_', 'empatados_', 'perdidos_', 'dif_goles_', 
    'ratio_', 'ppp_', 'abs_dif_'
])]

# ------------------------ #
# SPLIT TRAIN / TEST
# ------------------------ #
df_entrenamiento = df_partidos[(df_partidos['temporada'] != temporada_objetivo) | 
                               (df_partidos['jornada'] < proxima_jornada)].copy()
df_pred = df_partidos[(df_partidos['temporada'] == temporada_objetivo) & 
                      (df_partidos['jornada'] == proxima_jornada) & 
                      df_partidos['goles_local'].isna()].copy()

X_train = df_entrenamiento[features].replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = df_pred[features].replace([np.inf, -np.inf], np.nan).fillna(0)

y_train = df_entrenamiento['resultado_1x2'].map({'1':0,'X':1,'2':2}).astype(int)
y_train_cat = to_categorical(y_train, num_classes=3)

# ------------------------ #
# NORMALIZACIÓN
# ------------------------ #
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------ #
# PESOS DE CLASE + TEMPORADA
# ------------------------ #
class_weights = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_train)
cw_dict = {i: w for i, w in enumerate(class_weights)}

sample_weights = np.ones(len(df_entrenamiento))
sample_weights[df_entrenamiento['temporada'] == temporada_objetivo] *= 4.0

# ------------------------ #
# MODELO
# ------------------------ #
def crear_modelo(input_dim, output_dim=3):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(128))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.3))
    model.add(Dense(64))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0007), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = crear_modelo(X_train.shape[1])
model.fit(X_train_scaled, y_train_cat, epochs=40, batch_size=16, verbose=0,
          class_weight=cw_dict, sample_weight=sample_weights)

# ------------------------ #
# PREDICCIONES
# ------------------------ #
probas = model.predict(X_test_scaled, verbose=0)
preds = probas.argmax(axis=1)
mapa = {0:'1',1:'X',2:'2'}

df_pred['prediccion_1x2'] = [mapa[p] for p in preds]
df_pred['confianza'] = probas.max(axis=1)

print(f"\n🔮 Predicciones jornada {proxima_jornada} ({temporada_objetivo}):")
for i,row in df_pred.iterrows():
    print(f" - {row['equipo_local']} vs {row['equipo_visitante']}: {row['prediccion_1x2']} (confianza: {row['confianza']:.2%})")

df_output = df_pred[['temporada', 'jornada', 'equipo_local', 'equipo_visitante']].copy()
df_output['modelo'] = 'Neuronal_1X2_Softmax'   # nombre del modelo
df_output['prediccion_1x2'] = df_pred['prediccion_1x2']
df_output['confianza'] = df_pred['confianza']

cargar_dataframe_postgresql(
    df_output,
    schema='dbo',
    tabla='predicciones_jornada',
    clave_conflicto=['temporada', 'jornada', 'equipo_local', 'equipo_visitante', 'modelo']
)

# ------------------------ #
# DISTRIBUCIÓN
# ------------------------ #
# Distribución real en entrenamiento
dist_real = df_entrenamiento['resultado_1x2'].value_counts(normalize=True).rename_axis('resultado').reset_index(name='proporcion')
print("\n📊 Distribución real en entrenamiento:")
print(dist_real)

# Distribución de predicciones en jornada futura
dist_pred = df_pred['prediccion_1x2'].value_counts(normalize=True).rename_axis('prediccion').reset_index(name='proporcion')
print("\n📊 Distribución de predicciones en jornada futura:")
print(dist_pred)
