# MisterBot - Predicción Goleador Probable (Estructura Unificada)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input

from utils import log, conexion_db
from patterns.df_postgresql import cargar_dataframe_postgresql

# ------------------------ #
# CONFIGURACIÓN
# ------------------------ #
TEMPORADAS_HIST = ['23/24', '24/25', '25/26']
TEMPORADA_OBJETIVO = '25/26'
UMBRAL_PREDICCION = 0.65

# ------------------------ #
# CONSULTAS
# ------------------------ #
def _query_proxima_jornada(temporada_objetivo: str) -> str:
    return rf"""
    WITH jornadas_status AS (
        SELECT jornada::integer,
               COUNT(*) AS partidos_totales,
               SUM(CASE WHEN resultado ~ '^\d+\s*[-·]\s*\d+$' THEN 1 ELSE 0 END) AS jugados
        FROM chavalitos.jornadas_liga
        WHERE temporada = '{temporada_objetivo}'
        GROUP BY jornada
    )
    SELECT MIN(jornada) AS proxima_jornada
    FROM jornadas_status
    WHERE (partidos_totales - jugados) > 2; 
    """

def _query_datos_jugadores(temporadas: list) -> str:
    temps_str = "', '".join(temporadas)
    # Incluimos features ofensivas clave
    return f"""
    SELECT 
        jornada, temporada, nombre, apellido, eventos,
        minutos_jugados, goles_esperados, asistencias_esperadas, 
        pases_clave, tiros_fuera, tiros_a_puerta,
        duelos_ganados, intercepciones, faltas_cometidas
    FROM chavalitos.v_datos_jornadas 
    WHERE temporada IN ('{temps_str}')
    ORDER BY temporada, jornada
    """

def _query_equipos_actuales() -> str:
    return """
    SELECT nombre, apellido, equipo
    FROM chavalitos.datos_jugador
    """

def _query_partidos_jornada(temporada: str, jornada: int) -> str:
    return f"""
    SELECT local AS equipo_local, visitante AS equipo_visitante
    FROM chavalitos.jornadas_liga
    WHERE temporada = '{temporada}' AND jornada = '{jornada}'
    """

# ------------------------ #
# MODELO
# ------------------------ #
def crear_modelo(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(64)); model.add(LeakyReLU(0.1)); model.add(Dropout(0.3))
    model.add(Dense(32)); model.add(LeakyReLU(0.1)); model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.0007), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ------------------------ #
# ETIQUETADO
# ------------------------ #
def extraer_evento(eventos, palabra):
    if pd.isna(eventos):
        return 0
    return int(palabra in eventos.lower())

# ------------------------ #
# LOGICA PRINCIPAL
# ------------------------ #
def main():
    log("Iniciando predicción Red Neuronal Goleador...")

    with conexion_db() as conn:
        # 1. Obtener próxima jornada
        try:
            df_prox = pd.read_sql(_query_proxima_jornada(TEMPORADA_OBJETIVO), conn)
            proxima_jornada = df_prox.iloc[0]['proxima_jornada']
        except Exception:
            proxima_jornada = None

        if pd.isna(proxima_jornada):
            log("⚠️ No se detectó ninguna jornada futura pendiente. Salgo.")
            return

        log(f"📅 Próxima jornada OBJETIVO: {int(proxima_jornada)}")

        # 2. Cargar Datos Históricos
        query = _query_datos_jugadores(TEMPORADAS_HIST)
        log(f"Cargando datos de jugadores... ({TEMPORADAS_HIST})")
        df = pd.read_sql(query, conn)
        
        # 3. Cargar Equipos Actuales
        log("Cargando equipos actuales...")
        df_equipos = pd.read_sql(_query_equipos_actuales(), conn)
        df_equipos = df_equipos[~df_equipos['equipo'].astype(str).str.match(r'^\d+$')]
        
        # 4. Cargar Partidos de la Jornada Objetivo
        log(f"Cargando partidos jornada {int(proxima_jornada)}...")
        df_partidos = pd.read_sql(_query_partidos_jornada(TEMPORADA_OBJETIVO, proxima_jornada), conn)
    
    if df.empty:
        log("No hay datos de jugadores.")
        return

    # Construir diccionario de rivales
    mapa_rivales = {}
    for _, row in df_partidos.iterrows():
        local = row['equipo_local']
        visitante = row['equipo_visitante']
        mapa_rivales[local] = f"vs {visitante} (L)"
        mapa_rivales[visitante] = f"vs {local} (V)"

    # Preprocesado
    df['target_goleador'] = df['eventos'].apply(lambda x: extraer_evento(x, 'gol'))
    
    features = [
        'goles_esperados', 'asistencias_esperadas', 'pases_clave', 'tiros_fuera', 'tiros_a_puerta',
        'duelos_ganados', 'intercepciones', 'faltas_cometidas', 'minutos_jugados'
    ]
    
    df[features] = df[features].fillna(0)
    df = df[df[features].notnull().all(axis=1)]

    # Split Train/Test
    X_train = df[features]
    y_train = df['target_goleador']

    # --- PREDICCION ---
    log("Generando promedios para predicción...")
    # Filtramos temporada actual para usar stats recientes como input
    df_actual = df[df['temporada'] == TEMPORADA_OBJETIVO].copy()
    
    # Calculamos medias por Jugador
    df_medias = df_actual.groupby(['nombre', 'apellido'])[features].mean().reset_index()
    
    # Unimos con Equipos Actuales
    promedios = pd.merge(df_medias, df_equipos, on=['nombre', 'apellido'], how='inner')
    
    EQUIPOS_INVALIDOS = ['Desconocido', 'void', 'Sin Equipo']
    promedios = promedios[~promedios['equipo'].isin(EQUIPOS_INVALIDOS)]
    promedios = promedios.dropna(subset=['equipo'])

    # FILTRO DE ACTIVIDAD RECIENTE (5 Jornadas)
    if not df_actual.empty:
        ultima_jornada_datos = df_actual['jornada'].max()
        umbral_actividad = ultima_jornada_datos - 5
        
        df_actividad = df_actual.groupby(['nombre', 'apellido'])['jornada'].max().reset_index()
        df_actividad.rename(columns={'jornada': 'ultima_jornada_jugada'}, inplace=True)
        
        promedios = pd.merge(promedios, df_actividad, on=['nombre', 'apellido'], how='inner')
        promedios = promedios[promedios['ultima_jornada_jugada'] >= umbral_actividad]
        
        log(f"Filtro Actividad: Se han eliminado jugadores inactivos desde la jornada {umbral_actividad}.")

    X_pred = promedios[features]
    
    # Entrenar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled = scaler.transform(X_pred)

    log("Entrenando modelo...")
    model = crear_modelo(X_train.shape[1])
    
    if set(y_train.unique()) == {0, 1}:
        w = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
        class_weight = {0: w[0], 1: w[1]}
    else:
        class_weight = None

    model.fit(X_train_scaled, y_train, epochs=40, batch_size=32, verbose=0, class_weight=class_weight)

    # Predecir
    probas = model.predict(X_pred_scaled, verbose=0).flatten()
    
    promedios['proba_goleador'] = probas
    promedios['prediccion'] = (probas >= UMBRAL_PREDICCION)
    
    # Info partido
    def get_partido_label(equipo):
        info = mapa_rivales.get(equipo)
        if not info: return "Sin Partido"
        parts = info.replace("vs ", "").rsplit(" (", 1)
        return f"{equipo} - {parts[0]}" if parts[1].replace(")", "") == 'L' else f"{parts[0]} - {equipo}"

    def get_match_column(row):
        lbl = row['partido_label']
        return lbl if lbl == "Sin Partido" else f"{lbl} (J{int(proxima_jornada)})"

    promedios['partido_label'] = promedios['equipo'].apply(get_partido_label)
    promedios['match_col'] = promedios.apply(get_match_column, axis=1)

    # --- RELATO 2: PROMEDIOS POR PARTIDO (Goles Esperados) ---
    df_con_partido = promedios[promedios['partido_label'] != "Sin Partido"]
    
    if not df_con_partido.empty:
        stats = df_con_partido.groupby('partido_label')['proba_goleador'].agg(['mean', 'sum']).reset_index()
        
        # Factor de calibración (estimado, similar a tarjetas pero ajustado para goles)
        # Podríamos empezar con un factor 1.0 y ajustar luego si vemos desviación muy grande.
        # En tarjetas fue 0.28. En goles, la sumatoria de prob_goleador de todos los jugadores suele 
        # sobreestimar el total de goles reales. Usaremos 0.35 como starting point conservador.
        FACTOR_CALIBRACION_GOL = 0.35
        stats['goles_esperados'] = stats['sum'] * FACTOR_CALIBRACION_GOL
        
        stats.rename(columns={'mean': 'prob_media'}, inplace=True)
        stats = stats.sort_values('goles_esperados', ascending=False)

        print(f"\n{'PARTIDO (Jornada ' + str(int(proxima_jornada)) + ')':<40} {'PROB. MEDIA':<15} {'GOLES ESPERADOS (Est.)'}")
        print("-" * 90)
        for _, row in stats.iterrows():
            print(f"{row['partido_label']:<40} {row['prob_media']:.2%}           {row['goles_esperados']:.2f}")
        print("-" * 90 + "\n")

    # --- RELATO 3: JUGADORES TOP ---
    df_resultados = promedios[promedios['prediccion'] == True].sort_values(by='proba_goleador', ascending=False)
    log(f"Jugadores con ALTA probabilidad de gol (>{UMBRAL_PREDICCION:.0%}): {len(df_resultados)}")

    # --- GUARDADO EN BBDD (OPCIÓN A: JUGADORES) ---
    log("Guardando predicciones de JUGADORES en dbo.predicciones_goleadores...")
    
    df_db_players = df_resultados[df_resultados['match_col'] != "Sin Partido"].copy()
    
    if df_db_players.empty:
        log("⚠️ No hay goleadores probables para guardar.")
    else:
        df_db_players['temporada'] = TEMPORADA_OBJETIVO
        df_db_players['jornada'] = int(proxima_jornada)
        df_db_players['jugador'] = df_db_players['nombre'] + ' ' + df_db_players['apellido']
        df_db_players['partido'] = df_db_players['match_col']
        df_db_players['probabilidad'] = df_db_players['proba_goleador']
        
        df_db_players = df_db_players[['temporada', 'jornada', 'jugador', 'equipo', 'partido', 'probabilidad']]
        df_db_players = df_db_players.drop_duplicates(subset=['temporada', 'jornada', 'jugador'])
        
        try:
            cargar_dataframe_postgresql(
                df_db_players,
                schema='dbo',
                tabla='predicciones_goleadores',
                clave_conflicto=['temporada', 'jornada', 'jugador']
            )
            log(f"✅ [Jugadores] {len(df_db_players)} registros guardados/actualizados.")
        except Exception as e:
            log(f"❌ [Jugadores] Error al guardar: {e}")

    # --- GUARDADO EN BBDD (OPCIÓN B: PARTIDOS) ---
    log("Guardando resumen de GOLES en dbo.predicciones_jornada...")
    
    if not df_con_partido.empty:
        df_db_matches = stats.copy()
        
        # Parseamos partido_label
        def split_teams(label):
            parts = label.split(' - ')
            if len(parts) == 2:
                return pd.Series([parts[0].strip(), parts[1].strip()])
            return pd.Series([None, None])

        df_db_matches[['equipo_local', 'equipo_visitante']] = df_db_matches['partido_label'].apply(split_teams)
        
        df_db_matches['temporada'] = TEMPORADA_OBJETIVO
        df_db_matches['jornada'] = int(proxima_jornada)
        df_db_matches['modelo'] = 'Neuronal_Goleador'
        # Usamos promedio de prob de gol como "confianza" genérica del modelo para ese partido
        df_db_matches['confianza'] = df_db_matches['prob_media']
        df_db_matches['prediccion_1x2'] = '-'
        df_db_matches['tarjetas_esperadas'] = None
        
        cols_match = ['temporada', 'jornada', 'equipo_local', 'equipo_visitante', 'modelo', 'confianza', 'goles_esperados', 'prediccion_1x2']
        df_db_matches = df_db_matches[cols_match]
        
        try:
            cargar_dataframe_postgresql(
                df_db_matches,
                schema='dbo',
                tabla='predicciones_jornada',
                clave_conflicto=['temporada', 'jornada', 'equipo_local', 'equipo_visitante', 'modelo']
            )
            log("✅ [Partidos] Guardado OK.")
        except Exception as e:
            log(f"❌ [Partidos] Error al guardar: {e}")

    log("Red_Neuronal_Goleador: fin OK")
