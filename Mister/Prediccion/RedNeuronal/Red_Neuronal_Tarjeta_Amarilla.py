# MisterBot - Predicción Tarjeta Amarilla (Estructura Unificada)
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
    # Misma lógica que en 1X2: buscar la primera jornada con > 2 partidos pendientes
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
    # Cargar datos de múltiples temporadas
    temps_str = "', '".join(temporadas)
    # v_datos_jornadas NO tiene id_jugador ni equipo, usamos nombre/apellido
    return f"""
    SELECT 
        jornada, temporada, nombre, apellido, eventos,
        minutos_jugados, faltas_cometidas, entradas_totales, posesiones_perdidas,
        duelos_ganados, intercepciones, regates_totales
    FROM chavalitos.v_datos_jornadas 
    WHERE temporada IN ('{temps_str}')
    ORDER BY temporada, jornada
    """

def _query_equipos_actuales() -> str:
    # Obtener el equipo actual de cada jugador para saber contra quién juega
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
    log("Iniciando predicción Red Neuronal Tarjeta Amarilla...")

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
        # Limpieza: Eliminar equipos que sean IDs numéricos (dirty data en DB)
        df_equipos = df_equipos[~df_equipos['equipo'].astype(str).str.match(r'^\d+$')]
        
        # 4. Cargar Partidos de la Jornada Objetivo
        log(f"Cargando partidos jornada {int(proxima_jornada)}...")
        df_partidos = pd.read_sql(_query_partidos_jornada(TEMPORADA_OBJETIVO, proxima_jornada), conn)
    
    if df.empty:
        log("No hay datos de jugadores.")
        return

    # Construir diccionario de rivales: {Equipo: Rival} para la jornada objetivo
    mapa_rivales = {}
    for _, row in df_partidos.iterrows():
        local = row['equipo_local']
        visitante = row['equipo_visitante']
        mapa_rivales[local] = f"vs {visitante} (L)"
        mapa_rivales[visitante] = f"vs {local} (V)"

    # Preprocesado
    df['target_amarilla'] = df['eventos'].apply(lambda x: extraer_evento(x, 'tarjeta amarilla'))
    
    features = [
        'minutos_jugados', 'faltas_cometidas', 'entradas_totales', 'posesiones_perdidas',
        'duelos_ganados', 'intercepciones', 'regates_totales'
    ]
    
    df[features] = df[features].fillna(0)
    df = df[df[features].notnull().all(axis=1)]

    # Split Train/Test
    X_train = df[features]
    y_train = df['target_amarilla']

    # --- PREDICCION ---
    log("Generando promedios para predicción...")
    # Filtramos temporada actual
    df_actual = df[df['temporada'] == TEMPORADA_OBJETIVO].copy()
    
    # Calculamos medias por Jugador (Nombre+Apellido)
    df_medias = df_actual.groupby(['nombre', 'apellido'])[features].mean().reset_index()
    
    # Unimos con Equipos Actuales
    # INNER JOIN estricto
    promedios = pd.merge(df_medias, df_equipos, on=['nombre', 'apellido'], how='inner')
    
    # Limpiamos equipos inválidos explícitamente y duplicados
    EQUIPOS_INVALIDOS = ['Desconocido', 'void', 'Sin Equipo']
    promedios = promedios[~promedios['equipo'].isin(EQUIPOS_INVALIDOS)]
    promedios = promedios.dropna(subset=['equipo'])

    # FILTRO DE ACTIVIDAD RECIENTE
    # Para evitar predecir jugadores que ya no están en la liga, filtramos aquellos
    # que no hayan jugado ninguna de las últimas 5 jornadas DISPUTADAS en los datos cargados.
    if not df_actual.empty:
        ultima_jornada_datos = df_actual['jornada'].max()
        umbral_actividad = ultima_jornada_datos - 5
        
        # Obtenemos la última jornada jugada por cada jugador
        df_actividad = df_actual.groupby(['nombre', 'apellido'])['jornada'].max().reset_index()
        df_actividad.rename(columns={'jornada': 'ultima_jornada_jugada'}, inplace=True)
        
        # Unimos y filtramos
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
    
    promedios['proba_amarilla'] = probas
    promedios['prediccion'] = (probas >= UMBRAL_PREDICCION)
    
    # Añadir info de rival y definir el partido
    def get_partido_label(equipo):
        info = mapa_rivales.get(equipo)
        if not info: return "Sin Partido"
        
        # info es "vs Rival (L)" o "vs Rival (V)"
        parts = info.replace("vs ", "").rsplit(" (", 1)
        rival_name = parts[0]
        side = parts[1].replace(")", "") # L o V
        
        if side == 'L':
            return f"{equipo} - {rival_name}"
        else:
            return f"{rival_name} - {equipo}"

    # Nueva columna solicitada: Local - Visitante Jornada
    def get_match_column(row):
        lbl = row['partido_label']
        if lbl == "Sin Partido": return lbl
        return f"{lbl} (J{int(proxima_jornada)})"

    promedios['rival_info'] = promedios['equipo'].map(mapa_rivales).fillna("Sin partido/Descansa")
    promedios['partido_label'] = promedios['equipo'].apply(get_partido_label)
    promedios['match_col'] = promedios.apply(get_match_column, axis=1)

    # --- RELATO 1: RANKING POR EQUIPOS (CONFIANZA MEDIA) ---
    # Agrupamos por equipo para ver cuáles son los más propensos a tarjeta esta jornada
    df_equipos_stats = promedios.groupby('equipo')['proba_amarilla'].mean().reset_index()
    df_equipos_stats.rename(columns={'proba_amarilla': 'confianza_media'}, inplace=True)
    df_equipos_stats = df_equipos_stats.sort_values('confianza_media', ascending=False)
    
    # Filtramos equipos que no tienen partido (opcional, pero mejor limpiar)
    equipos_con_partido = promedios[promedios['partido_label'] != "Sin Partido"]['equipo'].unique()
    df_equipos_stats = df_equipos_stats[df_equipos_stats['equipo'].isin(equipos_con_partido)]

    #print(f"\n{'EQUIPO':<25} {'CONFIANZA (PROB. MEDIA)':<25}")
    #print("-" * 50)
    #for _, row in df_equipos_stats.iterrows():
    #    print(f"{row['equipo']:<25} {row['confianza_media']:.2%}")
    #print("-" * 50 + "\n")

    # --- RELATO 2: PROMEDIOS POR PARTIDO ---
    df_con_partido = promedios[promedios['partido_label'] != "Sin Partido"]
    
    if not df_con_partido.empty:
        stats = df_con_partido.groupby('partido_label')['proba_amarilla'].agg(['mean', 'sum']).reset_index()
        
        FACTOR_CALIBRACION = 0.28 
        stats['tarjetas_esperadas'] = stats['sum'] * FACTOR_CALIBRACION
        
        stats.rename(columns={'mean': 'prob_media'}, inplace=True)
        stats = stats.sort_values('prob_media', ascending=False)

        print(f"\n{'PARTIDO (Jornada ' + str(int(proxima_jornada)) + ')':<40} {'PROB. MEDIA':<15} {'AMARILLAS ESPERADAS (Est.)'}")
        print("-" * 90)
        for _, row in stats.iterrows():
            print(f"{row['partido_label']:<40} {row['prob_media']:.2%}           {row['tarjetas_esperadas']:.2f}")
        print("-" * 90 + "\n")

    # --- RELATO 3: JUGADORES (TOP) ---
    df_resultados = promedios[promedios['prediccion'] == True].sort_values(by='proba_amarilla', ascending=False)

    log(f"Jugadores con ALTA probabilidad de amarilla (>{UMBRAL_PREDICCION:.0%}): {len(df_resultados)}")
    
    # Entendemos "Confianza" del jugador como su probabilidad calculada
    #print(f"\n{'JUGADOR':<30} {'EQUIPO':<20} {'LOCAL - VISITANTE JORNADA':<40} {'CONFIANZA'}")
    #print("-" * 105)
    #for _, row in df_resultados.iterrows():
    #    nombre_completo = f"{row['nombre']} {row['apellido']}"
    #    print(f"{nombre_completo:<30} {row['equipo']:<20} {row['match_col']:<40} {row['proba_amarilla']:.2%}")

    # --- GUARDADO EN BBDD (OPCIÓN A: JUGADORES) ---
    log("Guardando predicciones de JUGADORES en dbo.predicciones_tarjetas...")
    
    # Filtramos para guardar solo lo que el usuario ve: ALTA PROBABILIDAD + CON PARTIDO
    df_db_players = df_resultados[df_resultados['match_col'] != "Sin Partido"].copy()
    
    if df_db_players.empty:
        log("⚠️ No hay jugadores con alta probabilidad y partido para guardar.")
    else:
        df_db_players['temporada'] = TEMPORADA_OBJETIVO
        df_db_players['jornada'] = int(proxima_jornada)
        df_db_players['jugador'] = df_db_players['nombre'] + ' ' + df_db_players['apellido']
        df_db_players['partido'] = df_db_players['match_col']
        df_db_players['probabilidad'] = df_db_players['proba_amarilla']
        
        df_db_players = df_db_players[['temporada', 'jornada', 'jugador', 'equipo', 'partido', 'probabilidad']]
        # Deduplicar para evitar fallos en el UPSERT (por si acaso)
        df_db_players = df_db_players.drop_duplicates(subset=['temporada', 'jornada', 'jugador'])
        
        try:
            cargar_dataframe_postgresql(
                df_db_players,
                schema='dbo',
                tabla='predicciones_tarjetas',
                clave_conflicto=['temporada', 'jornada', 'jugador']
            )
            log(f"✅ [Jugadores] {len(df_db_players)} registros guardados/actualizados.")
        except Exception as e:
            log(f"❌ [Jugadores] Error al guardar: {e}")

    # --- GUARDADO EN BBDD (OPCIÓN B: PARTIDOS) ---
    log("Guardando resumen de PARTIDOS en dbo.predicciones_jornada...")
    
    if not df_con_partido.empty:
        # Reutilizamos 'stats' calculado para el print
        df_db_matches = stats.copy()
        
        # Parseamos partido_label "Local - Visitante"
        def split_teams(label):
            parts = label.split(' - ')
            if len(parts) == 2:
                return pd.Series([parts[0].strip(), parts[1].strip()])
            return pd.Series([None, None])

        df_db_matches[['equipo_local', 'equipo_visitante']] = df_db_matches['partido_label'].apply(split_teams)
        
        df_db_matches['temporada'] = TEMPORADA_OBJETIVO
        df_db_matches['jornada'] = int(proxima_jornada)
        df_db_matches['modelo'] = 'Neuronal_Amarillas'
        df_db_matches['confianza'] = df_db_matches['prob_media']
        # El 1X2 es obligatorio en la tabla, marcamos que es un modelo de amarillas
        df_db_matches['prediccion_1x2'] = '-' 
        df_db_matches['goles_esperados'] = None
        
        cols_match = ['temporada', 'jornada', 'equipo_local', 'equipo_visitante', 'modelo', 'confianza', 'tarjetas_esperadas', 'prediccion_1x2']
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

    log("Red_Neuronal_Tarjeta_Amarilla: fin OK")