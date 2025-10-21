# config.py
from datetime import datetime
from Clasificacion.obtener_clasificacion_jornada import obtener_clasificacion_jornada
from Clasificacion.obtener_clasificacion_general import obtener_clasificacion_general
from Clasificacion.obtener_saldos import obtener_saldos
from Jugadores.obtener_datos_jugador import (
    obtener_datos_jugador, obtener_registros_transferencia,
    obtener_puntos, obtener_valores, obtener_datos_jornadas,
    obtener_urls_jugadores,obtener_datos_jornadas_inicial
)
from Jugadores.obtener_mercado import obtener_mercado
from Jugadores.panel_principal import datos_tarjeta
from LaLiga.obtener_clasificacion_liga import obtener_datos_liga
from LaLiga.obtener_datos_jornadas_liga import obtener_datos_jornadas_liga
from Jugadores.obtener_once_ideal import obtener_best_xi_jornadas_finalizadas
from Jugadores.poner_mercado_todo import poner_en_venta_equipo,poner_mercado_todo
from PostProceso.extraccion_datos_historicos import (
    api_football_data_co_uk_jornadas,
    api_football_data_co_uk_datos_laliga,
    api_football_data_co_uk_jornadas_raw,
    api_football_data_co_uk_datos_laliga_raw
)
from PostProceso.backup_bbdd import backup_bbdd
from PostProceso.cleanup_archivos import cleanup_archivos
from Prediccion.RedNeuronal.Red_Neuronal_1X2_con_softmax import post_neuronal_1x2
from Robo.robo_jugador import robo_jugador

# Diccionario base para los tipos de CSV que vas a generar
BASE_FILENAME_CONFIG = {
    "clasificacion_jornadas": {"nombre": "clasificacion_jornadas"},
    "clasificacion_general": {"nombre": "clasificacion_general"},
    "mercado": {"nombre": "mercado"},
    "datos_jugador": {"nombre": "datos_jugador"},
    "datos_jornadas": {"nombre": "datos_jornadas"},
    "datos_transferencia": {"nombre": "datos_transferencia"},
    "datos_puntos" : {"nombre": "datos_puntos"},
    "datos_valores" : {"nombre": "datos_valores"},
    "datos_tarjetas" : {"nombre": "datos_tarjetas"},
    "datos_laliga" : {"nombre": "datos_laliga"},
    "datos_jornadas_liga" : {"nombre": "datos_jornadas_liga"},
    "url_jugadores" : {"nombre": "url_jugadores"},
    "datos_jornadas_inicial" : {"nombre" : "datos_jornadas_inicial"},
    "datos_best_xi" : {"nombre" : "datos_best_xi"},
    "datos_saldos" : {"nombre" : "datos_saldos"}
}

def get_filename_config(tipo, fecha=None, schema=None):
    """
    Devuelve configuración para el nombre del archivo CSV.
    Incluye 'archivo' con el nombre final: schema_nombre_fecha.csv
    """
    config = BASE_FILENAME_CONFIG.get(tipo, {}).copy()
    config["fecha"] = fecha if fecha else datetime.now().strftime("%Y%m%d")
    
    base_nombre = config.get("nombre", tipo)
    
    if schema:
        config["archivo"] = f"{schema}_{base_nombre}_{config['fecha']}.csv"
    else:
        config["archivo"] = f"{base_nombre}_{config['fecha']}.csv"

    return config

def get_funciones_disponibles():
    return {
        # Extracción y carga
        "url_jugadores": obtener_urls_jugadores,
        "clasificacion_general": obtener_clasificacion_general,
        "clasificacion_jornadas": obtener_clasificacion_jornada,
        "mercado": obtener_mercado,
        "datos_jugador": obtener_datos_jugador,
        "datos_transferencia": obtener_registros_transferencia,
        "datos_puntos": obtener_puntos,
        "datos_valores": obtener_valores,
        "datos_tarjetas": datos_tarjeta,
        "datos_laliga": obtener_datos_liga,
        "datos_jornadas_liga": obtener_datos_jornadas_liga,
        "datos_jornadas": obtener_datos_jornadas,
        "datos_jornadas_inicial": obtener_datos_jornadas_inicial,
        "datos_best_xi": obtener_best_xi_jornadas_finalizadas,
        "datos_saldos": obtener_saldos,
        # PostProceso
        "api_football_data_co_uk_jornadas": api_football_data_co_uk_jornadas,
        "api_football_data_co_uk_datos_laliga": api_football_data_co_uk_datos_laliga,
        "api_football_data_co_uk_jornadas_raw":api_football_data_co_uk_jornadas_raw,
        "api_football_data_co_uk_datos_laliga_raw":api_football_data_co_uk_datos_laliga_raw,
        "backup_bbdd":backup_bbdd,
        "cleanup_archivos": cleanup_archivos,
        "red_neuronal_softmax_1X2": post_neuronal_1x2,
        "robo_jugador": robo_jugador,
        # Accion
        "poner_mercado_todo" : poner_en_venta_equipo,
        "poner_mercado_todo" : poner_mercado_todo
    }