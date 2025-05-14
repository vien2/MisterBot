# config.py
from datetime import datetime
from Clasificacion.obtener_clasificacion_jornada import obtener_clasificacion_jornada
from Clasificacion.obtener_clasificacion_general import obtener_clasificacion_general
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
    "datos_best_xi" : {"nombre" : "datos_best_xi"}
}

def get_filename_config(tipo, fecha=None):
    """
    Devuelve configuración para el nombre del archivo CSV.
    Incluye 'archivo' con el nombre final: nombre_fecha.csv
    """
    config = BASE_FILENAME_CONFIG.get(tipo, {}).copy()
    config["fecha"] = fecha if fecha else datetime.now().strftime("%Y%m%d")
    
    # Añadimos la clave 'archivo' con el nombre final del CSV
    if "nombre" in config:
        config["archivo"] = f"{config['nombre']}_{config['fecha']}.csv"
    else:
        config["archivo"] = f"{tipo}_{config['fecha']}.csv"

    return config

def get_funciones_disponibles():
    return {
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
        "datos_best_xi": obtener_best_xi_jornadas_finalizadas
    }