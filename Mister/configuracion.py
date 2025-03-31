# config.py
from datetime import datetime

# Diccionario base para los tipos de CSV que vas a generar
BASE_FILENAME_CONFIG = {
    "jornadas": {"nombre": "clasificacion_jornadas"},
    "usuarios": {"nombre": "clasificacion_general"},
    "mercado": {"nombre": "mercado"},
    "datos_jugador": {"nombre": "datos_jugador"},
    "datos_jornadas": {"nombre": "datos_jornadas"},
    "datos_transferencia": {"nombre": "datos_transferencia"},
    "datos_puntos" : {"nombre": "datos_puntos"},
    "datos_valores" : {"nombre": "datos_valores"},
    "datos_tarjetas" : {"nombre": "datos_tarjetas"},
    "datos_liga" : {"nombre": "datos_liga"},
    "datos_jornadas_liga" : {"nombre": "datos_jornadas_liga"},
    "url_jugadores" : {"nombre": "url_jugadores"}
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