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
    "datos_jornadas_liga" : {"nombre": "datos_jornadas_liga"}
}

def get_filename_config(tipo, fecha=None):
    """
    Devuelve un diccionario de configuración para el nombre del archivo CSV,
    según el tipo de datos y la temporada.
    
    :param tipo: Clave para identificar el tipo de CSV (ej. "jornadas", "usuarios", "mercado").
    :param temporada: Valor de la temporada que se incluirá en el nombre.
    :param fecha: Fecha en formato string (opcional). Si no se proporciona, se usa la fecha actual.
    :return: Diccionario de configuración, por ejemplo:
             {"nombre": "clasificacion_jornadas", "seccion": "2025", "fecha": "20250323"}
    """
    config = BASE_FILENAME_CONFIG.get(tipo, {}).copy()  # Obtener la configuración base y evitar modificar el original
    # Si no se proporciona la fecha, se genera automáticamente
    config["fecha"] = fecha if fecha is not None else datetime.now().strftime("%Y%m%d")
    return config
