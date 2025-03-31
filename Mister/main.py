from Clasificacion.obtener_clasificacion_jornada import obtener_clasificacion_jornada
from Clasificacion.obtener_clasificacion_general import obtener_clasificacion_general
from Jugadores.obtener_datos_jugador import obtener_datos_jugador,obtener_registros_transferencia,obtener_puntos,obtener_valores,obtener_datos_jornadas,obtener_urls_jugadores
from Jugadores.datos_jugador import cargar_urls_jugadores_csv
from Jugadores.obtener_mercado import obtener_mercado
from Jugadores.panel_principal import datos_tarjeta
from LaLiga.obtener_clasificacion_liga import obtener_datos_liga
from LaLiga.obtener_datos_jornadas_liga import obtener_datos_jornadas_liga
from Robo.robo_jugador import robo_jugador
from iniciar_sesion import iniciar_sesion
from utils import guardar_en_csv,aplanar_datos,añadir_temporada
from configuracion import get_filename_config
from utils import log,añadir_hash,añadir_f_carga
import os
from patterns.csv_postgresql import cargar_csv_postgresql

usuario_jugador = [
    ("Titi10","C. Riad"),
    ("Titi10","J. Hernández"),
    ("Titi10","J. Bamba")
]

#nombre_usuario = "Megamister"
#nombre_robo = "J. Oblak"
base_path = "./data/csv"

def main():
    log("Iniciamos sesión")
    driver = iniciar_sesion()

    datos_url_jugadores = obtener_urls_jugadores(driver)
    datos_url_jugadores = añadir_temporada(datos_url_jugadores)
    datos_url_jugadores = añadir_f_carga(datos_url_jugadores)
    datos_url_jugadores = añadir_hash(datos_url_jugadores)
    filename_config_datos_url_jugadores = get_filename_config("url_jugadores")
    datos_url_jugadores_lista = aplanar_datos(datos_url_jugadores)
    guardar_en_csv(datos_url_jugadores_lista, base_path, filename_config_datos_url_jugadores)
    ruta_completa_csv = os.path.join(base_path, filename_config_datos_url_jugadores["archivo"])
    cargar_csv_postgresql(
        ruta_csv=ruta_completa_csv,
        schema="dbo",
        tabla="urls_jugadores",
        tipo_carga="total",
        incremental_field="id_jugador",
        clave_conflicto=["id_jugador", "temporada"],
        hash_field="hash"
    )
    """
    datos_usuarios = obtener_clasificacion_general(driver)
    datos_usuarios = añadir_temporada(datos_usuarios)
    filename_config_usuarios = get_filename_config("usuarios")
    datos_usuarios_lista = aplanar_datos(datos_usuarios)
    guardar_en_csv(datos_usuarios_lista, base_path, filename_config_usuarios)

    datos_jornadas = obtener_clasificacion_jornada(driver)
    datos_jornadas = añadir_temporada(datos_jornadas)
    filename_config_jornadas = get_filename_config("jornadas")
    datos_jornadas_lista = aplanar_datos(datos_jornadas)
    guardar_en_csv(datos_jornadas_lista, base_path, filename_config_jornadas)
    
    datos_mercado = obtener_mercado(driver)
    datos_mercado = añadir_temporada(datos_mercado)
    filename_config_mercado = get_filename_config("mercado")
    datos_mercado_lista = aplanar_datos(datos_mercado)
    guardar_en_csv(datos_mercado_lista, base_path, filename_config_mercado)
    
    datos_jugador = obtener_datos_jugador(driver)
    datos_jugador = añadir_temporada(datos_jugador)
    filename_config_datos_jugador = get_filename_config("datos_jugador")
    datos_jugador_lista = aplanar_datos(datos_jugador)
    guardar_en_csv(datos_jugador_lista, base_path, filename_config_datos_jugador)
    """
    """
    datos_jornadas = obtener_datos_jornadas(driver)
    datos_jornadas = añadir_temporada(datos_jornadas)
    filename_config_datos_jornadas = get_filename_config("datos_jornadas")
    datos_jornadas_lista = aplanar_datos(datos_jornadas)
    guardar_en_csv(datos_jornadas_lista, base_path, filename_config_datos_jornadas)
    """
    """
    datos_registros_transferencia = obtener_registros_transferencia(driver)
    datos_registros_transferencia = añadir_temporada(datos_registros_transferencia)
    filename_config_datos_transferencia = get_filename_config("datos_transferencia")
    datos_transferencia_lista = aplanar_datos(datos_registros_transferencia)
    guardar_en_csv(datos_transferencia_lista, base_path, filename_config_datos_transferencia)

    datos_puntos = obtener_puntos(driver)
    datos_puntos = añadir_temporada(datos_puntos)
    filename_config_datos_puntos = get_filename_config("datos_puntos")
    datos_puntos_lista = aplanar_datos(datos_puntos)
    guardar_en_csv(datos_puntos_lista, base_path, filename_config_datos_puntos)
    
    datos_valores = obtener_valores(driver)
    datos_valores = añadir_temporada(datos_valores)
    filename_config_datos_valores = get_filename_config("datos_valores")
    datos_valores_lista = aplanar_datos(datos_valores)
    guardar_en_csv(datos_valores_lista, base_path, filename_config_datos_valores)
    """
    """
    #No termina de convencerme
    datos_tarjetas = datos_tarjeta(driver)  # Hay que pensar como modificarlo
    datos_tarjetas = añadir_temporada(datos_tarjetas)
    filename_config_datos_tarjetas = get_filename_config("datos_tarjetas")
    datos_tarjetas_lista = aplanar_datos(datos_tarjetas)
    guardar_en_csv(datos_tarjetas_lista, base_path, filename_config_datos_tarjetas)
    """
    """
    datos_liga = obtener_datos_liga(driver)
    datos_liga = añadir_temporada(datos_liga)
    filename_config_datos_liga = get_filename_config("datos_liga")
    datos_liga_lista = aplanar_datos(datos_liga)
    guardar_en_csv(datos_liga_lista, base_path, filename_config_datos_liga)
    
    datos_jornadas_liga = obtener_datos_jornadas_liga(driver)
    datos_jornadas_liga = añadir_temporada(datos_jornadas_liga)
    filename_config_datos_jornadas_liga = get_filename_config("datos_jornadas_liga")
    datos_jornadas_liga_lista = aplanar_datos(datos_jornadas_liga)
    guardar_en_csv(datos_jornadas_liga_lista, base_path, filename_config_datos_jornadas_liga)
    
    """
    log("Quitamos el driver")
    driver.quit()

if __name__ == "__main__":
    main()