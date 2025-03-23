from Clasificacion.obtener_clasificacion_jornada import obtener_clasificacion_jornada
from Clasificacion.obtener_clasificacion_general import obtener_clasificacion_general
from Jugadores.obtener_jugadores import obtener_jugadores
from Jugadores.obtener_datos_jugador import obtener_datos_jugador,obtener_datos_jornadas,obtener_registros_transferencia,obtener_puntos,obtener_valores
from Jugadores.obtener_mercado import obtener_mercado
from LaLiga.obtener_clasificacion_liga import obtener_datos_liga
from LaLiga.obtener_datos_jornadas_liga import obtener_datos_jornadas_liga
from Robo.robo_jugador import robo_jugador
from iniciar_sesion import iniciar_sesion
from create_db_collections import crear_base_datos_y_colecciones
from Usuarios.usuarios import insertar_datos_usuarios
from Clasificacion.clasificacion_general import insertar_datos_clasificacion_general
from Clasificacion.clasificacion_jornada import insertar_datos_jornada
from Jugadores.jugadores import insertar_jugadores
from Jugadores.mercado_jugadores import insertar_datos_mercado
from Jugadores.datos_jugador import insertar_datos_jugadores
from Jugadores.panel_principal import datos_tarjeta
from Clasificacion.insert_jornada import insertar_datos_clasificacion_jornada
from create_db_collections import crear_base_datos_y_colecciones
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import json
import re
import csv
import time
from utils import guardar_en_csv,aplanar_datos,añadir_temporada
from config import get_filename_config

usuario_jugador = [
    ("Titi10","C. Riad"),
    ("Titi10","J. Hernández"),
    ("Titi10","J. Bamba")
]

#nombre_usuario = "Megamister"
#nombre_robo = "J. Oblak"
temporada_actual = "23/24"
base_path = "./data/csv"

def main():
    # Comprobamos si existe la bbdd y las colecciones
    #crear_base_datos_y_colecciones()
    # Iniciamos sesion en MisterMundoDeportivo
    driver = iniciar_sesion()
    #drivers = []
    
    # Obtenemos los datos de las jornadas
    datos_jornadas = obtener_clasificacion_jornada(driver)
    datos_jornadas = añadir_temporada(datos_jornadas)
    filename_config_jornadas = get_filename_config("jornadas")
    datos_jornadas_lista = aplanar_datos(datos_jornadas)
    guardar_en_csv(datos_jornadas_lista, base_path, filename_config_jornadas)
    """ 
    # Insertamos en la colección los datos de las jornadas
    insertar_datos_clasificacion_jornada(temporada_actual,datos_jornadas)
    # Obtenemos los datos de los usuarios
    datos_usuarios = obtener_clasificacion_general(driver)
    # Insertamos en la colección los datos de los usuarios
    insertar_datos_usuarios(datos_usuarios)
    # Insertamos los datos de la clasificacion general
    insertar_datos_clasificacion_general(temporada_actual,datos_usuarios)
    # Obtenemos los datos del mercado
    datos_mercado = obtener_mercado(driver)
    # Insertamos los datos del mercado
    insertar_datos_mercado(temporada_actual,datos_mercado)
    #for usuario, robo in usuario_jugador:
    #    driver = iniciar_sesion()
    #    robo_jugador(driver,usuario,robo)  # Funciona 20230724
    #    drivers.append(driver)
    # Obtenemos los datos de los jugadores
    datos_jugadores = obtener_jugadores(driver)
    # Insertamos los datos de los jugadores
    insertar_jugadores(temporada_actual,datos_jugadores)
    
    # Obtenemos todos los datos para un jugador
    #datos_jugador, datos_jornadas, registros_transferencia, puntos, valores = obtener_datos_jugador(driver)
    datos_jugador = obtener_datos_jugador(driver)
    datos_jornadas = obtener_datos_jornadas(driver)
    datos_registros_transferencia = obtener_registros_transferencia(driver)
    datos_puntos = obtener_puntos(driver)
    datos_valores = obtener_valores(driver)
    # Insertamos todos los datos de cada jugador
    #insertar_datos_jugadores(temporada_actual, datos_jugador, datos_jornadas, registros_transferencia, puntos, valores)
    #la función de arriba devuelve datos_jugador, datos_jornada, transferencias, historial_puntos, historial_valores
    # datos_tarjeta(driver,temporada)  # Hay que pensar como modificarlo
    # print(datos_tarjetas)
    #obtener_datos_liga(driver) # Funciona 20240317
    #obtener_datos_jornadas_liga(driver)  # Funciona 20240317
    """
    
    driver.quit()

if __name__ == "__main__":
    main()