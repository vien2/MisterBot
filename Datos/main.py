from obtener_clasificacion_jornada import obtener_clasificacion_jornada
from obtener_clasificacion_general import obtener_clasificacion_general
from obtener_jugadores import obtener_jugadores
from obtener_datos_jugador import obtener_datos_jugador
from obtener_mercado import obtener_mercado
from robo_jugador import robo_jugador
from iniciar_sesion import iniciar_sesion
from connection import connect_to_mysql
from usuarios import insertar_datos_usuarios
from clasificacion_general import insertar_datos_clasificacion_general
from clasificacion_jornada import insertar_datos_jornada
from jugadores import insertar_jugadores
from mercado_jugadores import insertar_datos_mercado
from panel_principal import datos_tarjeta
from insert_jornada import insertar_datos_clasificacion_jornada
from grafico_jornada import grafico_jornada
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

nombre_usuario = "Libri"
nombre_robo = "J. Carlos Martín"
temporada = "23/24"

def main():
    #cnn = connect_to_mysql()
    driver = iniciar_sesion()

    #datos_jornadas = obtener_clasificacion_jornada(driver)  #Funciona 20230130
    #print(datos_jornadas)
    #insertar_datos_clasificacion_jornada(cnn)
    #grafico_jornada(cnn)
    #datos_usuarios = obtener_clasificacion_general(driver)  # Funciona 20230130
    #print(datos_usuarios)
    #insertar_datos_usuarios(cnn,datos_usuarios)
    #insertar_datos_clasificacion_general(cnn,datos_usuarios)
    #insertar_datos_jornada(cnn,datos_jornadas)  # Probar cuando existan las jornadas
    #datos_mercado = obtener_mercado(driver)  # Funciona 20240227
    #print(datos_mercado)
    # insertar_datos_mercado(cnn,datos_mercado,temporada)
    # robo_jugador(driver,nombre_usuario,nombre_robo)  # Funciona 20230724
    #datos_jugadores = obtener_jugadores(driver)  # Funciona 20230130
    #print(datos_jugadores)
    # insertar_jugadores(cnn,datos_jugadores,temporada)
    #obtener_datos_jugador(driver)  # Funciona 20230130
    #la función de arriba devuelve datos_jugador, datos_jornada, transferencias, historial_puntos, historial_valores
    # datos_tarjeta(driver,temporada)  # Hay que pensar como modificarlo
    # print(datos_tarjetas)
    # print("hola")

    driver.quit()
    #cnn.close()

if __name__ == "__main__":
    main()