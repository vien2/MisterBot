# main.py
from obtener_clasificacion_jornada import obtener_clasificacion_jornada
from obtener_clasificacion_general import obtener_clasificacion_general
from obtener_jugadores import obtener_jugadores
from obtener_datos_jugador import obtener_datos_jugador
from obtener_jugadores import obtener_jugadores
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
from selenium import webdriver
from bs4 import BeautifulSoup
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait,Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
import json, re, csv

nombre_usuario = "Libri"
nombre_robo = "J. Carlos Martín"
temporada = "23/24"
datos_usuarios = []
datos_jornadas =  []
datos_jugadores = []
datos_mercado = []
datos_tarjetas = []

def main():
    #cnn = connect_to_mysql()
    driver = iniciar_sesion()

    datos_jornadas = obtener_clasificacion_jornada(driver)#Funciona 20230724
    print(datos_jornadas)
    #datos_usuarios = obtener_clasificacion_general(driver)#Funciona 20230724
    #insertar_datos_usuarios(cnn,datos_usuarios)
    #insertar_datos_clasificacion_general(cnn,datos_usuarios)
    #insertar_datos_jornada(cnn,datos_jornadas)#Probar cuando existan las jornadas
    #datos_mercado = obtener_mercado(driver)#Funciona 20230724
    #insertar_datos_mercado(cnn,datos_mercado,temporada)
    #robo_jugador(driver,nombre_usuario,nombre_robo)#Funciona 20230724
    #datos_jugadores = obtener_jugadores(driver)#Funciona 20230724
    #insertar_jugadores(cnn,datos_jugadores,temporada)
    #obtener_datos_jugador(driver)#Funciona 20230925
    #datos_tarjeta(driver,temporada)#Hay que pensar como modificarlo
    #print(datos_tarjetas)
    #print("hola")
    driver.quit
    #cnn.close()

if __name__ == "__main__":
    main()