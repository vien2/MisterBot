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
import json, re, csv

def obtener_clasificacion_general(driver):
  datos_usuarios = []
  wait = WebDriverWait(driver, 2)
  # Localizar el enlace de "Tabla"
  #enlace_tabla = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[contains(text(), 'Tabla')]")))
  enlace_tabla = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Tabla')]/parent::li/a")))
  enlace_tabla.click()

  #time.sleep(5000)
  driver.implicitly_wait(2)

  # Encuentra los elementos de los botones de pestañas
  tabs = driver.find_elements(By.XPATH, '//div[@class="tabs tabs-round tabs-standings"]/button')

  # Haz clic en el botón de clasificación general
  tabs[0].click()

  # Encuentra los elementos de la clasificación general
  general_standings = driver.find_elements(By.XPATH, '//div[@class="panel panel-total"]//li')

  # Imprime los datos de la clasificación general
  for item in general_standings:
    position = item.find_element(By.CLASS_NAME, 'position').text.strip()
    name = item.find_element(By.CLASS_NAME, 'name').text.strip()
    points = item.find_element(By.CLASS_NAME, 'points').text.strip()

    cadena_puntos = points.split("+")
    puntos = cadena_puntos[0].strip() if len(cadena_puntos) >= 1 else "0"
    match_puntos = re.search(r'\d+', puntos)
    if match_puntos:
        puntos = match_puntos.group()
    if "-" in points:
      cadena_puntos_negativos = points.split("-")
      puntos = cadena_puntos_negativos[0].strip() if len(cadena_puntos_negativos) >= 1 else "0"
      match_puntos = re.search(r'\d+', puntos)
      if match_puntos:
          puntos = match_puntos.group()

    jugadores_elements = item.find_elements(By.CLASS_NAME, 'played')
    cadena_jugadores = jugadores_elements[0].text.strip() if len(jugadores_elements) > 0 else "0"
    
    partes_jugadores = cadena_jugadores.split("·")
    parte_jugadores = partes_jugadores[0].strip() if len(partes_jugadores) >= 1 else "0"
    match_jugadores = re.search(r'\d+', parte_jugadores)
    if match_jugadores:
        parte_jugadores = match_jugadores.group()
    parte_valor = partes_jugadores[1].strip() if len(partes_jugadores) >= 2 else "0"
    datos_usuario = {
      "Usuario": name,
      "Posicion": position,
      "Puntos": puntos,
      "Jugadores": parte_jugadores,
      "Valor total": parte_valor
    }
    datos_usuarios.append(datos_usuario)
  return datos_usuarios
    #print(f"Posición: {position}, Nombre: {name}, Puntos: {puntos}, Jugadores: {parte_jugadores}, Valor: {parte_valor}")