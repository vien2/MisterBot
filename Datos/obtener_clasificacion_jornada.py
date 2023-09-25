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

def obtener_clasificacion_jornada(driver):
    datos_jornadas = []

    wait = WebDriverWait(driver, 10)
    enlace_tabla = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Tabla')]/parent::li/a")))
    enlace_tabla.click()
    # Encuentra los elementos de los botones de pestañas
    tabs = driver.find_elements(By.XPATH, '//div[@class="tabs tabs-round tabs-standings"]/button')

    # Haz clic en el botón de clasificación por jornada
    tabs[1].click()

    try:
        # Encuentra el elemento del selector de jornada
        dropdown = driver.find_element(By.XPATH, '//div[@class="top"]/select')
        # Obtén las opciones del selector de jornada
        options = dropdown.find_elements(By.TAG_NAME, 'option')

        # Verifica si hay suficientes opciones en el selector
        if len(options) == 0:
            print("No hay opciones disponibles en el selector de jornadas.")
            return datos_jornadas

        # Itera sobre las opciones (jornadas)
        for option in options:
            gameweek = option.get_attribute('value')
            option.click()

            # Vuelve a buscar el elemento del selector de jornada y las opciones
            dropdown = driver.find_element(By.XPATH, '//div[@class="top"]/select')
            options = dropdown.find_elements(By.TAG_NAME, 'option')

            # Encuentra los elementos de la clasificación por jornada
            wait = WebDriverWait(driver, 2)
            wait.until(EC.presence_of_element_located((By.XPATH, '//div[@class="panel panel-gameweek"]//li')))
            gameweek_standings = driver.find_elements(By.XPATH, '//div[@class="panel panel-gameweek"]//li')
            if len(gameweek_standings) > 0:
                # Imprime los datos de la clasificación por jornada
                print(f"Clasificación Jornada {gameweek}:")
                for item in gameweek_standings:
                    position = item.find_element(By.CLASS_NAME, 'position').text.strip()
                    name = item.find_element(By.CLASS_NAME, 'name').text.strip()
                    points = item.find_element(By.CLASS_NAME, 'points').text.strip()
                    datos_jornada = {
                        "Jornada": gameweek,
                        "Nombre": name,
                        "Posicion": position,
                        "Puntos": points
                    }
                    datos_jornadas.append(datos_jornada)
            else:
                print(f"No hay datos de clasificación para la Jornada {gameweek}")

    except TimeoutException:
        print("Se ha producido un error de tiempo de espera.")

    return datos_jornadas