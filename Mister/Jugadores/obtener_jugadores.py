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

def obtener_jugadores(driver):
    datos_jugadores = []

    wait = WebDriverWait(driver, 2)
    # Localizar el enlace de "Más"
    #enlace_mas = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[contains(text(), 'Más')]")))
    enlace_mas = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Más')]/parent::li/a")))
    enlace_mas.click()

    driver.implicitly_wait(2)

    enlace_jugadores = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Jugadores')]")))
    enlace_jugadores.click()

    #time.sleep(5000)
    # Hacer scroll hacia abajo en la página
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
    # Bucle para hacer clic en el botón "Ver más jugadores" hasta que no haya más jugadores nuevos
    while True:
        try:
            # Espera hasta que el botón esté visible
            button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Ver más')]")))

            # Haz clic en el botón "Ver más jugadores"
            button.click()

            # Espera a que se carguen los nuevos jugadores
            WebDriverWait(driver, 5).until(EC.invisibility_of_element_located((By.XPATH, '//div[@class="player-list"]')))
        except:
            # Si no se encuentra el botón o no hay más jugadores nuevos, sale del bucle
            break
    # Recorrer la lista de jugadores
    players = driver.find_elements(By.XPATH, '//ul[@class="player-list search-players-list"]/li')
    for player in players:
        # Obtén los detalles del jugador, como nombre, posición, etc.
        name  = player.find_element(By.CLASS_NAME, 'name').text.strip()
        position_element = player.find_element(By.XPATH, './/i[contains(@class, "pos-")]')

        # Obtén la clase completa del elemento
        position_class = position_element.get_attribute('class')

        # Extraer el número de la clase
        position_number = position_class.split('-')[-1]

        # Mapear el número a la posición correspondiente
        position_mapping = {
            '1': 'PT',
            '2': 'DF',
            '3': 'MC',
            '4': 'DL'
        }

        # Obtener la posición del jugador según el mapeo
        position = position_mapping.get(position_number, 'Desconocida')
        # Haz algo con la información del jugador
        datos_jugador = {
            "Jugador" : name,
            "Posicion" : position
        }
        datos_jugadores.append(datos_jugador)
        #print(f"Jugador: {name}, Posicion: {position}")

    driver.implicitly_wait(2)
    return datos_jugadores