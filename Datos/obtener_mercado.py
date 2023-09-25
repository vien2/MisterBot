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

def obtener_mercado(driver):
    datos_mercado = []
    wait = WebDriverWait(driver, 10)  # Aumentamos el tiempo de espera a 10 segundos

    # Localizar el enlace de "Mercado"
    enlace_mercado = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Mercado')]/parent::li/a")))
    enlace_mercado.click()

    filtrar = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Filtrar')]")))
    filtrar.click()

    libres = wait.until(EC.element_to_be_clickable((By.XPATH, "//label[contains(text(), 'Libres')]")))
    libres.click()

    aplicar = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Aplicar')]")))
    aplicar.click()

    lista_mercado = driver.find_elements(By.XPATH, '//ul[@class="player-list list-sale search-players-list"]//li')

    # Obtén el contenido HTML de la página con BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    lista_jugadores = soup.select('ul.player-list.list-sale.search-players-list > li')

    for jugador in lista_jugadores:
        nombre = jugador.select_one('div.info > div.name').text.strip()
        puntuacion = jugador.select_one('div.points').text.strip()
        precio = jugador.select_one('button.btn-popup.btn-bid.btn-grey').text.strip()
        puntuacion_media = jugador.select_one('div.avg').text.strip()

        datos_jugador = {
            'Nombre': nombre,
            'Puntuacion': puntuacion,
            'Precio': precio,
            'Puntuacion_media': puntuacion_media,
        }

        datos_mercado.append(datos_jugador)
    
    return datos_mercado