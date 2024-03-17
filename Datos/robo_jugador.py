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

def robo_jugador(driver,nombre_usuario,nombre_robo):
    wait = WebDriverWait(driver, 2)
    # Localizar el enlace de "Tabla"
    enlace_tabla = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Tabla')]/parent::li/a")))
    enlace_tabla.click()

    driver.implicitly_wait(2)

    # Encuentra los elementos de los botones de pestañas
    tabs = driver.find_elements(By.XPATH, '//div[@class="tabs tabs-round tabs-standings"]/button')

    # Haz clic en el botón de clasificación general
    tabs[0].click()

    # Encuentra los elementos de la clasificación general
    general_standings = driver.find_elements(By.XPATH, '//div[@class="panel panel-total"]//li')

    encontrado = False

    for item in general_standings:
        name = item.find_element(By.CLASS_NAME, 'name').text.strip()
        if name == nombre_usuario:
            player_url = item.find_element(By.TAG_NAME, 'a')
            player_url.click()
            equipo = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Equipo')]")))
            equipo.click()
            try:
                item_robo = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//div[@class="wrapper sw-profile"]//li')))
                for item in item_robo:
                    try:
                        name_robo = item.find_element(By.CLASS_NAME, 'name').text.strip()
                        if name_robo == nombre_robo:
                            encontrado = True
                            player_robo_url = item.find_element(By.TAG_NAME, 'a')
                            player_robo_url.click()
                            calusulazo = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Pagar cláusula')]")))
                            calusulazo.click()
                            calusulazo_pagar = wait.until(EC.element_to_be_clickable((By.ID, "btn-send")))
                            calusulazo_pagar.click()
                            break
                    except NoSuchElementException:
                        # Manejar el caso en que no se encuentra el elemento
                        print("Elemento 'name' no encontrado en el ítem.")
                        break
            except TimeoutException:
                # Manejar el caso en que no se encuentran elementos 'li' en 'item_robo'
                print("No se encontraron elementos 'li' en 'item_robo'")
                break


    if encontrado:
        print(f"{nombre_usuario} ha robado al jugador {nombre_robo}.")
    else:
        print(f"{nombre_usuario} no tiene al jugador {nombre_robo}.")
    return print("Robo completado")