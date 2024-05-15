from selenium import webdriver
from bs4 import BeautifulSoup
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException

def obtener_datos_jornadas_liga(driver):
    datos_jornadas = []
    wait = WebDriverWait(driver, 10)
    enlace_mas = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Más')]/parent::li/a")))
    enlace_mas.click()
    time.sleep(2)  # Es importante darle tiempo a la página para que cargue completamente los elementos
    
    enlace_jugadores = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'LaLiga')]")))
    enlace_jugadores.click()
    time.sleep(2)  # Espera a que la tabla de clasificación se cargue completamente
    
    # Espera a que los contenedores de las jornadas se carguen
    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".box-matches")))

    # Supongamos que cada jornada está dentro de un '.box-matches' y todas son accesibles sin clics adicionales
    jornadas_contenedores = driver.find_elements(By.CSS_SELECTOR, ".box-matches")

    jornada_numero = 1
    for contenedor in jornadas_contenedores:
        partidos = contenedor.find_elements(By.CSS_SELECTOR, "ul li")
        for partido in partidos:
            equipo_local = partido.find_element(By.CSS_SELECTOR, ".home").text
            equipo_visitante = partido.find_element(By.CSS_SELECTOR, ".away").text.strip()
            resultado = partido.find_element(By.CSS_SELECTOR, ".mid").text
            datos_jornadas.append({
                'jornada': jornada_numero,
                'local': equipo_local,
                'visitante': equipo_visitante,
                'resultado': resultado,
            })
        jornada_numero += 1  # Incrementa el número de jornada después de procesar todos los partidos de una
    print(datos_jornadas)
    return datos_jornadas