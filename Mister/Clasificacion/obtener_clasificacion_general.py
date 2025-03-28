from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
from utils import log

def obtener_clasificacion_general(driver):
    log("obtener_clasificacion_general: Inicio de la función")

    datos_usuarios = []
    wait = WebDriverWait(driver, 2)

    try:
        enlace_tabla = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Tabla')]/parent::li/a")))
        enlace_tabla.click()
        log("obtener_clasificacion_general: Enlace 'Tabla' clickeado")
    except Exception as e:
        log(f"obtener_clasificacion_general: Error al hacer clic en 'Tabla': {e}")
        return []

    driver.implicitly_wait(2)

    try:
        tabs = driver.find_elements(By.XPATH, '//div[@class="tabs tabs-round tabs-standings"]/button')
        tabs[0].click()
        log("obtener_clasificacion_general: Pestaña 'Clasificación general' clickeada")
    except Exception as e:
        log(f"obtener_clasificacion_general: Error al hacer clic en la pestaña de clasificación: {e}")
        return []

    general_standings = driver.find_elements(By.XPATH, '//div[@class="panel panel-total"]//li')
    log(f"obtener_clasificacion_general: {len(general_standings)} elementos encontrados en la clasificación")

    for item in general_standings:
        try:
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
            #log(f"obtener_clasificacion_general: Usuario procesado - {datos_usuario}")
        except Exception as e:
            log(f"obtener_clasificacion_general: Error procesando un elemento de la clasificación: {e}")

    log("obtener_clasificacion_general: Finalización exitosa de la función")
    return datos_usuarios