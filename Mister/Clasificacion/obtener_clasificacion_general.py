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
        enlace_tabla = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Tabla')]/parent::li/a")
        ))
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

            # Extraer puntos
            points_element = item.find_element(By.CLASS_NAME, 'points')
            points_text = points_element.get_attribute("textContent").strip()
            puntos_match = re.search(r"\d[\d.]*", points_text)
            puntos = puntos_match.group(0).replace(".", "") if puntos_match else "0"

            # Extraer jugado y valor usando textContent
            played_element = item.find_element(By.CLASS_NAME, 'played')
            played_text = played_element.get_attribute("textContent").strip()
            partes = [x.strip() for x in played_text.split("·")]
            jugadores = re.search(r"\d+", partes[0]).group() if partes else "0"
            valor_total = partes[1].replace(".", "") if len(partes) > 1 else "0"

            datos_usuario = {
                "usuario": name,
                "posicion": position,
                "puntos": puntos,
                "jugadores": jugadores,
                "valor_total": valor_total
            }

            datos_usuarios.append(datos_usuario)
        except Exception as e:
            log(f"obtener_clasificacion_general: Error procesando un elemento de la clasificación: {e}")

    log("obtener_clasificacion_general: Finalización exitosa de la función")
    return datos_usuarios
