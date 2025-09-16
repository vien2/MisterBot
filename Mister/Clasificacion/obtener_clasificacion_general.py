from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
from utils import log

def obtener_clasificacion_general(driver, schema=None):
    _ = schema
    log("obtener_clasificacion_general: Inicio de la función")

    datos_usuarios = []
    wait = WebDriverWait(driver, 5)

    # --- Ir a la pestaña Tabla ---
    try:
        enlace_tabla = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//ul[@class='menu']//li[@data-pag='standings']/a")
        ))
        enlace_tabla.click()
        log("obtener_clasificacion_general: Enlace 'Tabla' clickeado")
    except Exception as e:
        log(f"obtener_clasificacion_general: Error al hacer clic en 'Tabla': {e}")
        return []

    driver.implicitly_wait(2)

    # --- Pulsar botón General ---
    try:
        boton_general = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//div[@class='segments']//button[@data-tab='total']")
        ))
        if "active" not in boton_general.get_attribute("class"):
            boton_general.click()
        log("obtener_clasificacion_general: Pestaña 'General' activada")
    except Exception as e:
        log(f"obtener_clasificacion_general: Error al activar la pestaña General: {e}")
        return []


    # --- Extraer elementos de la clasificación ---
    try:
        general_standings = wait.until(
            EC.presence_of_all_elements_located((By.XPATH, '//div[@class="panel panel-total"]//li'))
        )
    except Exception as e:
        log(f"obtener_clasificacion_general: No se encontraron elementos de clasificación: {e}")
        return []

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

            # Extraer jugadores y valor total
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

