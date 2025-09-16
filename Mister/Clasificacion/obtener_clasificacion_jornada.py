from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, NoSuchWindowException,NoSuchElementException
from collections import defaultdict
from utils import log
import re,time

def obtener_clasificacion_jornada(driver, schema=None):
    _ = schema
    log("obtener_clasificacion_jornada: Inicio de la función")

    datos_jornadas = []
    wait = WebDriverWait(driver, 10)

    # --- Ir a la pestaña Tabla ---
    try:
        enlace_tabla = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//ul[@class='menu']//li[@data-pag='standings']/a")
        ))
        enlace_tabla.click()
        log("obtener_clasificacion_jornada: Enlace 'Tabla' clickeado")
    except Exception as e:
        log(f"obtener_clasificacion_jornada: Error al hacer clic en el enlace 'Tabla': {e}")
        return []

    # --- Activar pestaña Jornada ---
    try:
        boton_jornada = wait.until(EC.presence_of_element_located(
            (By.XPATH, "//div[@class='segments']//button[@data-tab='gameweek']")
        ))

        if "active" not in boton_jornada.get_attribute("class"):
            boton_jornada.click()
            log("obtener_clasificacion_jornada: Pestaña 'Jornada' clickeada")
        else:
            log("obtener_clasificacion_jornada: Pestaña 'Jornada' ya estaba activa")
    except Exception as e:
        log(f"obtener_clasificacion_jornada: Error al hacer clic en la pestaña 'Jornada': {e}")
        return []

    # --- Obtener las jornadas disponibles ---
    enlaces = wait.until(EC.presence_of_all_elements_located(
        (By.CSS_SELECTOR, "div.gameweek-selector-inline a")
    ))
    jornadas_disponibles = [e.text.strip() for e in enlaces if e.text.strip().startswith("J")]

    log(f"obtener_clasificacion_jornada: Jornadas disponibles detectadas: {jornadas_disponibles}")

    for jornada_text in jornadas_disponibles:
        try:
            jornada_num = re.search(r"J(\d+)", jornada_text).group(1)

            # volver a buscar el enlace en cada iteración (DOM cambia al recargar)
            selector = f"//div[@class='gameweek-selector-inline']//a[contains(text(),'{jornada_text}')]"
            enlace = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
            enlace.click()
            log(f"obtener_clasificacion_jornada: Jornada {jornada_num} seleccionada")
            time.sleep(0.5)

            wait.until(EC.presence_of_all_elements_located(
                (By.XPATH, '//div[@class="panel panel-gameweek"]//li')
            ))
            jugadores = driver.find_elements(By.XPATH, '//div[@class="panel panel-gameweek"]//li')

            for item in jugadores:
                try:
                    position = item.find_element(By.CLASS_NAME, 'position').text.strip()
                    name = item.find_element(By.CLASS_NAME, 'name').text.strip()
                    points = item.find_element(By.CLASS_NAME, 'points').text.strip()
                    played_text = item.find_element(By.CLASS_NAME, 'played').text.strip()

                    # En jornada: "6 / 11 Jugadores"
                    jugadores_num = re.search(r"(\d+)", played_text).group() if played_text else "0"

                    datos_jornadas.append({
                        "jornada": jornada_num,
                        "nombre": name,
                        "posicion": position,
                        "puntos": points,
                        "jugadores": jugadores_num,
                        "valor_equipo": 0  # en jornada no aparece valor en €
                    })
                except Exception as e:
                    log(f"obtener_clasificacion_jornada: Error procesando jugador en Jornada {jornada_num}: {e}")

        except Exception as e:
            log(f"obtener_clasificacion_jornada: Error procesando {jornada_text}: {e}")
            continue

    log("obtener_clasificacion_jornada: Finalización de la función")
    return datos_jornadas
