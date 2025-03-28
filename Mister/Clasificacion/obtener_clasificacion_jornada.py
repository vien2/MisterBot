from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, NoSuchWindowException,NoSuchElementException
from collections import defaultdict
from utils import log

def obtener_clasificacion_jornada(driver):
    log("obtener_clasificacion_jornada: Inicio de la función")

    datos_por_jornada = defaultdict(list)
    datos_jornadas = []
    wait = WebDriverWait(driver, 10)

    try:
        enlace_tabla = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Tabla')]/parent::li/a")))
        enlace_tabla.click()
        log("obtener_clasificacion_jornada: Enlace 'Tabla' clickeado")
    except Exception as e:
        log(f"obtener_clasificacion_jornada: Error al hacer clic en el enlace 'Tabla': {e}")
        return datos_por_jornada

    try:
        wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Jornada')]"))).click()
        log("obtener_clasificacion_jornada: Pestaña 'Jornada' clickeada")
    except Exception as e:
        log(f"obtener_clasificacion_jornada: Error al hacer clic en la pestaña 'Jornada': {e}")
        return datos_por_jornada

    for i in range(1, 39):  # Asumiendo 38 jornadas
        try:
            log(f"obtener_clasificacion_jornada: Procesando Jornada {i}")

            selector_jornada = f'//div[@class="top"]/select/option[contains(text(), "Jornada {i}")]'
            try:
                wait.until(EC.element_to_be_clickable((By.XPATH, selector_jornada))).click()
                log(f"obtener_clasificacion_jornada: Jornada {i} seleccionada")
            except NoSuchElementException:
                log(f"obtener_clasificacion_jornada: La Jornada {i} no se encuentra.")
                break
            except TimeoutException:
                log(f"obtener_clasificacion_jornada: Timeout al seleccionar Jornada {i}")
                break

            tab_jornada = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[@data-tab="gameweek"]')))
            if "active" not in tab_jornada.get_attribute("class"):
                tab_jornada.click()
                log(f"obtener_clasificacion_jornada: Pestaña 'Jornada' activada manualmente")

            wait.until(EC.presence_of_all_elements_located((By.XPATH, '//div[@class="panel panel-gameweek"]//li')))
            gameweek_standings = driver.find_elements(By.XPATH, '//div[@class="panel panel-gameweek"]//li')

            if gameweek_standings:
                log(f"obtener_clasificacion_jornada: {len(gameweek_standings)} elementos encontrados en Jornada {i}")

                for item in gameweek_standings:
                    position = item.find_element(By.XPATH, './/div[@class="position"]').text.strip()
                    name = item.find_element(By.XPATH, './/div[contains(@class, "name ")]').text.strip()
                    points = item.find_element(By.XPATH, './/div[@class="points"]').text.strip()
                    played_text = item.find_element(By.XPATH, './/div[contains(@class, "played")]').text.strip()

                    played_parts = played_text.split('·')
                    if len(played_parts) == 2:
                        players = played_parts[0].strip()
                        amount = played_parts[1].strip()
                        players_numbers = players.split(' ')[0]
                        numeric_value = amount.replace('.', '')
                        value_team = int(numeric_value)

                        datos_jornada = {
                            "Jornada": str(i),
                            "Nombre": name,
                            "Posicion": position,
                            "Puntos": points,
                            "Jugadores": players_numbers,
                            "Valor_equipo": value_team
                        }

                        datos_por_jornada[i].append(datos_jornada)
                        #log(f"obtener_clasificacion_jornada: Datos añadidos para {name} en Jornada {i}")
            else:
                log(f"obtener_clasificacion_jornada: No hay datos para la Jornada {i}")
        except (TimeoutException, StaleElementReferenceException, NoSuchWindowException) as e:
            log(f"obtener_clasificacion_jornada: Error procesando Jornada {i}: {e}")
            if isinstance(e, NoSuchWindowException):
                log("obtener_clasificacion_jornada: La ventana del navegador se ha cerrado.")
                break
            continue

    log("obtener_clasificacion_jornada: Finalización de la función")
    return datos_por_jornada
