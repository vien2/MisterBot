from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from utils import log
import time
import re

def obtener_best_xi_jornadas_finalizadas(driver):
    log("obtener_best_xi_jornadas_finalizadas: Inicio")
    datos = []
    wait = WebDriverWait(driver, 10)

    try:
        icono = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "gameweek__icon")))
        icono.click()
        log("obtener_best_xi_jornadas_finalizadas: Icono de jornadas clickeado")
    except Exception as e:
        log(f"obtener_best_xi_jornadas_finalizadas: Error al hacer clic en el icono: {e}")
        return []

    try:
        # Inicialmente capturamos cuántas jornadas hay
        select_element = wait.until(EC.presence_of_element_located((By.ID, "sw-top-gw")))
        select = Select(select_element)
        total_jornadas = len(select.options)
        log(f"obtener_best_xi_jornadas_finalizadas: {total_jornadas} jornadas encontradas")

        for i in range(total_jornadas):
            try:
                # Volver a obtener el select (evita stale element)
                select_element = wait.until(EC.presence_of_element_located((By.ID, "sw-top-gw")))
                select = Select(select_element)
                opcion = select.options[i]
                jornada_valor = opcion.get_attribute("value")
                jornada_texto = opcion.text.strip()

                select.select_by_value(jornada_valor)
                log(f"obtener_best_xi_jornadas_finalizadas: Jornada '{jornada_texto}' seleccionada (valor {jornada_valor})")

                time.sleep(2)

                # Confirmar si es jornada finalizada
                if not driver.find_elements(By.XPATH, "//h3[contains(text(), 'Finalizada')]"):
                    continue

                jugadores = driver.find_elements(By.CSS_SELECTOR, ".lineup-best-xi a.lineup-player")
                for jugador in jugadores:
                    try:
                        url = jugador.get_attribute("href")
                        puntos = jugador.find_element(By.CLASS_NAME, "points").get_attribute("data-points")
                        match = re.search(r'/players/(\d+)/([\w\-]+)', url)
                        if match:
                            id_jugador = match.group(1)
                            nombre = match.group(2).replace("-", " ").title()
                            datos.append({
                                "id_jugador": id_jugador,
                                "nombre": nombre,
                                "url": url,
                                "puntos": puntos,
                                "jornada": jornada_texto
                            })
                    except Exception as e:
                        log(f"Error al procesar jugador en '{jornada_texto}': {e}")
                        continue
            except Exception as e:
                log(f"Error al procesar jornada '{jornada_texto}': {e}")
                continue

    except Exception as e:
        log(f"obtener_best_xi_jornadas_finalizadas: Error general: {e}")

    log(f"obtener_best_xi_jornadas_finalizadas: Total registros capturados: {len(datos)}")
    return datos