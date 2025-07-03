
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from utils import log

def obtener_datos_jornadas_liga(driver,schema=None):
    _ = schema
    log("obtener_datos_jornadas_liga: Inicio de la función")

    datos_jornadas = []
    wait = WebDriverWait(driver, 10)

    try:
        enlace_mas = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Más')]/parent::li/a")))
        enlace_mas.click()
        log("obtener_datos_jornadas_liga: Enlace 'Más' clickeado")
        time.sleep(2)

        enlace_laliga = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//button[contains(text(), 'LaLiga')]")))
        enlace_laliga.click()
        log("obtener_datos_jornadas_liga: Botón 'LaLiga' clickeado")
        time.sleep(2)

        wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".box-matches")))
        jornadas_contenedores = driver.find_elements(By.CSS_SELECTOR, ".box-matches")
        log(f"obtener_datos_jornadas_liga: {len(jornadas_contenedores)} jornadas encontradas")

        jornada_numero = 1
        for contenedor in jornadas_contenedores:
            partidos = contenedor.find_elements(By.CSS_SELECTOR, "ul li")
            log(f"obtener_datos_jornadas_liga: Jornada {jornada_numero} - {len(partidos)} partidos encontrados")

            for partido in partidos:
                try:
                    equipo_local = partido.find_element(By.CSS_SELECTOR, ".home").text
                    equipo_visitante = partido.find_element(By.CSS_SELECTOR, ".away").text.strip()
                    resultado = partido.find_element(By.CSS_SELECTOR, ".mid").text

                    registro = {
                        'jornada': jornada_numero,
                        'local': equipo_local,
                        'visitante': equipo_visitante,
                        'resultado': resultado,
                    }

                    datos_jornadas.append(registro)
                    #log(f"obtener_datos_jornadas_liga: Jornada {jornada_numero} - {equipo_local} vs {equipo_visitante} -> {resultado}")
                except Exception as e:
                    log(f"obtener_datos_jornadas_liga: Error procesando partido en Jornada {jornada_numero}: {e}")
                    continue

            jornada_numero += 1

        log(f"obtener_datos_jornadas_liga: Finalización con {len(datos_jornadas)} partidos procesados")
        return datos_jornadas

    except TimeoutException:
        log("obtener_datos_jornadas_liga: Timeout al cargar los elementos de la página")
        return []
    except NoSuchElementException:
        log("obtener_datos_jornadas_liga: No se encontró un elemento esperado")
        return []
    except Exception as e:
        log(f"obtener_datos_jornadas_liga: Error inesperado: {e}")
        return []
