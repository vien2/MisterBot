from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, NoSuchWindowException
import time

def obtener_clasificacion_jornada(driver):
    datos_jornadas = []

    # Esperar a que el enlace de la tabla esté disponible y hacer clic en él
    wait = WebDriverWait(driver, 10)
    enlace_tabla = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Tabla')]/parent::li/a")))
    enlace_tabla.click()

    # Hacer clic en la pestaña "Jornada" de la tabla
    tabs = driver.find_elements(By.XPATH, '//div[@class="tabs tabs-round tabs-standings"]/button')
    tabs[1].click()

    # Inicializar gameweek antes del bucle
    gameweek = ""

    try:
        # Buscar el selector de jornadas y sus opciones
        dropdown = driver.find_element(By.XPATH, '//div[@class="top"]/select')
        options = dropdown.find_elements(By.TAG_NAME, 'option')

        # Si no hay opciones disponibles, salir de la función
        if len(options) == 0:
            print("No hay opciones disponibles en el selector de jornadas.")
            return datos_jornadas

        # Iterar sobre las opciones del selector de jornadas
        time.sleep(3)
        for option in options:
            try:
                # Hacer clic en la opción de la jornada actual
                option.click()

                time.sleep(2)

                # Esperar a que la pestaña "Jornada" esté presente en el DOM
                jornada_tab = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, '//button[@data-tab="gameweek"]'))
                )

                # Hacer clic en la pestaña "Jornada"
                jornada_tab.click()

                # Esperar a que los elementos de la jornada se carguen completamente
                WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.XPATH, '//div[@class="panel panel-gameweek"]//li'))
                )

                # Obtener el número de la jornada actual y los elementos de la clasificación
                gameweek = option.get_attribute('value')
                gameweek_standings = driver.find_elements(By.XPATH, '//div[@class="panel panel-gameweek"]//li')

                # Si hay elementos de la clasificación, procesarlos
                if len(gameweek_standings) > 0:
                    print(f"Clasificación Jornada {gameweek}:")
                    for item in gameweek_standings:
                        position = item.find_element(By.XPATH, './/div[@class="position"]').text.strip()
                        name = item.find_element(By.XPATH, './/div[@class="name "]').text.strip()
                        points = item.find_element(By.XPATH, './/div[@class="points"]').text.strip()

                        # Agregar los datos de la jornada actual a la lista de datos
                        datos_jornada = {
                            "Jornada": gameweek,
                            "Nombre": name,
                            "Posicion": position,
                            "Puntos": points
                        }
                        datos_jornadas.append(datos_jornada)
                    print(datos_jornadas)
                else:
                    print(f"No hay datos de clasificación para la Jornada {gameweek}")

            except StaleElementReferenceException:
                # Si el elemento de la jornada se ha vuelto obsoleto, volver a la pestaña original
                print(f"Elemento de la jornada {gameweek} se ha vuelto obsoleto. Intentando de nuevo...")
                option.click()
                time.sleep(2)
                continue
            except NoSuchWindowException:
                # Si la pestaña original ya no existe, salir de la función
                print("La pestaña original ya no existe. Terminando...")
                break

    except TimeoutException:
        # Si se produce un error de tiempo de espera, salir de la función
        print("Se ha producido un error de tiempo de espera.")

    return datos_jornadas