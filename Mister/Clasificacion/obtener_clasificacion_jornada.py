from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, NoSuchWindowException,NoSuchElementException
from collections import defaultdict
import json

def obtener_clasificacion_jornada(driver):
    datos_por_jornada = defaultdict(list)
    datos_jornadas = []
    wait = WebDriverWait(driver, 10)

    # Esperar a que el enlace de la tabla esté disponible y hacer clic en él
    enlace_tabla = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Tabla')]/parent::li/a")))
    enlace_tabla.click()

    # Esperar a que la pestaña "Jornada" esté disponible y hacer clic en ella
    wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Jornada')]"))).click()

    # Iterar sobre todas las jornadas
    for i in range(1, 39):  # Asumiendo que hay 38 jornadas
        try:
            # Seleccionar la jornada actual del dropdown
            selector_jornada = f'//div[@class="top"]/select/option[contains(text(), "Jornada {i}")]'
            try:
                wait.until(EC.element_to_be_clickable((By.XPATH, selector_jornada))).click()
            except NoSuchElementException:
                print(f"La Jornada {i} no se encuentra.")
                break
            except TimeoutException:
                # Si se produce un error de tiempo de espera al intentar hacer clic en la jornada, asumir que no hay más jornadas disponibles.
                print(f"No se pudo seleccionar la Jornada {i} dentro del tiempo esperado.")
                break
            # Verificar si la pestaña 'Jornada' está activa y hacer clic si no lo está
            tab_jornada = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[@data-tab="gameweek"]')))
            # Comprobar si ya está activo; si no lo está, hacer clic
            if "active" not in tab_jornada.get_attribute("class"):
                tab_jornada.click()

            # Esperar a que los elementos de la jornada se carguen completamente
            wait.until(EC.presence_of_all_elements_located((By.XPATH, '//div[@class="panel panel-gameweek"]//li')))

            # Obtener el número de la jornada actual y los elementos de la clasificación
            gameweek_standings = driver.find_elements(By.XPATH, '//div[@class="panel panel-gameweek"]//li')

            # Si hay elementos de la clasificación, procesarlos
            if len(gameweek_standings) > 0:
                for item in gameweek_standings:
                    position = item.find_element(By.XPATH, './/div[@class="position"]').text.strip()
                    name = item.find_element(By.XPATH, './/div[contains(@class, "name ")]').text.strip()
                    points = item.find_element(By.XPATH, './/div[@class="points"]').text.strip()
                    played_text = item.find_element(By.XPATH, './/div[contains(@class, "played")]').text.strip()
                    # Separa los valores '11/11 Jugadores' y '55.592.000'
                    played_parts = played_text.split('·')
                    if len(played_parts) == 2:
                        players = played_parts[0].strip()  # '11 / 11 Jugadores'
                        amount = played_parts[1].strip() 
                        players_numbers = players.split(' ')[0]  # '11/11'
                        numeric_value = amount.replace('.', '')  # '55592000'
                        # Si necesitas convertirlo a un tipo int, asegúrate de que no haya ningún otro caracter no numérico.
                        value_team = int(numeric_value)  # 55592000
                    # Agregar los datos de la jornada actual a la lista de datos
                    datos_jornada = {
                        "Jornada": str(i),
                        "Nombre": name,
                        "Posicion": position,
                        "Puntos": points,
                        "Jugadores": players_numbers,
                        "Valor_equipo": value_team
                    }
                    datos_por_jornada[i].append(datos_jornada)
            else:
                print(f"No hay datos de clasificación para la Jornada {i}")
        except (TimeoutException, StaleElementReferenceException, NoSuchWindowException) as e:
            print(f"Se ha producido un error con la Jornada {i}: {e}")
            if isinstance(e, NoSuchWindowException):
                print("La ventana del navegador se ha cerrado.")
                break
            continue  # Continuar con la siguiente jornada
    return datos_por_jornada
