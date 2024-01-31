from selenium import webdriver
from bs4 import BeautifulSoup
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait,Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException,StaleElementReferenceException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import json, re, csv

def obtener_datos_jugador(driver):
    datos_de_jugadores = []
    evento = []
    wait = WebDriverWait(driver, 2)
    # Inicializa una lista para almacenar los eventos de gol por jornada
    eventos_gol_por_jornada = []
    # Localizar el enlace de "Más"
    #enlace_mas = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[contains(text(), 'Más')]")))
    """
    enlace_mas = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Más')]/parent::li/a")))
    enlace_mas.click()

    driver.implicitly_wait(2)

    enlace_jugadores = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Jugadores')]")))
    enlace_jugadores.click()

    #time.sleep(5000)
    """
    enlace_mas = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Más')]/parent::li/a")))
    enlace_mas.click()

    driver.implicitly_wait(2)

    enlace_jugadores = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Jugadores')]")))
    enlace_jugadores.click()
    # Hacer scroll hacia abajo en la página
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
    # Bucle para hacer clic en el botón "Ver más jugadores" hasta que no haya más jugadores nuevos
    while True:
        try:
            # Espera hasta que el botón esté visible
            button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Ver más')]")))

            # Haz clic en el botón "Ver más jugadores"
            button.click()

            # Espera a que se carguen los nuevos jugadores
            WebDriverWait(driver, 1).until(EC.invisibility_of_element_located((By.XPATH, '//div[@class="player-list"]')))
        except:
            # Si no se encuentra el botón o no hay más jugadores nuevos, sale del bucle
            break
    # Recorrer la lista de jugadores
    players = driver.find_elements(By.XPATH, '//ul[@class="player-list search-players-list"]/li')
    url_jugadores = []
    for player in players:
        player_link = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'btn.btn-sw-link.player')))
        player_url = player.find_element(By.TAG_NAME, 'a').get_attribute('href')
        url_jugadores.append(player_url)
    for player_url in url_jugadores:
        driver.get(player_url)
        name = driver.find_element(By.XPATH, '//div[@class="left"]//div[@class="name"]').text
        surname  = driver.find_element(By.CLASS_NAME, 'surname').text.strip()

        # Obtener el elemento de posición directamente desde el driver actual
        position_element = driver.find_element(By.XPATH, '//div[@class="team-position"]/i[contains(@class, "pos-")]')

        # Obtén la clase completa del elemento
        position_class = position_element.get_attribute('class')

        # Extraer el número de la clase
        position_number = re.search(r'pos-(\d+)', position_class).group(1)

        # Mapear el número a la posición correspondiente
        position_mapping = {
            '1': 'PT',
            '2': 'DF',
            '3': 'MC',
            '4': 'DL'
        }

        # Obtener la posición del jugador según el mapeo
        position = position_mapping.get(position_number, 'Desconocida')

        print(f"Nombre: {name}, Apellido: {surname}, Posición: {position}")
        # Encontrar el elemento de la sección de estadísticas del jugador
        stats_wrapper = driver.find_element(By.CLASS_NAME, 'player-stats-wrapper')

        # Encontrar todos los elementos de tipo "item" dentro de la sección de estadísticas
        stats_items = stats_wrapper.find_elements(By.CLASS_NAME, 'item')

        # Crear un diccionario para almacenar las etiquetas y valores
        stats_dict = {}

        # Recorrer los elementos de estadísticas y almacenar la información en el diccionario
        for item in stats_items:
            label = item.find_element(By.CLASS_NAME, 'label').text
            value = item.find_element(By.CLASS_NAME, 'value').text
            stats_dict[label] = value

        valor = stats_dict.get('Valor')
        claúsula = stats_dict.get('Cláusula', 'Sin cláusula')
        puntos = stats_dict.get('Puntos')
        media = stats_dict.get('Media')
        partidos = stats_dict.get('Partidos')
        goles = stats_dict.get('Goles')
        tarjetas = stats_dict.get('Tarjetas')


        # Imprimir los valores
        print(f"Valor: {valor}")
        print(f"Cláusula: {claúsula}")
        print(f"Puntos: {puntos}")
        print(f"Media: {media}")
        print(f"Partidos: {partidos}")
        print(f"Goles: {goles}")
        print(f"Tarjetas: {tarjetas}")

        try:
            owner_element = driver.find_element(By.XPATH, '//div[@class="box box-owner"]')
            owner_text = owner_element.text
            owner_info = re.search(r'De (.+), fichado el (\d+ \w+ \d{4}) por (.+)', owner_text)
            
            if owner_info:
                owner_name = owner_info.group(1)
                owner_date = owner_info.group(2)
                owner_price = owner_info.group(3)
                print(f"Propietario: {owner_name}, Fecha: {owner_date}, Precio: {owner_price}")
            elif re.search(r'De (.+)', owner_text):
                owner_name = re.search(r'De (.+)', owner_text).group(1)
                print(f"Propietario: {owner_name}")
            else:
                print("Información del propietario no válida")
        
        except NoSuchElementException:
            print("Jugador libre")
        
        try:
            alert_status = driver.find_element(By.XPATH, '//div[@class="box alert-status"]')
            alert_text = alert_status.text
            print(alert_text)
        except NoSuchElementException:
            print("Jugador sin alertas")

        elements = driver.find_elements(By.XPATH, '//div[@class="line btn btn-player-gw"]')

        for element in elements:
            gw = element.find_element(By.XPATH, './/div[@class="gw"]').text
            
            # Buscar elementos con la clase "score"
            scores = element.find_elements(By.XPATH, './/div[contains(@class, "score ")]')
            
            if scores:
                score = scores[0].text
            else:
                # Si no se encuentra ningún elemento con la clase "score", asignar un valor predeterminado
                score = "Sin puntuación"
            
            # Buscar elementos con la clase "bar negative"
            bar_negatives = element.find_elements(By.XPATH, './/div[contains(@class, "bar negative")]')
            if bar_negatives:
                bar_negative_text = bar_negatives[0].text
            else:
                # Si no se encuentra ningún elemento con la clase "bar negative", asignar un valor predeterminado
                bar_negative_text = "Sin texto de sanción o lesión"
            #EVENTOS: TARJETAS GOLES ETC
            # Obtener el elemento <div> con la clase "events"
            #eventos_div = element.find_elements(By.XPATH, './/div[contains(@class, "events")]')
            print("GW:", gw)
            print("Score:", score)
            print("Bar Negative Text:", bar_negative_text)
            datos_jugador = {}

            if "Sancionado" in bar_negative_text:
                datos_jugador['Error'] = 'Sancionado'
                print("Datos jornada", datos_jugador)
                continue  # Salta al siguiente elemento en el bucle for
            elif (score == "Sin puntuación" and not bar_negatives):
                datos_jugador['Error'] = 'No jugó la joranda'
                print("Datos jornada", datos_jugador)
                continue  # Salta al siguiente elemento en el bucle for
            else:
                eventos_div = element.find_elements(By.XPATH, './/div[@class="bar"]')
                # Iterar sobre cada elemento div que contiene elementos SVG
                for evento in eventos_div:
                    #datos_jugador = {}  # Asegúrate de reiniciar los datos del jugador para cada evento
                    intentos = 3  # Número de reintentos
                    while intentos > 0:
                        try:
                            evento.click()
                            wait.until(EC.presence_of_element_located((By.XPATH, './/div[@class="footer"]'))).click()
                            tabla = wait.until(EC.presence_of_element_located((By.XPATH, './/div[contains(@class, "content player-breakdown")]')))
                            filas = tabla.find_elements(By.TAG_NAME, 'tr')
                            for fila in filas:
                                columnas = fila.find_elements(By.TAG_NAME, 'td')
                                if len(columnas) == 2:  # Asegúrate de que hay dos columnas para evitar errores
                                    campo = columnas[0].text
                                    valor = columnas[1].text
                                    datos_jugador[campo] = valor
                            # Una vez que has recogido los datos, intenta cerrar el popup
                            if wait.until(EC.presence_of_element_located((By.ID, 'popup'))):
                                driver.find_element(By.CSS_SELECTOR, '#popup .popup-close').click()
                            break  # Si todo salió bien, rompe el ciclo while
                        except StaleElementReferenceException:
                            intentos -= 1  # Decrementa el contador de intentos y vuelve a intentar
                            if intentos == 0:
                                print("No se pudo recuperar la información del evento después de varios intentos.")
                        except Exception:
                            datos_jugador['Error'] = 'No jugó la joranda'
                            break
                    # Imprimir el número de la jornada, los puntos y los eventos
                    print("Datos jonada", datos_jugador)
                print("---------")
        # Encuentra el contenedor principal
        box_container = driver.find_element(By.CLASS_NAME, 'boxes-2')

        # Encuentra el historial de valores
        historial_valores_container = box_container.find_element(By.XPATH, "//h4[text()='Historial de valores']/parent::div[@class='section-title']/following-sibling::div[@class='box box-records']")
        valores_items = historial_valores_container.find_elements(By.TAG_NAME, 'li')

        # Recorre los elementos del historial de valores y extrae la información
        valores = []
        for item in valores_items:
            top = item.find_element(By.CLASS_NAME, 'top').text
            bottom = item.find_element(By.CLASS_NAME, 'bottom').text
            right = item.find_element(By.CLASS_NAME, 'right').text
            valores.append((top, bottom, right))

        # Encuentra el historial de puntos
        historial_puntos_container = box_container.find_element(By.XPATH, "//h4[text()='Historial de puntos']/parent::div[@class='section-title']/following-sibling::div[@class='box box-records']")
        puntos_items = historial_puntos_container.find_elements(By.TAG_NAME, 'li')

        # Recorre los elementos del historial de puntos y extrae la información
        puntos = []
        for item in puntos_items:
            top = item.find_element(By.CLASS_NAME, 'top').text
            bottom = item.find_element(By.CLASS_NAME, 'bottom').text
            right = item.find_element(By.CLASS_NAME, 'right').text
            puntos.append((top, bottom, right))

        registros_transferencia = []

        box_records_div = driver.find_element(By.XPATH, '//div[@class="box box-records"]')
        lis = box_records_div.find_elements(By.XPATH, './ul/li')

        for li in lis:
            text_elements = li.find_elements(By.XPATH, ".//div[@class='left']//div[@class='top' or @class='bottom']")
            text = [element.text for element in text_elements]
            # Extraer la información con expresiones regulares
            message = '\n'.join(text)
            match = re.search(r'(\d+\s\w+\s\d+)\s·\s(Cláusula|Fichaje)\sDe\s(.+)\sa\s(.+)', message)
            
            if match:
                fecha = match.group(1)
                tipo_operacion = match.group(2)
                usuario_origen = match.group(3)
                usuario_destino = match.group(4)
                precio = li.find_element(By.XPATH, ".//div[@class='right']").text
                
                registro = {
                    "fecha": fecha,
                    "tipo_operacion": tipo_operacion,
                    "usuario_origen": usuario_origen,
                    "usuario_destino": usuario_destino,
                    "precio": precio
                }
                
                registros_transferencia.append(registro)

        for registro in registros_transferencia:
            print("Fecha:", registro["fecha"])
            print("Tipo de operación:", registro["tipo_operacion"])
            print("Usuario origen:", registro["usuario_origen"])
            print("Usuario destino:", registro["usuario_destino"])
            print("Precio:", registro["precio"])
            print()
        # Imprime el historial de valores
        print("Historial de valores:")
        for valor in valores:
            print("Top:", valor[0])
            print("Bottom:", valor[1])
            print("Right:", valor[2])
            print()

        # Imprime el historial de puntos
        print("Historial de puntos:")
        for punto in puntos:
            print("Top:", punto[0])
            print("Bottom:", punto[1])
            print("Right:", punto[2])
            print()

    #print(url_jugadores)
    #time.sleep(500)