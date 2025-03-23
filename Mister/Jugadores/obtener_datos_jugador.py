from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException,StaleElementReferenceException
from selenium.webdriver.common.keys import Keys
import re

def obtener_datos_jugador(driver):
    datos_de_jugadores = []
    evento = []
    wait = WebDriverWait(driver, 2)
    eventos_gol_por_jornada = []
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
        datos_jugador = {}
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

        #print(f"Nombre: {name}, Apellido: {surname}, Posición: {position}")
        datos_jugador['Nombre'] = name
        datos_jugador['Apellido'] = surname
        datos_jugador['Posicion'] = position
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

        datos_jugador['Valor'] = valor
        datos_jugador['Clausula'] = claúsula
        datos_jugador['Puntos'] = puntos
        datos_jugador['Media'] = media
        datos_jugador['Partidos'] = partidos
        datos_jugador['Goles'] = goles
        datos_jugador['Tarjetas'] = tarjetas

        try:
            owner_element = driver.find_element(By.XPATH, '//div[@class="box box-owner"]')
            owner_text = owner_element.text
            owner_info = re.search(r'De (.+), fichado el (\d+ \w+ \d{4}) por (.+)', owner_text)
            
            if owner_info:
                owner_name = owner_info.group(1)
                owner_date = owner_info.group(2)
                owner_price = owner_info.group(3)
                datos_jugador['Propietario'] = owner_name
                datos_jugador['Fecha'] = owner_date
                datos_jugador['Precio'] = owner_price
            elif re.search(r'De (.+)', owner_text):
                owner_name = re.search(r'De (.+)', owner_text).group(1)
                datos_jugador['Propietario'] = owner_name
            else:
                print("Información del propietario no válida")
                datos_jugador['Propietario'] = "Información del propietario no válida"
        
        except NoSuchElementException:
            datos_jugador['Propietario'] = "Jugador libre"
        
        try:
            alert_status = driver.find_element(By.XPATH, '//div[@class="box alert-status"]')
            alert_text = alert_status.text
            datos_jugador['Alerta'] = alert_text
        except NoSuchElementException:
            datos_jugador['Alerta'] = "Jugador sin alertas"
    return datos_jugador

def obtener_datos_jornadas(driver):
    evento = []
    wait = WebDriverWait(driver, 2)
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
        elements = driver.find_elements(By.XPATH, '//div[@class="line btn btn-player-gw"]')
        datos_jornadas = []
        for element in elements:
            datos_jornada = {}
            gw = element.find_element(By.XPATH, './/div[@class="gw"]').text
            
            # Buscar elementos con la clase "score"
            scores = element.find_elements(By.XPATH, './/div[contains(@class, "score ")]')
            
            if scores:
                score = scores[0].text
            else:
                # Si no se encuentra ningún elemento con la clase "score", asignar un valor predeterminado
                score = "Sin puntuación"

            
            eventos_jornada = []
            # Buscar el contenedor de eventos
            eventos_div = element.find_elements(By.XPATH, './/div[contains(@class, "events")]')
            # Iterar sobre cada SVG dentro del contenedor de eventos
            for div in eventos_div:
                eventos_use = div.find_elements(By.XPATH, ".//*[name()='svg' and @class='match-event']")
                # Iterar sobre cada elemento 'use' para extraer el atributo 'href'
                for use in eventos_use:
                    # Asegurarse de que la búsqueda sea relativa al elemento 'use' actual
                    evento_href_element = use.find_element(By.XPATH, ".//*[contains(@href, 'events')]")  # Cambio aquí: './/*' en lugar de '//*'
                    if evento_href_element:
                        # Extraer el tipo de evento basado en el ID del SVG
                        evento_href = evento_href_element.get_attribute('href')
                        # Aquí debes aplicar split en evento_href que es un string, no en evento_href_element que es un WebElement
                        tipo_evento = evento_href.split('#')[1] if '#' in evento_href else evento_href
                        #print(tipo_evento)  # Esto debería mostrar el tipo de evento (por ejemplo, 'events-goal')
                        
                        # Añadir el tipo de evento a la lista de eventos de la jornada
                        eventos_jornada.append(tipo_evento)
            datos_jornada['Nombre'] = name
            datos_jornada['Apellidos'] = surname
            datos_jornada['Jornada'] = gw
            datos_jornada['Puntuacion'] = score
            datos_jornada['Eventos'] = eventos_jornada
            # Buscar elementos con la clase "bar negative"
            bar_negatives = element.find_elements(By.XPATH, './/div[contains(@class, "bar negative")]')
            if bar_negatives:
                bar_negative_text = bar_negatives[0].text
            else:
                # Si no se encuentra ningún elemento con la clase "bar negative", asignar un valor predeterminado
                bar_negative_text = "Sin texto de sanción o lesión"
                datos_jornada['SancionOLesion'] = bar_negative_text

            if "Sancionado" in bar_negative_text:
                datos_jornada['SancionOLesion'] = 'Sancionado'
                #print("Datos jornada", datos_jugador)
                continue  # Salta al siguiente elemento en el bucle for
            elif (score == "Sin puntuación" and not bar_negatives):
                datos_jornada['SancionOLesion'] = 'No jugó la jornada'
                #print("Datos jornada", datos_jugador)
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
                            #wait.until(EC.element_to_be_clickable((By.XPATH, './/div[@class="footer"]'))).click()
                            button_xpath = "//button[contains(text(), 'Ver más estadísticas')]"
                            wait.until(EC.element_to_be_clickable((By.XPATH, button_xpath))).click()
                            tabla = wait.until(EC.presence_of_element_located((By.XPATH, './/div[contains(@class, "content player-breakdown")]')))
                            filas = tabla.find_elements(By.TAG_NAME, 'tr')
                            for fila in filas:
                                columnas = fila.find_elements(By.TAG_NAME, 'td')
                                if len(columnas) == 2:  # Asegúrate de que hay dos columnas para evitar errores
                                    campo = columnas[0].text
                                    valor = columnas[1].text
                                    datos_jornada[campo] = valor
                            # Una vez que has recogido los datos, intenta cerrar el popup
                            if wait.until(EC.presence_of_element_located((By.ID, 'popup'))):
                                driver.find_element(By.CSS_SELECTOR, '#popup .popup-close').click()
                            break  # Si todo salió bien, rompe el ciclo while
                        except StaleElementReferenceException:
                            intentos -= 1  # Decrementa el contador de intentos y vuelve a intentar
                            if intentos == 0:
                                print("No se pudo recuperar la información del evento después de varios intentos.")
                        except Exception:
                            datos_jornada['Error'] = 'No jugó la joranda'
                            break
            datos_jornadas.append(datos_jornada)
    return datos_jornadas

def obtener_registros_transferencia(driver):
    wait = WebDriverWait(driver, 2)
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
                    "Nombre": name,
                    "Apellido": surname,
                    "fecha": fecha,
                    "tipo_operacion": tipo_operacion,
                    "usuario_origen": usuario_origen,
                    "usuario_destino": usuario_destino,
                    "precio": precio
                }
                
                registros_transferencia.append(registro)
    return registros_transferencia

def obtener_puntos(driver):
    wait = WebDriverWait(driver, 2)
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
        # Encuentra el contenedor principal
        box_container = driver.find_element(By.CLASS_NAME, 'boxes-2')
        # Encuentra el historial de puntos
        historial_puntos_container = box_container.find_element(By.XPATH, "//h4[text()='Historial de puntos']/parent::div[@class='section-title']/following-sibling::div[@class='box box-records']")
        puntos_items = historial_puntos_container.find_elements(By.TAG_NAME, 'li')

        # Recorre los elementos del historial de puntos y extrae la información
        puntos = []
        for item in puntos_items:
            punto = {}
            top = item.find_element(By.CLASS_NAME, 'top').text
            bottom = item.find_element(By.CLASS_NAME, 'bottom').text
            right = item.find_element(By.CLASS_NAME, 'right').text
            punto = {
                "Nombre": name,
                "Apellido": surname,
                "top": top,
                "bottom": bottom,
                "right": right
            }
            puntos.append(punto)
    return puntos

def obtener_valores(driver):
    wait = WebDriverWait(driver, 2)
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
        # Encuentra el contenedor principal
        box_container = driver.find_element(By.CLASS_NAME, 'boxes-2')

        # Encuentra el historial de valores
        historial_valores_container = box_container.find_element(By.XPATH, "//h4[text()='Historial de valores']/parent::div[@class='section-title']/following-sibling::div[@class='box box-records']")
        valores_items = historial_valores_container.find_elements(By.TAG_NAME, 'li')

        # Recorre los elementos del historial de valores y extrae la información
        valores = []
        for item in valores_items:
            valor = {}
            top = item.find_element(By.CLASS_NAME, 'top').text
            bottom = item.find_element(By.CLASS_NAME, 'bottom').text
            right = item.find_element(By.CLASS_NAME, 'right').text
            valor = {
                "Nombre": name,
                "Apellido": surname,
                "top": top,
                "bottom": bottom,
                "right": right
            }
            valores.append(valor)
    return valores