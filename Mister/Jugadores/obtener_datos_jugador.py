from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException,StaleElementReferenceException
from selenium.webdriver.common.keys import Keys
import re
from utils import log

def obtener_datos_jugador(driver):
    log("obtener_datos_jugador: Inicio de la función")

    datos_de_jugadores = []
    wait = WebDriverWait(driver, 2)

    try:
        enlace_mas = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Más')]/parent::li/a")
        ))
        enlace_mas.click()
        log("obtener_datos_jugador: Enlace 'Más' clickeado")
    except Exception as e:
        log(f"obtener_datos_jugador: Error al hacer clic en 'Más': {e}")
        return datos_de_jugadores

    driver.implicitly_wait(2)

    try:
        enlace_jugadores = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//button[contains(text(), 'Jugadores')]")
        ))
        enlace_jugadores.click()
        log("obtener_datos_jugador: Enlace 'Jugadores' clickeado")
    except Exception as e:
        log(f"obtener_datos_jugador: Error al hacer clic en 'Jugadores': {e}")
        return datos_de_jugadores

    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)

    while True:
        try:
            button = wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(text(), 'Ver más')]")
            ))
            button.click()
            WebDriverWait(driver, 1).until(EC.invisibility_of_element_located(
                (By.XPATH, '//div[@class="player-list"]')
            ))
            log("obtener_datos_jugador: Botón 'Ver más' clickeado, cargando más jugadores")
        except:
            break

    players = driver.find_elements(By.XPATH, '//ul[@class="player-list search-players-list"]/li')
    url_jugadores = []

    for player in players:
        try:
            player_link = wait.until(EC.element_to_be_clickable(
                (By.CLASS_NAME, 'btn.btn-sw-link.player')
            ))
            player_url = player.find_element(By.TAG_NAME, 'a').get_attribute('href')
            url_jugadores.append(player_url)
        except Exception as e:
            log(f"obtener_datos_jugador: Error obteniendo URL de un jugador: {e}")
            continue

    log(f"obtener_datos_jugador: {len(url_jugadores)} URLs de jugadores recolectadas")

    for player_url in url_jugadores:
        datos_jugador = {}
        driver.get(player_url)
        #log(f"obtener_datos_jugador: Accediendo a perfil {player_url}")
        try:
            name = driver.find_element(By.XPATH, '//div[@class="left"]//div[@class="name"]').text
            surname = driver.find_element(By.CLASS_NAME, 'surname').text.strip()
        except Exception as e:
            log(f"obtener_datos_jugador: Error obteniendo nombre o apellido: {e}")
            continue

        try:
            position_element = driver.find_element(By.XPATH, '//div[@class="team-position"]/i[contains(@class, "pos-")]')
            position_class = position_element.get_attribute('class')
            position_number = re.search(r'pos-(\d+)', position_class).group(1)
            position_mapping = {
                '1': 'PT',
                '2': 'DF',
                '3': 'MC',
                '4': 'DL'
            }
            position = position_mapping.get(position_number, 'Desconocida')
        except Exception as e:
            log(f"obtener_datos_jugador: Error obteniendo posición: {e}")
            position = 'Desconocida'

        datos_jugador['Nombre'] = name
        datos_jugador['Apellido'] = surname
        datos_jugador['Posicion'] = position

        try:
            stats_wrapper = driver.find_element(By.CLASS_NAME, 'player-stats-wrapper')
            stats_items = stats_wrapper.find_elements(By.CLASS_NAME, 'item')
            stats_dict = {}
            for item in stats_items:
                label = item.find_element(By.CLASS_NAME, 'label').text
                value = item.find_element(By.CLASS_NAME, 'value').text
                stats_dict[label] = value

            datos_jugador['Valor'] = stats_dict.get('Valor')
            datos_jugador['Clausula'] = stats_dict.get('Cláusula', 'Sin cláusula')
            datos_jugador['Puntos'] = stats_dict.get('Puntos')
            datos_jugador['Media'] = stats_dict.get('Media')
            datos_jugador['Partidos'] = stats_dict.get('Partidos')
            datos_jugador['Goles'] = stats_dict.get('Goles')
            datos_jugador['Tarjetas'] = stats_dict.get('Tarjetas')
        except Exception as e:
            log(f"obtener_datos_jugador: Error obteniendo estadísticas de {name}: {e}")

        try:
            owner_element = driver.find_element(By.XPATH, '//div[@class="box box-owner"]')
            owner_text = owner_element.text
            owner_info = re.search(r'De (.+), fichado el (\d+ \w+ \d{4}) por (.+)', owner_text)
            if owner_info:
                datos_jugador['Propietario'] = owner_info.group(1)
                datos_jugador['Fecha'] = owner_info.group(2)
                datos_jugador['Precio'] = owner_info.group(3)
            elif re.search(r'De (.+)', owner_text):
                datos_jugador['Propietario'] = re.search(r'De (.+)', owner_text).group(1)
            else:
                datos_jugador['Propietario'] = "Información del propietario no válida"
        except NoSuchElementException:
            datos_jugador['Propietario'] = "Jugador libre"

        try:
            alert_status = driver.find_element(By.XPATH, '//div[@class="box alert-status"]')
            raw_alert_text = alert_status.text
            alert_text = " ".join(raw_alert_text.split())
            datos_jugador['Alerta'] = alert_text
        except NoSuchElementException:
            datos_jugador['Alerta'] = "Jugador sin alertas"

        datos_de_jugadores.append(datos_jugador)
        log(f"obtener_datos_jugador: Datos añadidos para {name} {surname}")

    log(f"obtener_datos_jugador: Finalización de la función con {len(datos_de_jugadores)} jugadores procesados")
    return datos_de_jugadores


def obtener_datos_jornadas(driver):
    log("obtener_datos_jornadas: Inicio de la función")

    wait = WebDriverWait(driver, 2)

    try:
        enlace_mas = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Más')]/parent::li/a")
        ))
        enlace_mas.click()
        log("obtener_datos_jornadas: Enlace 'Más' clickeado")
    except Exception as e:
        log(f"obtener_datos_jornadas: Error al hacer clic en 'Más': {e}")
        return []

    driver.implicitly_wait(2)

    try:
        enlace_jugadores = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//button[contains(text(), 'Jugadores')]")
        ))
        enlace_jugadores.click()
        log("obtener_datos_jornadas: Enlace 'Jugadores' clickeado")
    except Exception as e:
        log(f"obtener_datos_jornadas: Error al hacer clic en 'Jugadores': {e}")
        return []

    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)

    while True:
        try:
            button = wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(text(), 'Ver más')]")
            ))
            button.click()
            WebDriverWait(driver, 1).until(EC.invisibility_of_element_located(
                (By.XPATH, '//div[@class="player-list"]')
            ))
            log("obtener_datos_jornadas: Botón 'Ver más' clickeado")
        except:
            break

    players = driver.find_elements(By.XPATH, '//ul[@class="player-list search-players-list"]/li')
    url_jugadores = []

    for player in players:
        try:
            player_link = wait.until(EC.element_to_be_clickable(
                (By.CLASS_NAME, 'btn.btn-sw-link.player')
            ))
            player_url = player.find_element(By.TAG_NAME, 'a').get_attribute('href')
            url_jugadores.append(player_url)
        except Exception as e:
            log(f"obtener_datos_jornadas: Error obteniendo URL de jugador: {e}")
            continue

    log(f"obtener_datos_jornadas: {len(url_jugadores)} URLs de jugadores recopiladas")

    datos_jornadas = []

    for player_url in url_jugadores:
        driver.get(player_url)
        try:
            name = driver.find_element(By.XPATH, '//div[@class="left"]//div[@class="name"]').text
            surname = driver.find_element(By.CLASS_NAME, 'surname').text.strip()
        except Exception as e:
            log(f"obtener_datos_jornadas: Error obteniendo nombre/apellido en {player_url}: {e}")
            continue

        elements = driver.find_elements(By.XPATH, '//div[@class="line btn btn-player-gw"]')

        for element in elements:
            datos_jornada = {}
            try:
                gw = element.find_element(By.XPATH, './/div[@class="gw"]').text
                scores = element.find_elements(By.XPATH, './/div[contains(@class, "score ")]')
                score = scores[0].text if scores else "Sin puntuación"

                eventos_jornada = []
                eventos_div = element.find_elements(By.XPATH, './/div[contains(@class, "events")]')
                for div in eventos_div:
                    eventos_use = div.find_elements(By.XPATH, ".//*[name()='svg' and @class='match-event']")
                    for use in eventos_use:
                        evento_href_element = use.find_element(By.XPATH, ".//*[contains(@href, 'events')]")
                        if evento_href_element:
                            evento_href = evento_href_element.get_attribute('href')
                            tipo_evento = evento_href.split('#')[1] if '#' in evento_href else evento_href
                            eventos_jornada.append(tipo_evento)

                datos_jornada.update({
                    'Nombre': name,
                    'Apellidos': surname,
                    'Jornada': gw,
                    'Puntuacion': score,
                    'Eventos': eventos_jornada
                })

                bar_negatives = element.find_elements(By.XPATH, './/div[contains(@class, "bar negative")]')
                bar_negative_text = bar_negatives[0].text if bar_negatives else "Sin texto de sanción o lesión"
                datos_jornada['SancionOLesion'] = bar_negative_text

                if "Sancionado" in bar_negative_text:
                    datos_jornada['SancionOLesion'] = 'Sancionado'
                    log(f"obtener_datos_jornadas: {name} sancionado en jornada {gw}")
                    continue
                elif score == "Sin puntuación" and not bar_negatives:
                    datos_jornada['SancionOLesion'] = 'No jugó la jornada'
                    continue
                else:
                    eventos_div = element.find_elements(By.XPATH, './/div[@class="bar"]')
                    for evento in eventos_div:
                        intentos = 3
                        while intentos > 0:
                            try:
                                evento.click()
                                button_xpath = "//button[contains(text(), 'Ver más estadísticas')]"
                                wait.until(EC.element_to_be_clickable((By.XPATH, button_xpath))).click()
                                tabla = wait.until(EC.presence_of_element_located(
                                    (By.XPATH, './/div[contains(@class, "content player-breakdown")]')
                                ))
                                filas = tabla.find_elements(By.TAG_NAME, 'tr')
                                for fila in filas:
                                    columnas = fila.find_elements(By.TAG_NAME, 'td')
                                    if len(columnas) == 2:
                                        campo = columnas[0].text
                                        valor = columnas[1].text
                                        datos_jornada[campo] = valor

                                # Cerrar popup
                                if wait.until(EC.presence_of_element_located((By.ID, 'popup'))):
                                    driver.find_element(By.CSS_SELECTOR, '#popup .popup-close').click()
                                break
                            except StaleElementReferenceException:
                                intentos -= 1
                                if intentos == 0:
                                    log(f"obtener_datos_jornadas: No se pudo recuperar stats de {name} en jornada {gw}")
                            except Exception:
                                datos_jornada['Error'] = 'No jugó la jornada'
                                break

                datos_jornadas.append(datos_jornada)
                log(f"obtener_datos_jornadas: Datos añadidos para {name} en jornada {gw}")
            except Exception as e:
                log(f"obtener_datos_jornadas: Error procesando jornada de {name}: {e}")

    log(f"obtener_datos_jornadas: Finalización con {len(datos_jornadas)} registros procesados")
    return datos_jornadas


def obtener_registros_transferencia(driver):
    log("obtener_registros_transferencia: Inicio de la función")

    wait = WebDriverWait(driver, 2)

    try:
        enlace_mas = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Más')]/parent::li/a")
        ))
        enlace_mas.click()
        log("obtener_registros_transferencia: Enlace 'Más' clickeado")
    except Exception as e:
        log(f"obtener_registros_transferencia: Error al hacer clic en 'Más': {e}")
        return []

    driver.implicitly_wait(2)

    try:
        enlace_jugadores = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//button[contains(text(), 'Jugadores')]")
        ))
        enlace_jugadores.click()
        log("obtener_registros_transferencia: Enlace 'Jugadores' clickeado")
    except Exception as e:
        log(f"obtener_registros_transferencia: Error al hacer clic en 'Jugadores': {e}")
        return []

    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)

    while True:
        try:
            button = wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(text(), 'Ver más')]")
            ))
            button.click()
            WebDriverWait(driver, 1).until(EC.invisibility_of_element_located(
                (By.XPATH, '//div[@class="player-list"]')
            ))
            log("obtener_registros_transferencia: Botón 'Ver más' clickeado")
        except:
            break

    players = driver.find_elements(By.XPATH, '//ul[@class="player-list search-players-list"]/li')
    url_jugadores = []

    for player in players:
        try:
            player_link = wait.until(EC.element_to_be_clickable(
                (By.CLASS_NAME, 'btn.btn-sw-link.player')
            ))
            player_url = player.find_element(By.TAG_NAME, 'a').get_attribute('href')
            url_jugadores.append(player_url)
        except Exception as e:
            log(f"obtener_registros_transferencia: Error obteniendo URL de jugador: {e}")
            continue

    log(f"obtener_registros_transferencia: {len(url_jugadores)} URLs de jugadores recolectadas")

    todos_registros = []

    for player_url in url_jugadores:
        driver.get(player_url)
        try:
            name = driver.find_element(By.XPATH, '//div[@class="left"]//div[@class="name"]').text
            surname = driver.find_element(By.CLASS_NAME, 'surname').text.strip()
        except Exception as e:
            log(f"obtener_registros_transferencia: Error obteniendo nombre o apellido en {player_url}: {e}")
            continue

        registros_transferencia = []
        try:
            box_records_div = driver.find_element(By.XPATH, '//div[@class="box box-records"]')
            lis = box_records_div.find_elements(By.XPATH, './ul/li')
        except Exception as e:
            log(f"obtener_registros_transferencia: No se encontraron registros en {player_url}: {e}")
            continue

        for li in lis:
            try:
                text_elements = li.find_elements(By.XPATH, ".//div[@class='left']//div[@class='top' or @class='bottom']")
                text = [element.text for element in text_elements]
                message = '\n'.join(text)

                match = re.search(r'(\d+\s\w+\s\d+)\s·\s(Cláusula|Fichaje)\sDe\s(.+)\sa\s(.+)', message)
                if match:
                    fecha = match.group(1)
                    tipo_operacion = match.group(2)
                    usuario_origen = match.group(3)
                    usuario_destino = match.group(4)
                    try:
                        precio = li.find_element(By.XPATH, ".//div[@class='right']").text
                    except Exception:
                        precio = ""

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
                    log(f"obtener_registros_transferencia: Registro añadido - {name} {surname} | {tipo_operacion} de {usuario_origen} a {usuario_destino}")
            except Exception as e:
                log(f"obtener_registros_transferencia: Error procesando registro para {name}: {e}")
                continue

        todos_registros.extend(registros_transferencia)

    log(f"obtener_registros_transferencia: Finalización con {len(todos_registros)} registros totales")
    return todos_registros


def obtener_puntos(driver):
    log("obtener_puntos: Inicio de la función")

    wait = WebDriverWait(driver, 2)

    try:
        enlace_mas = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Más')]/parent::li/a")
        ))
        enlace_mas.click()
        log("obtener_puntos: Enlace 'Más' clickeado")
    except Exception as e:
        log(f"obtener_puntos: Error al hacer clic en 'Más': {e}")
        return []

    driver.implicitly_wait(2)

    try:
        enlace_jugadores = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//button[contains(text(), 'Jugadores')]")
        ))
        enlace_jugadores.click()
        log("obtener_puntos: Enlace 'Jugadores' clickeado")
    except Exception as e:
        log(f"obtener_puntos: Error al hacer clic en 'Jugadores': {e}")
        return []

    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)

    while True:
        try:
            button = wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(text(), 'Ver más')]")
            ))
            button.click()
            WebDriverWait(driver, 1).until(EC.invisibility_of_element_located(
                (By.XPATH, '//div[@class="player-list"]')
            ))
            log("obtener_puntos: Botón 'Ver más' clickeado")
        except:
            break

    players = driver.find_elements(By.XPATH, '//ul[@class="player-list search-players-list"]/li')
    url_jugadores = []

    for player in players:
        try:
            player_link = wait.until(EC.element_to_be_clickable(
                (By.CLASS_NAME, 'btn.btn-sw-link.player')
            ))
            player_url = player.find_element(By.TAG_NAME, 'a').get_attribute('href')
            url_jugadores.append(player_url)
        except Exception as e:
            log(f"obtener_puntos: Error obteniendo URL de jugador: {e}")
            continue

    log(f"obtener_puntos: {len(url_jugadores)} URLs de jugadores recolectadas")

    todos_puntos = []

    for player_url in url_jugadores:
        driver.get(player_url)
        try:
            name = driver.find_element(By.XPATH, '//div[@class="left"]//div[@class="name"]').text
            surname = driver.find_element(By.CLASS_NAME, 'surname').text.strip()
        except Exception as e:
            log(f"obtener_puntos: Error obteniendo nombre/apellido en {player_url}: {e}")
            continue

        try:
            box_container = driver.find_element(By.CLASS_NAME, 'boxes-2')
            historial_puntos_container = box_container.find_element(
                By.XPATH, "//h4[text()='Historial de puntos']/parent::div[@class='section-title']/following-sibling::div[@class='box box-records']"
            )
            puntos_items = historial_puntos_container.find_elements(By.TAG_NAME, 'li')
        except Exception as e:
            log(f"obtener_puntos: Error obteniendo historial de puntos para {name} {surname}: {e}")
            continue

        if len(puntos_items) == 1:
            try:
                class_attr = puntos_items[0].get_attribute("class")
                if "alert-no-info" in class_attr:
                    registro = {
                        "Nombre": name,
                        "Apellido": surname,
                        "top": None,
                        "bottom": None,
                        "right": None
                    }
                    todos_puntos.append(registro)
                    log(f"obtener_puntos: {name} {surname} no tiene historial de puntos")
                    continue
            except Exception:
                pass

        for item in puntos_items:
            try:
                try:
                    top = item.find_element(By.CLASS_NAME, 'top').text
                except NoSuchElementException:
                    top = None
                try:
                    bottom = item.find_element(By.CLASS_NAME, 'bottom').text
                except NoSuchElementException:
                    bottom = None
                try:
                    right = item.find_element(By.CLASS_NAME, 'right').text
                except NoSuchElementException:
                    right = None

                registro = {
                    "Nombre": name,
                    "Apellido": surname,
                    "top": top,
                    "bottom": bottom,
                    "right": right
                }
                todos_puntos.append(registro)
                log(f"obtener_puntos: Puntos añadidos para {name} {surname} - top: {top}, bottom: {bottom}, right: {right}")
            except Exception as e:
                log(f"obtener_puntos: Error extrayendo punto de {name} {surname}: {e}")
                continue

    log(f"obtener_puntos: Finalización con {len(todos_puntos)} registros de puntos")
    return todos_puntos


def obtener_valores(driver):
    log("obtener_valores: Inicio de la función")

    wait = WebDriverWait(driver, 2)

    try:
        enlace_mas = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Más')]/parent::li/a")
        ))
        enlace_mas.click()
        log("obtener_valores: Enlace 'Más' clickeado")
    except Exception as e:
        log(f"obtener_valores: Error al hacer clic en 'Más': {e}")
        return []

    driver.implicitly_wait(2)

    try:
        enlace_jugadores = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//button[contains(text(), 'Jugadores')]")
        ))
        enlace_jugadores.click()
        log("obtener_valores: Enlace 'Jugadores' clickeado")
    except Exception as e:
        log(f"obtener_valores: Error al hacer clic en 'Jugadores': {e}")
        return []

    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)

    while True:
        try:
            button = wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(text(), 'Ver más')]")
            ))
            button.click()
            WebDriverWait(driver, 1).until(EC.invisibility_of_element_located(
                (By.XPATH, '//div[@class="player-list"]')
            ))
            log("obtener_valores: Botón 'Ver más' clickeado")
        except:
            break

    players = driver.find_elements(By.XPATH, '//ul[@class="player-list search-players-list"]/li')
    url_jugadores = []

    for player in players:
        try:
            player_link = wait.until(EC.element_to_be_clickable(
                (By.CLASS_NAME, 'btn.btn-sw-link.player')
            ))
            player_url = player.find_element(By.TAG_NAME, 'a').get_attribute('href')
            url_jugadores.append(player_url)
        except Exception as e:
            log(f"obtener_valores: Error obteniendo URL de jugador: {e}")
            continue

    log(f"obtener_valores: {len(url_jugadores)} URLs de jugadores recolectadas")

    valores = []

    for player_url in url_jugadores:
        driver.get(player_url)
        try:
            name = driver.find_element(By.XPATH, '//div[@class="left"]//div[@class="name"]').text
        except Exception:
            name = None
        try:
            surname = driver.find_element(By.CLASS_NAME, 'surname').text.strip()
        except Exception:
            surname = None

        try:
            box_container = driver.find_element(By.CLASS_NAME, 'boxes-2')
            historial_valores_container = box_container.find_element(
                By.XPATH, "//h4[text()='Historial de valores']/parent::div[@class='section-title']/following-sibling::div[@class='box box-records']"
            )
            valores_items = historial_valores_container.find_elements(By.TAG_NAME, 'li')
        except Exception as e:
            log(f"obtener_valores: Error obteniendo historial de valores para {name} {surname}: {e}")
            continue

        if len(valores_items) == 1:
            try:
                class_attr = valores_items[0].get_attribute("class")
                if "alert-no-info" in class_attr:
                    registro = {
                        "Nombre": name,
                        "Apellido": surname,
                        "top": None,
                        "bottom": None,
                        "right": None
                    }
                    valores.append(registro)
                    log(f"obtener_valores: {name} {surname} sin historial de valores")
                    continue
            except Exception:
                pass

        for item in valores_items:
            try:
                try:
                    top = item.find_element(By.CLASS_NAME, 'top').text
                except NoSuchElementException:
                    top = None
                try:
                    bottom = item.find_element(By.CLASS_NAME, 'bottom').text
                except NoSuchElementException:
                    bottom = None
                try:
                    right = item.find_element(By.CLASS_NAME, 'right').text
                except NoSuchElementException:
                    right = None

                registro = {
                    "Nombre": name,
                    "Apellido": surname,
                    "top": top,
                    "bottom": bottom,
                    "right": right
                }
                valores.append(registro)
                log(f"obtener_valores: Valor añadido para {name} {surname} - top: {top}, bottom: {bottom}, right: {right}")
            except Exception as e:
                log(f"obtener_valores: Error extrayendo valor de {name} {surname}: {e}")
                continue

    log(f"obtener_valores: Finalización con {len(valores)} registros de valores")
    return valores
