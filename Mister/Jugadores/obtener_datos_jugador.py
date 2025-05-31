from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException,StaleElementReferenceException,TimeoutException
from selenium.webdriver.common.keys import Keys
import re
from utils import log,obtener_urls_desde_db
import unicodedata
from utils import conexion_db
import time

def normalizar_label(texto):
    return unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8').lower().strip()

def sesion_activa(driver):
    try:
        driver.find_element(By.XPATH, '//a[contains(@class, "btn-play") and contains(text(), "Jugar")]')
        return False  # botón "Jugar" = no logueado
    except NoSuchElementException:
        return True

def obtener_urls_jugadores(driver):
    log("obtener_urls_jugadores: Inicio")

    wait = WebDriverWait(driver, 2)
    datos_urls = []

    try:
        enlace_mas = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Más')]/parent::li/a")
        ))
        enlace_mas.click()
        log("obtener_urls_jugadores: Enlace 'Más' clickeado")
    except Exception as e:
        log(f"obtener_urls_jugadores: Error al hacer clic en 'Más': {e}")
        return []

    driver.implicitly_wait(2)

    try:
        enlace_jugadores = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//button[contains(text(), 'Jugadores')]")
        ))
        enlace_jugadores.click()
        log("obtener_urls_jugadores: Enlace 'Jugadores' clickeado")
    except Exception as e:
        log(f"obtener_urls_jugadores: Error al hacer clic en 'Jugadores': {e}")
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
            log("obtener_urls_jugadores: Botón 'Ver más' clickeado")
        except:
            break

    players = driver.find_elements(By.XPATH, '//ul[@class="player-list search-players-list"]/li')

    for player in players:
        try:
            url = player.find_element(By.TAG_NAME, 'a').get_attribute('href')
            match = re.search(r'/players/(\d+)/([\w\-\d]+)', url)
            if match:
                id_jugador = match.group(1)
                nombre_apellido = match.group(2).replace("-", " ").title()
                datos_urls.append({
                    "id_jugador": id_jugador,
                    "nombre_apellido": nombre_apellido,
                    "url": url
                })
        except Exception as e:
            log(f"obtener_urls_jugadores: Error procesando jugador: {e}")
            continue

    log(f"obtener_urls_jugadores: Total URLs formateadas: {len(datos_urls)}")
    return datos_urls

def obtener_datos_jugador(driver):
    log("obtener_datos_jugador: Inicio de la función")
    datos_de_jugadores = []
    urls_jugadores = obtener_urls_desde_db()

    for player_url in urls_jugadores:
        datos_jugador = {}
        driver.get(player_url)
        log(f"Accediendo a perfil: {player_url}")

        try:
            name = driver.find_element(By.XPATH, '//div[@class="left"]//div[@class="name"]').text
            surname = driver.find_element(By.CLASS_NAME, 'surname').text.strip()

            # ✅ Extraer nombre del equipo
            team_element = driver.find_element(By.XPATH, '//div[@class="team-position"]/a')
            equipo = team_element.get_attribute("data-title").strip()
        except Exception as e:
            log(f"Error obteniendo nombre, apellido o equipo: {e}")
            continue

        try:
            position_element = driver.find_element(By.XPATH, '//div[@class="team-position"]/i[contains(@class, "pos-")]')
            position_class = position_element.get_attribute('class')
            position_number = re.search(r'pos-(\d+)', position_class).group(1)
            position_mapping = {'1': 'PT', '2': 'DF', '3': 'MC', '4': 'DL'}
            position = position_mapping.get(position_number, 'Desconocida')
        except Exception as e:
            log(f"Error obteniendo posición: {e}")
            position = 'Desconocida'

        datos_jugador['Nombre'] = name
        datos_jugador['Apellido'] = surname
        datos_jugador['Equipo'] = equipo
        datos_jugador['Posicion'] = position
        player_id = player_url.split("/players/")[1].split("/")[0]
        datos_jugador['id_jugador'] = player_id

        try:
            stats_wrapper = driver.find_element(By.CLASS_NAME, 'player-stats-wrapper')
            stats_items = stats_wrapper.find_elements(By.CLASS_NAME, 'item')
            stats_dict = {}
            for item in stats_items:
                label = item.find_element(By.CLASS_NAME, 'label').text
                value = item.find_element(By.CLASS_NAME, 'value').text
                stats_dict[label] = value

            datos_jugador['Valor'] = stats_dict.get('Valor')
            datos_jugador['Clausula'] = stats_dict.get('Cláusula', '')
            datos_jugador['Puntos'] = stats_dict.get('Puntos')
            datos_jugador['Media'] = stats_dict.get('Media')
            datos_jugador['Partidos'] = stats_dict.get('Partidos')
            datos_jugador['Goles'] = stats_dict.get('Goles')
            datos_jugador['Tarjetas'] = stats_dict.get('Tarjetas')
        except Exception as e:
            log(f"Error obteniendo estadísticas de {name}: {e}")

        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "box-records"))
            )
            historial_boxes = driver.find_elements(By.CLASS_NAME, "box-records")
            movimientos = []
            for box in historial_boxes:
                lis = box.find_elements(By.TAG_NAME, "li")
                for li in lis:
                    texto = li.text.strip()
                    if "· Fichaje" in texto and "De" in texto and "a" in texto:
                        movimientos.append(li)

            if movimientos:
                ultimo_mov = movimientos[0]
                left = ultimo_mov.find_element(By.CLASS_NAME, "left")
                right = ultimo_mov.find_element(By.CLASS_NAME, "right")
                texto_bottom = left.find_element(By.CLASS_NAME, "bottom").text
                precio = right.text.strip()
                fecha = left.find_element(By.CLASS_NAME, "top").text.strip()

                match = re.search(r'De\s+(.*?)\s+a\s+(.*)', texto_bottom)
                if match:
                    datos_jugador['Propietario'] = match.group(2).strip()
                else:
                    datos_jugador['Propietario'] = "Jugador libre"

                datos_jugador['Fecha'] = fecha
                datos_jugador['Precio'] = precio
            else:
                datos_jugador['Propietario'] = "Jugador libre"
                datos_jugador['Fecha'] = ""
                datos_jugador['Precio'] = ""
        except Exception as e:
            log(f"Error extrayendo propietario: {e}")
            datos_jugador['Propietario'] = "Jugador libre"
            datos_jugador['Fecha'] = ""
            datos_jugador['Precio'] = ""

        try:
            alert_boxes = driver.find_elements(By.XPATH, '//div[@class="box alert-status"]')
            if not alert_boxes:
                datos_jugador['Alerta'] = "Jugador sin alertas"
            else:
                alertas = []
                for alert in alert_boxes:
                    texto = " ".join(alert.text.strip().split())
                    alertas.append(texto)
                datos_jugador['Alerta'] = " | ".join(alertas)
        except Exception as e:
            log(f"Error extrayendo alertas: {e}")
            datos_jugador['Alerta'] = "Jugador sin alertas"

        datos_de_jugadores.append(datos_jugador)
        log(f"Jugador procesado: {datos_jugador}")

    log(f"obtener_datos_jugador: Finalización con {len(datos_de_jugadores)} jugadores procesados")
    return datos_de_jugadores



def obtener_datos_jornadas(driver):
    log("obtener_datos_jornadas_inicial: Inicio de la función")

    wait = WebDriverWait(driver, 2)
    datos_jornadas = []
    urls_jugadores = obtener_urls_desde_db()

    with conexion_db() as conn:
        with conn.cursor() as cur:

            for player_url in urls_jugadores:
                driver.get(player_url)
                try:
                    name = driver.find_element(By.XPATH, '//div[@class="left"]//div[@class="name"]').text
                    surname = driver.find_element(By.CLASS_NAME, 'surname').text.strip()
                except Exception:
                    continue

                player_id = player_url.split("/players/")[1].split("/")[0]

                try:
                    cur.execute("SELECT ultima_jornada FROM dbo.progreso_jornadas WHERE id_jugador = %s", (player_id,))
                    resultado = cur.fetchone()
                    ultima_jornada = resultado[0] if resultado else 0
                except Exception:
                    ultima_jornada = 0

                try:
                    elements = driver.find_elements(By.XPATH, '//div[@class="line btn btn-player-gw"]')
                except Exception:
                    continue
                
                jornadas_encontradas = set()
                jornadas_insertadas = []

                for element in elements:
                    try:
                        gw_text = element.find_element(By.XPATH, './/div[@class="gw"]').text
                        numero_jornada = int(re.search(r'\d+', gw_text).group())
                    except Exception:
                        continue

                    jornadas_encontradas.add(numero_jornada)

                    if numero_jornada <= ultima_jornada:
                        continue

                    datos_jornada = {
                        "id_jugador": player_id,
                        "Nombre": name,
                        "Apellido": surname,
                        "Jornada": gw_text
                    }

                    try:
                        scores = element.find_elements(By.XPATH, './/div[contains(@class, "score ")]')
                        score = scores[0].text if scores else "Sin puntuación"

                        eventos_jornada = []
                        eventos_div = element.find_elements(By.XPATH, './/div[contains(@class, "events")]')
                        for div in eventos_div:
                            eventos_use = div.find_elements(By.XPATH, ".//*[name()='svg' and @class='match-event']")
                            for use in eventos_use:
                                try:
                                    evento_href_element = use.find_element(By.XPATH, ".//*[contains(@href, 'events')]")
                                    if evento_href_element:
                                        evento_href = evento_href_element.get_attribute('href')
                                        tipo_evento = evento_href.split('#')[1] if '#' in evento_href else evento_href
                                        eventos_jornada.append(tipo_evento)
                                except:
                                    continue

                        datos_jornada.update({
                            'Puntuacion': score,
                            'Eventos': " | ".join(eventos_jornada) if eventos_jornada else ""
                        })

                        bar_negatives = element.find_elements(By.XPATH, './/div[contains(@class, "bar negative")]')
                        bar_negative_text = bar_negatives[0].text if bar_negatives else "Sin texto de sanción o lesión"
                        datos_jornada['SancionOLesion'] = bar_negative_text

                        if "Sancionado" in bar_negative_text:
                            datos_jornada['SancionOLesion'] = 'Sancionado'
                            datos_jornadas.append(datos_jornada)
                            log(f"Datos extraídos para {name} {surname} en jornada {gw_text} (Sancionado)")
                            log(f"→ Datos: {datos_jornada}")
                            jornadas_insertadas.append(numero_jornada)
                            continue

                        elif score == "Sin puntuación" and not bar_negatives:
                            datos_jornada['SancionOLesion'] = 'No jugó la jornada'
                            datos_jornadas.append(datos_jornada)
                            log(f"Datos extraídos para {name} {surname} en jornada {gw_text} (No jugó)")
                            log(f"→ Datos: {datos_jornada}")
                            jornadas_insertadas.append(numero_jornada)
                            continue

                        # Extraer estadísticas si hay popup
                        stats_extraidos = False
                        eventos_div = element.find_elements(By.XPATH, './/div[@class="bar"]')

                        for evento in eventos_div:
                            try:
                                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", evento)
                                time.sleep(0.3)
                                driver.execute_script("arguments[0].click();", evento)

                                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "popup")))

                                try:
                                    toast = WebDriverWait(driver, 1).until(
                                        EC.presence_of_element_located((By.ID, "toast"))
                                    )
                                    if "no ha puntuado" in toast.text.lower():
                                        datos_jornada['SancionOLesion'] = 'No jugó la jornada'
                                        stats_extraidos = True
                                        break
                                except:
                                    pass

                                try:
                                    WebDriverWait(driver, 10).until(
                                        EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Ver más estadísticas')]"))
                                    )
                                    boton_stats = driver.find_element(By.XPATH, "//button[contains(text(), 'Ver más estadísticas')]")
                                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", boton_stats)
                                    time.sleep(0.5)
                                    driver.execute_script("arguments[0].click();", boton_stats)
                                except Exception as e:
                                    log(f"No se pudo hacer clic en 'Ver más estadísticas': {e}")
                                    try:
                                        boton_stats_fresco = driver.find_element(By.XPATH, "//button[contains(text(), 'Ver más estadísticas')]")
                                        data_stats = boton_stats_fresco.get_attribute("data-stats")
                                        if data_stats:
                                            import json
                                            stats_dict = json.loads(data_stats.replace('&quot;', '"'))
                                            for k, v in stats_dict.items():
                                                datos_jornada[k] = str(v)
                                            stats_extraidos = True
                                            break
                                    except Exception as e2:
                                        log(f"No se pudieron leer los data-stats directamente tras refrescar: {e2}")
                                    continue

                                tabla = WebDriverWait(driver, 10).until(
                                    EC.presence_of_element_located((By.XPATH, '//div[contains(@class, "content player-breakdown")]'))
                                )
                                filas = tabla.find_elements(By.TAG_NAME, 'tr')
                                for fila in filas:
                                    columnas = fila.find_elements(By.TAG_NAME, 'td')
                                    if len(columnas) == 2:
                                        campo = columnas[0].text.strip()
                                        valor = columnas[1].text.strip()
                                        datos_jornada[campo] = valor

                                try:
                                    popup_close = driver.find_element(By.CSS_SELECTOR, '#popup .popup-close')
                                    driver.execute_script("arguments[0].click();", popup_close)
                                except:
                                    pass

                                stats_extraidos = True
                                break

                            except Exception as e:
                                log(f"Error al intentar extraer estadísticas del popup: {e}")
                                continue

                        if not stats_extraidos and datos_jornada['SancionOLesion'] == "Sin texto de sanción o lesión":
                            datos_jornada['SancionOLesion'] = 'No se pudieron extraer estadísticas'

                        datos_jornadas.append(datos_jornada)
                        log(f"Datos extraídos para {name} {surname} en jornada {gw_text}")
                        log(f"→ Datos: {datos_jornada}")
                        jornadas_insertadas.append(numero_jornada)

                    except Exception:
                        continue

                if jornadas_insertadas:
                    nueva_jornada = max(jornadas_insertadas)
                    cur.execute("""
                        INSERT INTO dbo.progreso_jornadas(id_jugador, ultima_jornada)
                        VALUES (%s, %s)
                        ON CONFLICT (id_jugador)
                        DO UPDATE SET ultima_jornada = EXCLUDED.ultima_jornada
                    """, (player_id, nueva_jornada))

                    jornadas_teoricas = set(range(ultima_jornada + 1, max(jornadas_encontradas) + 1))
                    jornadas_faltantes = jornadas_teoricas - set(jornadas_insertadas)

                    for j_faltante in jornadas_faltantes:
                        cur.execute("""
                            INSERT INTO dbo.progreso_jornadas_pendientes(id_jugador, jornada)
                            VALUES (%s, %s)
                            ON CONFLICT DO NOTHING
                        """, (player_id, j_faltante))

                time.sleep(0.3)

        conn.commit()

    log(f"obtener_datos_jornadas_inicial: Finalización con {len(datos_jornadas)} registros procesados")
    return datos_jornadas


def obtener_datos_jornadas_inicial(driver, max_jornada=34, salto=2):
    log("obtener_datos_jornadas_inicial: Inicio de la función")

    wait = WebDriverWait(driver, 2)
    datos_jornadas = []
    urls_jugadores = obtener_urls_desde_db()

    with conexion_db() as conn:
        with conn.cursor() as cur:

            for player_url in urls_jugadores:
                driver.get(player_url)
                try:
                    name = driver.find_element(By.XPATH, '//div[@class="left"]//div[@class="name"]').text
                    surname = driver.find_element(By.CLASS_NAME, 'surname').text.strip()
                except Exception:
                    continue

                player_id = player_url.split("/players/")[1].split("/")[0]

                try:
                    cur.execute("SELECT ultima_jornada FROM dbo.progreso_jornadas WHERE id_jugador = %s", (player_id,))
                    resultado = cur.fetchone()
                    ultima_jornada = resultado[0] if resultado else 0
                except Exception:
                    ultima_jornada = 0

                if ultima_jornada >= max_jornada:
                    continue

                try:
                    elements = driver.find_elements(By.XPATH, '//div[@class="line btn btn-player-gw"]')
                except Exception:
                    continue

                jornadas_insertadas = []

                for element in elements:
                    try:
                        gw_text = element.find_element(By.XPATH, './/div[@class="gw"]').text
                        numero_jornada = int(re.search(r'\d+', gw_text).group())
                    except Exception:
                        continue

                    if not (ultima_jornada < numero_jornada <= ultima_jornada + salto):
                        continue

                    datos_jornada = {
                        "id_jugador": player_id,
                        "Nombre": name,
                        "Apellido": surname,
                        "Jornada": gw_text
                    }

                    try:
                        scores = element.find_elements(By.XPATH, './/div[contains(@class, "score ")]')
                        score = scores[0].text if scores else "Sin puntuación"

                        eventos_jornada = []
                        eventos_div = element.find_elements(By.XPATH, './/div[contains(@class, "events")]')
                        for div in eventos_div:
                            eventos_use = div.find_elements(By.XPATH, ".//*[name()='svg' and @class='match-event']")
                            for use in eventos_use:
                                try:
                                    evento_href_element = use.find_element(By.XPATH, ".//*[contains(@href, 'events')]")
                                    if evento_href_element:
                                        evento_href = evento_href_element.get_attribute('href')
                                        tipo_evento = evento_href.split('#')[1] if '#' in evento_href else evento_href
                                        eventos_jornada.append(tipo_evento)
                                except:
                                    continue

                        datos_jornada.update({
                            'Puntuacion': score,
                            'Eventos': " | ".join(eventos_jornada) if eventos_jornada else ""
                        })

                        bar_negatives = element.find_elements(By.XPATH, './/div[contains(@class, "bar negative")]')
                        bar_negative_text = bar_negatives[0].text if bar_negatives else "Sin texto de sanción o lesión"
                        datos_jornada['SancionOLesion'] = bar_negative_text

                        if "Sancionado" in bar_negative_text:
                            datos_jornada['SancionOLesion'] = 'Sancionado'
                            datos_jornadas.append(datos_jornada)
                            log(f"Datos extraídos para {name} {surname} en jornada {gw_text} (Sancionado)")
                            log(f"→ Datos: {datos_jornada}")
                            jornadas_insertadas.append(numero_jornada)
                            continue

                        elif score == "Sin puntuación" and not bar_negatives:
                            datos_jornada['SancionOLesion'] = 'No jugó la jornada'
                            datos_jornadas.append(datos_jornada)
                            log(f"Datos extraídos para {name} {surname} en jornada {gw_text} (No jugó)")
                            log(f"→ Datos: {datos_jornada}")
                            jornadas_insertadas.append(numero_jornada)
                            continue

                        # Extraer estadísticas si hay popup
                        stats_extraidos = False
                        eventos_div = element.find_elements(By.XPATH, './/div[@class="bar"]')

                        for evento in eventos_div:
                            try:
                                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", evento)
                                time.sleep(0.3)
                                driver.execute_script("arguments[0].click();", evento)

                                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "popup")))

                                try:
                                    toast = WebDriverWait(driver, 1).until(
                                        EC.presence_of_element_located((By.ID, "toast"))
                                    )
                                    if "no ha puntuado" in toast.text.lower():
                                        datos_jornada['SancionOLesion'] = 'No jugó la jornada'
                                        stats_extraidos = True
                                        break
                                except:
                                    pass

                                try:
                                    WebDriverWait(driver, 10).until(
                                        EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Ver más estadísticas')]"))
                                    )
                                    boton_stats = driver.find_element(By.XPATH, "//button[contains(text(), 'Ver más estadísticas')]")
                                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", boton_stats)
                                    time.sleep(0.5)
                                    driver.execute_script("arguments[0].click();", boton_stats)
                                except Exception as e:
                                    log(f"No se pudo hacer clic en 'Ver más estadísticas': {e}")
                                    try:
                                        boton_stats_fresco = driver.find_element(By.XPATH, "//button[contains(text(), 'Ver más estadísticas')]")
                                        data_stats = boton_stats_fresco.get_attribute("data-stats")
                                        if data_stats:
                                            import json
                                            stats_dict = json.loads(data_stats.replace('&quot;', '"'))
                                            for k, v in stats_dict.items():
                                                datos_jornada[k] = str(v)
                                            stats_extraidos = True
                                            break
                                    except Exception as e2:
                                        log(f"No se pudieron leer los data-stats directamente tras refrescar: {e2}")
                                    continue

                                tabla = WebDriverWait(driver, 10).until(
                                    EC.presence_of_element_located((By.XPATH, '//div[contains(@class, "content player-breakdown")]'))
                                )
                                filas = tabla.find_elements(By.TAG_NAME, 'tr')
                                for fila in filas:
                                    columnas = fila.find_elements(By.TAG_NAME, 'td')
                                    if len(columnas) == 2:
                                        campo = columnas[0].text.strip()
                                        valor = columnas[1].text.strip()
                                        datos_jornada[campo] = valor

                                try:
                                    popup_close = driver.find_element(By.CSS_SELECTOR, '#popup .popup-close')
                                    driver.execute_script("arguments[0].click();", popup_close)
                                except:
                                    pass

                                stats_extraidos = True
                                break

                            except Exception as e:
                                log(f"Error al intentar extraer estadísticas del popup: {e}")
                                continue

                        if not stats_extraidos and datos_jornada['SancionOLesion'] == "Sin texto de sanción o lesión":
                            datos_jornada['SancionOLesion'] = 'No se pudieron extraer estadísticas'

                        datos_jornadas.append(datos_jornada)
                        log(f"Datos extraídos para {name} {surname} en jornada {gw_text}")
                        log(f"→ Datos: {datos_jornada}")
                        jornadas_insertadas.append(numero_jornada)

                    except Exception:
                        continue

                if jornadas_insertadas:
                    nueva_jornada = min(max(jornadas_insertadas), max_jornada)
                    cur.execute("""
                        INSERT INTO dbo.progreso_jornadas(id_jugador, ultima_jornada)
                        VALUES (%s, %s)
                        ON CONFLICT (id_jugador)
                        DO UPDATE SET ultima_jornada = EXCLUDED.ultima_jornada
                    """, (player_id, nueva_jornada))

                time.sleep(0.3)

        conn.commit()

    log(f"obtener_datos_jornadas_inicial: Finalización con {len(datos_jornadas)} registros procesados")
    return datos_jornadas

def obtener_registros_transferencia(driver):
    log("obtener_registros_transferencia: Inicio de la función")

    wait = WebDriverWait(driver, 2)
    todos_registros = []
    urls_jugadores = obtener_urls_desde_db()

    for player_url in urls_jugadores:
        driver.get(player_url)
        try:
            name = driver.find_element(By.XPATH, '//div[@class="left"]//div[@class="name"]').text
            surname = driver.find_element(By.CLASS_NAME, 'surname').text.strip()
        except Exception as e:
            log(f"obtener_registros_transferencia: Error obteniendo nombre o apellido en {player_url}: {e}")
            continue

        registros_transferencia = []
        player_id = player_url.split("/players/")[1].split("/")[0]
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
                        "precio": precio,
                        "id_jugador": player_id
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
    todos_puntos = []
    
    urls_jugadores = obtener_urls_desde_db()

    for player_url in urls_jugadores:
        driver.get(player_url)
        player_id = player_url.split("/players/")[1].split("/")[0]
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
                        "id_jugador": player_id,
                        "Nombre": name,
                        "Apellido": surname,
                        "top": None,
                        "bottom": None,
                        "right": None
                    }
                    todos_puntos.append(registro)
                    #log(f"obtener_puntos: {name} {surname} no tiene historial de puntos")
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
                    "id_jugador": player_id,
                    "Nombre": name,
                    "Apellido": surname,
                    "top": top,
                    "bottom": bottom,
                    "right": right
                }
                todos_puntos.append(registro)
                #log(f"obtener_puntos: Puntos añadidos para {name} {surname} - top: {top}, bottom: {bottom}, right: {right}")
            except Exception as e:
                log(f"obtener_puntos: Error extrayendo punto de {name} {surname}: {e}")
                continue

    log(f"obtener_puntos: Finalización con {len(todos_puntos)} registros de puntos")
    return todos_puntos


def obtener_valores(driver):
    log("obtener_valores: Inicio de la función")

    wait = WebDriverWait(driver, 2)
    valores = []
    
    urls_jugadores = obtener_urls_desde_db()

    for player_url in urls_jugadores:
        driver.get(player_url)
        player_id = player_url.split("/players/")[1].split("/")[0]
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
                        "id_jugador": player_id,
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
                    "id_jugador": player_id,
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
