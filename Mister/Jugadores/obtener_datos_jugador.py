from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException,StaleElementReferenceException,TimeoutException
from selenium.webdriver.common.keys import Keys
import re
from utils import log,obtener_urls_desde_db
import unicodedata
from utils import conexion_db, obtener_temporada_actual
import time

def normalizar_label(texto):
    return unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8').lower().strip()

def sesion_activa(driver):
    try:
        driver.find_element(By.XPATH, '//a[contains(@class, "btn-play") and contains(text(), "Jugar")]')
        return False  # botón "Jugar" = no logueado
    except NoSuchElementException:
        return True

def obtener_urls_jugadores(driver, schema=None):
    _ = schema
    log("obtener_urls_jugadores: Inicio")

    wait = WebDriverWait(driver, 5)
    datos_urls = []

    # --- NUEVO: ir directamente a la pestaña "Buscar" ---
    try:
        enlace_buscar = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//ul[@class='menu']//li[@data-pag='search']/a")
        ))
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", enlace_buscar)
        enlace_buscar.click()
        log("obtener_urls_jugadores: Enlace 'Buscar' clickeado")
    except Exception as e:
        log(f"obtener_urls_jugadores: No se pudo clickar 'Buscar' (continuo): {e}")

    # --- Asegurar que hay lista de jugadores ---
    players_xpath = "//ul[contains(@class,'player-list') and contains(@class,'search-players-list')]/li"
    try:
        wait.until(EC.presence_of_all_elements_located((By.XPATH, players_xpath)))
    except Exception:
        log("obtener_urls_jugadores: No se detectó lista inicial de jugadores")
        return []

    # --- Bajar al final para que aparezca el botón ---
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
    time.sleep(0.2)

    # --- Cargar “Ver más” esperando incremento de <li> ---
    while True:
        try:
            antes = len(driver.find_elements(By.XPATH, players_xpath))
            button = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button.search-players-more"))
            )
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", button)
            time.sleep(0.15)
            button.click()
            WebDriverWait(driver, 8).until(
                lambda d: len(d.find_elements(By.XPATH, players_xpath)) > antes
            )
            log("obtener_urls_jugadores: Botón 'Ver más' clickeado")
        except Exception:
            break

    players = driver.find_elements(By.XPATH, players_xpath)

    # --- Regex que admite href relativo o absoluto ---
    patron = re.compile(r'(?:^|/)players/(\d+)/(?:profile/)?([\w\-]+)', re.IGNORECASE)

    for player in players:
        try:
            a = player.find_element(By.CSS_SELECTOR, "a.btn.btn-sw-link.player")
            url = (a.get_attribute('href') or "").strip()
            m = patron.search(url)
            if not m:
                log(f"obtener_urls_jugadores: href sin match: {url}")
                continue

            id_jugador = m.group(1)
            nombre_apellido = m.group(2).replace("-", " ").title()
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


def obtener_datos_jugador(driver, schema=None):
    _ = schema
    log("obtener_datos_jugador: Inicio de la función")
    temporada = obtener_temporada_actual()
    datos_de_jugadores = []
    urls_jugadores = obtener_urls_desde_db(schema)

    for player_url in urls_jugadores:
        datos_jugador = {}
        driver.get(player_url)
        log(f"Accediendo a perfil: {player_url}")

        try:
            # Nombre y apellido
            name = driver.find_element(By.CSS_SELECTOR, ".player-profile-header .name").text.strip()
            surname = driver.find_element(By.CSS_SELECTOR, ".player-profile-header .surname").text.strip()

            # Equipo
            team_element = driver.find_element(By.CSS_SELECTOR, ".player-profile-header .team-position a")
            equipo = team_element.get_attribute("href").split("/")[-1].replace("-", " ").title()
        except Exception as e:
            log(f"Error obteniendo nombre, apellido o equipo: {e}")
            continue

        try:
            # Posición
            position_element = driver.find_element(By.CSS_SELECTOR, ".player-profile-header .player-position")
            position_number = position_element.get_attribute("data-position")
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

        # Estadísticas principales
        try:
            stats_wrapper = driver.find_element(By.CLASS_NAME, 'player-stats-wrapper')
            stats_items = stats_wrapper.find_elements(By.CLASS_NAME, 'item')
            stats_dict = {}
            for item in stats_items:
                label = item.find_element(By.CLASS_NAME, 'label').text.strip()
                value = item.find_element(By.CLASS_NAME, 'value').text.strip()
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

        # Historial de movimientos (fichajes)
        try:
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

                fecha = left.find_element(By.CLASS_NAME, "label").text.strip()
                texto_valor = left.find_element(By.CLASS_NAME, "value").text.strip()
                precio = right.text.strip()

                match = re.search(r'De\s+(.*?)\s+a\s+(.*)', texto_valor)
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

        # Alertas (player-notices)
        try:
            notices_box = driver.find_elements(By.CSS_SELECTOR, ".player-notices .box")
            if not notices_box:
                datos_jugador['Alerta'] = "Jugador sin alertas"
            else:
                alertas = []
                for box in notices_box:
                    texto = " ".join(box.text.strip().split())
                    if texto:
                        alertas.append(texto)
                datos_jugador['Alerta'] = " | ".join(alertas)
        except Exception as e:
            log(f"Error extrayendo alertas: {e}")
            datos_jugador['Alerta'] = "Jugador sin alertas"

        datos_de_jugadores.append(datos_jugador)
        log(f"Jugador procesado: {datos_jugador}")

    log(f"obtener_datos_jugador: Finalización con {len(datos_de_jugadores)} jugadores procesados")
    return datos_de_jugadores


def obtener_datos_jornadas(driver, schema):
    log("obtener_datos_jornadas: Inicio de la función")
    temporada = obtener_temporada_actual()
    wait = WebDriverWait(driver, 2)
    datos_jornadas = []
    urls_jugadores = obtener_urls_desde_db(schema)

    with conexion_db() as conn:
        with conn.cursor() as cur:

            for player_url in urls_jugadores:
                driver.get(player_url)
                time.sleep(0.5)

                try:
                    name = driver.find_element(By.XPATH, '//div[@class="left"]//div[@class="name"]').text.strip()
                    surname = driver.find_element(By.CLASS_NAME, 'surname').text.strip()
                except Exception:
                    continue

                player_id = player_url.split("/players/")[1].split("/")[0]

                # recuperar progreso por temporada
                try:
                    cur.execute(
                        f"SELECT ultima_jornada FROM {schema}.progreso_jornadas "
                        f"WHERE id_jugador = %s AND temporada = %s",
                        (player_id, temporada)
                    )
                    resultado = cur.fetchone()
                    ultima_jornada = resultado[0] if resultado else 0
                except Exception:
                    ultima_jornada = 0

                # recuperar pendientes
                cur.execute(
                    f"SELECT jornada FROM {schema}.progreso_jornadas_pendientes "
                    f"WHERE id_jugador = %s AND temporada = %s",
                    (player_id, temporada)
                )
                pendientes = {row[0] for row in cur.fetchall()}

                try:
                    elements = driver.find_elements(By.XPATH, '//div[contains(@class,"gw btn btn-player-gw")]')
                except Exception:
                    continue

                jornadas_insertadas = []
                jornadas_encontradas = set()

                for element in elements:
                    try:
                        if "gw-played" not in element.get_attribute("class"):
                            continue

                        gw_text = element.find_element(By.CLASS_NAME, "title").text.strip()
                        numero_jornada = int(re.search(r'\d+', gw_text).group())
                        jornadas_encontradas.add(numero_jornada)

                        # saltar jornadas ya procesadas salvo que estén en pendientes
                        if numero_jornada <= ultima_jornada and numero_jornada not in pendientes:
                            continue

                        datos_jornada = {
                            "id_jugador": player_id,
                            "Nombre": name,
                            "Apellido": surname,
                            "Jornada": numero_jornada,
                            "Temporada": temporada
                        }

                        # Puntuación
                        try:
                            score_div = element.find_element(By.XPATH, './/div[contains(@class,"bar")]/div')
                            score = score_div.text.strip()
                        except:
                            score = "0"
                        datos_jornada["Puntuacion"] = score

                        # Eventos
                        eventos = []
                        eventos_svg = element.find_elements(By.CSS_SELECTOR, ".events svg use")
                        for ev in eventos_svg:
                            href = ev.get_attribute("href") or ""
                            if "#events-" in href:
                                eventos.append("events-" + href.split("#events-")[1])
                        datos_jornada["Eventos"] = " | ".join(eventos)

                        # --- Detectar estados especiales ---
                        inner_html = element.get_attribute("innerHTML")

                        if "#not-played" in inner_html:
                            datos_jornada["SancionOLesion"] = "No jugó la jornada"
                            datos_jornadas.append(datos_jornada)
                            log(f"{name} {surname} J{numero_jornada}: detectado not-played → No jugó la jornada")
                            jornadas_insertadas.append(numero_jornada)
                            # borrar de pendientes si estaba
                            cur.execute(
                                f"DELETE FROM {schema}.progreso_jornadas_pendientes "
                                f"WHERE id_jugador = %s AND temporada = %s AND jornada = %s",
                                (player_id, temporada, numero_jornada)
                            )
                            continue

                        elif "#injury" in inner_html:
                            datos_jornada["SancionOLesion"] = "Lesionado"
                            datos_jornadas.append(datos_jornada)
                            log(f"{name} {surname} J{numero_jornada}: detectado injury → Lesionado")
                            jornadas_insertadas.append(numero_jornada)
                            cur.execute(
                                f"DELETE FROM {schema}.progreso_jornadas_pendientes "
                                f"WHERE id_jugador = %s AND temporada = %s AND jornada = %s",
                                (player_id, temporada, numero_jornada)
                            )
                            continue

                        elif "#suspension" in inner_html:
                            datos_jornada["SancionOLesion"] = "Sancionado"
                            datos_jornadas.append(datos_jornada)
                            log(f"{name} {surname} J{numero_jornada}: detectado suspension → Sancionado")
                            jornadas_insertadas.append(numero_jornada)
                            cur.execute(
                                f"DELETE FROM {schema}.progreso_jornadas_pendientes "
                                f"WHERE id_jugador = %s AND temporada = %s AND jornada = %s",
                                (player_id, temporada, numero_jornada)
                            )
                            continue

                        elif "#other" in inner_html:
                            datos_jornada["SancionOLesion"] = "No convocado"
                            datos_jornadas.append(datos_jornada)
                            log(f"{name} {surname} J{numero_jornada}: detectado other → No convocado")
                            jornadas_insertadas.append(numero_jornada)
                            cur.execute(
                                f"DELETE FROM {schema}.progreso_jornadas_pendientes "
                                f"WHERE id_jugador = %s AND temporada = %s AND jornada = %s",
                                (player_id, temporada, numero_jornada)
                            )
                            continue

                        else:
                            datos_jornada["SancionOLesion"] = ""

                        # --- Extraer estadísticas detalladas del popup ---
                        stats_extraidos = False
                        try:
                            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", element)
                            time.sleep(0.2)
                            driver.execute_script("arguments[0].click();", element)

                            # esperar popup o toast
                            try:
                                toast = WebDriverWait(driver, 2).until(
                                    EC.presence_of_element_located((By.ID, "toast"))
                                )
                                if "no ha puntuado" in toast.text.lower():
                                    datos_jornada["SancionOLesion"] = "No jugó la jornada"
                                    stats_extraidos = True
                                    log(f"{name} {surname} J{numero_jornada}: detectado toast → No jugó la jornada")
                                    datos_jornadas.append(datos_jornada)
                                    jornadas_insertadas.append(numero_jornada)
                                    cur.execute(
                                        f"DELETE FROM {schema}.progreso_jornadas_pendientes "
                                        f"WHERE id_jugador = %s AND temporada = %s AND jornada = %s",
                                        (player_id, temporada, numero_jornada)
                                    )
                                    continue
                            except TimeoutException:
                                pass

                            WebDriverWait(driver, 5).until(
                                EC.presence_of_element_located((By.ID, "popup"))
                            )

                            try:
                                boton_stats = WebDriverWait(driver, 3).until(
                                    EC.presence_of_element_located((By.CSS_SELECTOR, "button[data-popup='player-breakdown']"))
                                )
                                driver.execute_script("arguments[0].click();", boton_stats)
                            except TimeoutException:
                                if datos_jornada.get("Eventos"):
                                    log(f"{name} {surname} J{numero_jornada}: sin estadísticas avanzadas pero con eventos")
                                    datos_jornada["SancionOLesion"] = ""
                                else:
                                    log(f"{name} {surname} J{numero_jornada}: jugó pero Mister no da estadísticas → Sin estadísticas avanzadas")
                                    datos_jornada["SancionOLesion"] = "Sin estadísticas avanzadas"

                                stats_extraidos = True
                                datos_jornadas.append(datos_jornada)
                                jornadas_insertadas.append(numero_jornada)
                                cur.execute(
                                    f"DELETE FROM {schema}.progreso_jornadas_pendientes "
                                    f"WHERE id_jugador = %s AND temporada = %s AND jornada = %s",
                                    (player_id, temporada, numero_jornada)
                                )
                                continue

                            tabla = WebDriverWait(driver, 5).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, "div.content.player-breakdown table"))
                            )
                            filas = tabla.find_elements(By.TAG_NAME, "tr")
                            for fila in filas:
                                columnas = fila.find_elements(By.TAG_NAME, "td")
                                if len(columnas) == 2:
                                    campo = columnas[0].text.strip()
                                    valor = columnas[1].text.strip()
                                    datos_jornada[campo] = valor

                            stats_extraidos = True

                            try:
                                popup_close = driver.find_element(By.CSS_SELECTOR, "#popup .popup-close")
                                driver.execute_script("arguments[0].click();", popup_close)
                            except:
                                pass

                        except Exception as e:
                            log(f"No se pudieron extraer estadísticas para {name} {surname} J{numero_jornada}: {e}")

                        if not stats_extraidos and datos_jornada["SancionOLesion"] == "":
                            datos_jornada["SancionOLesion"] = "No se pudieron extraer estadísticas"

                        datos_jornadas.append(datos_jornada)
                        log(f"Datos extraídos para {name} {surname} en jornada {numero_jornada}")
                        log(f"→ Datos: {datos_jornada}")
                        jornadas_insertadas.append(numero_jornada)

                        # borrar pendiente si estaba
                        cur.execute(
                            f"DELETE FROM {schema}.progreso_jornadas_pendientes "
                            f"WHERE id_jugador = %s AND temporada = %s AND jornada = %s",
                            (player_id, temporada, numero_jornada)
                        )

                    except Exception as e:
                        log(f"Error procesando jornada {numero_jornada} para {name} {surname}: {e}")
                        continue

                # actualizar progreso
                if jornadas_insertadas:
                    nueva_jornada = max(jornadas_insertadas)
                    cur.execute(f"""
                        INSERT INTO {schema}.progreso_jornadas(id_jugador, ultima_jornada, temporada)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (id_jugador, temporada)
                        DO UPDATE SET ultima_jornada = EXCLUDED.ultima_jornada
                    """, (player_id, nueva_jornada, temporada))

                    jornadas_teoricas = set(range(ultima_jornada + 1, max(jornadas_insertadas) + 1))
                    jornadas_faltantes = jornadas_teoricas - set(jornadas_insertadas)

                    for j_faltante in jornadas_faltantes:
                        cur.execute(f"""
                            INSERT INTO {schema}.progreso_jornadas_pendientes(id_jugador, temporada, jornada)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (id_jugador, temporada, jornada) DO NOTHING
                        """, (player_id, temporada, j_faltante))

                time.sleep(0.3)

        conn.commit()

    log(f"obtener_datos_jornadas: Finalización con {len(datos_jornadas)} registros procesados")
    return datos_jornadas

def obtener_datos_jornadas_inicial(driver, schema, max_jornada=34, salto=2):
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
                    cur.execute(f"SELECT ultima_jornada FROM {schema}.progreso_jornadas WHERE id_jugador = %s", (player_id,))
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
                    cur.execute(f"""
                        INSERT INTO {schema}.progreso_jornadas(id_jugador, ultima_jornada)
                        VALUES (%s, %s)
                        ON CONFLICT (id_jugador)
                        DO UPDATE SET ultima_jornada = EXCLUDED.ultima_jornada
                    """, (player_id, nueva_jornada))

                time.sleep(0.3)

        conn.commit()

    log(f"obtener_datos_jornadas_inicial: Finalización con {len(datos_jornadas)} registros procesados")
    return datos_jornadas

def obtener_registros_transferencia(driver,schema=None):
    _ = schema
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


def obtener_puntos(driver, schema=None):
    _ = schema
    log("obtener_puntos: Inicio de la función")
    wait = WebDriverWait(driver, 5)
    todos_puntos = []
    urls_jugadores = obtener_urls_desde_db(schema)

    for player_url in urls_jugadores:
        driver.get(player_url)
        player_id = player_url.split("/players/")[1].split("/")[0]

        # --- Nombre y apellido ---
        try:
            name = driver.find_element(By.CSS_SELECTOR, "div.player-profile-header div.name").text.strip()
            surname = driver.find_element(By.CSS_SELECTOR, "div.player-profile-header div.surname").text.strip()
        except Exception as e:
            log(f"obtener_puntos: Error obteniendo nombre/apellido en {player_url}: {e}")
            continue

        # --- Historial de puntos ---
        try:
            # Buscar el bloque con h4 = Historial de puntos
            historial_puntos_container = driver.find_element(
                By.XPATH, "//h4[text()='Historial de puntos']/ancestor::div[@class='box-wrapper']//div[@class='box box-records']/ul"
            )
            puntos_items = historial_puntos_container.find_elements(By.TAG_NAME, "li")
        except Exception as e:
            log(f"obtener_puntos: Error obteniendo historial de puntos para {name} {surname}: {e}")
            continue

        if not puntos_items:
            registro = {
                "id_jugador": player_id,
                "Nombre": name,
                "Apellido": surname,
                "top": None,
                "bottom": None,
                "right": None
            }
            todos_puntos.append(registro)
            continue

        for item in puntos_items:
            try:
                try:
                    top = item.find_element(By.CLASS_NAME, "label").text.strip()
                except NoSuchElementException:
                    top = None
                try:
                    bottom = item.find_element(By.CLASS_NAME, "value").text.strip()
                except NoSuchElementException:
                    bottom = None
                try:
                    right = item.find_element(By.CLASS_NAME, "right").text.strip()
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
                # log(f"obtener_puntos: {name} {surname} - top: {top}, bottom: {bottom}, right: {right}")
            except Exception as e:
                log(f"obtener_puntos: Error extrayendo punto de {name} {surname}: {e}")
                continue

    log(f"obtener_puntos: Finalización con {len(todos_puntos)} registros de puntos")
    return todos_puntos


def obtener_valores(driver,schema=None):
    _ = schema
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
