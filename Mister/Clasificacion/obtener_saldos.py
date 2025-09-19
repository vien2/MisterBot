from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
from utils import log

def obtener_saldos(driver, schema=None):
    _ = schema
    log("obtener_saldos: Inicio de la función")

    datos_saldos = []
    wait = WebDriverWait(driver, 5)

    # --- Ir a la pestaña Tabla ---
    try:
        enlace_tabla = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//ul[@class='menu']//li[@data-pag='standings']/a")
        ))
        enlace_tabla.click()
        log("obtener_saldos: Enlace 'Tabla' clickeado")
    except Exception as e:
        log(f"obtener_saldos: Error al hacer clic en 'Tabla': {e}")
        return []

    driver.implicitly_wait(2)

    # --- Pulsar botón 'Ver saldo de los rivales' ---
    try:
        boton_saldo = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//div[@class='bottom-buttons']//button[@data-sw='users-balances']")
        ))
        boton_saldo.click()
        log("obtener_saldos: Botón 'Ver saldo de los rivales' clickeado")
    except Exception as e:
        log(f"obtener_saldos: Error al pulsar el botón de saldos: {e}")
        return []

    # --- Extraer lista de usuarios con saldo ---
    try:
        lista_saldos = wait.until(
            EC.presence_of_all_elements_located(
                (By.XPATH, '//div[contains(@class,"sw-balances")]//li')
            )
        )
    except Exception as e:
        log(f"obtener_saldos: No se encontraron elementos de saldo: {e}")
        return []


    log(f"obtener_saldos: {len(lista_saldos)} elementos encontrados en la lista de saldos")

    for item in lista_saldos:
        try:
            # Nombre del usuario
            name = item.find_element(By.CLASS_NAME, 'name').text.strip()

            # Saldo (limpiar formato de miles con puntos)
            saldo_element = item.find_element(By.CLASS_NAME, 'points')
            saldo_text = saldo_element.get_attribute("textContent").strip()
            saldo_match = re.search(r"-?\d[\d.]*", saldo_text)
            saldo = saldo_match.group(0).replace(".", "") if saldo_match else "0"

            datos_saldos.append({
                "usuario": name,
                "saldo": saldo
            })
        except Exception as e:
            log(f"obtener_saldos: Error procesando un elemento de saldo: {e}")

    log("obtener_saldos: Finalización exitosa de la función")
    return datos_saldos
