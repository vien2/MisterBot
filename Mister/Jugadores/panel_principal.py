import time
from selenium.webdriver.common.by import By

def datos_tarjeta(driver,schema=None):
    _ = schema

    # Tiempo de espera para cargar contenido adicional (ajusta según la velocidad de carga de la página)
    time.sleep(2)

    # Desplázate hacia abajo varias veces para cargar todas las tarjetas
    SCROLL_PAUSE_TIME = 2

    # Obtén la altura actual de la ventana del navegador
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Desplázate hacia abajo
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
        # Espera a que la página se cargue después de desplazarte
        time.sleep(SCROLL_PAUSE_TIME)
        
        # Calcula la nueva altura después del desplazamiento
        new_height = driver.execute_script("return document.body.scrollHeight")
        
        # Si la altura no cambia, has llegado al final de la página
        if new_height == last_height:
            break
            
        last_height = new_height

    # Ahora todas las tarjetas deberían estar cargadas en la página.

    # Obtén los elementos que representan cada tarjeta
    tarjetas = driver.find_elements(By.XPATH, '//div[@class="card card-player_transfer"]')
    # Crea una lista para guardar los datos de cada tarjeta
    datos_tarjetas = []
    # Itera a través de los elementos y extrae la información deseada
    for tarjeta in tarjetas:
        titulos = tarjeta.find_elements(By.XPATH, '//div[@class="title"]')
        datos_tarjeta = {}
        for titulo in titulos:
            texto_titulo = titulo.text.lower()
            #jugador = tarjeta.find_elements(By.XPATH, './/div[contains(@class, "name")]').text
            evento = "Abandono" if "abadona" in texto_titulo else "Fichaje" if "cambia" in texto_titulo else "Robo"
            # Crea un diccionario con la información de la tarjeta actual y añádelo a la lista
            datos_tarjeta = {
            "titulo": texto_titulo,
            #"jugador": jugador,
            "evento": evento
            }
            datos_tarjetas.append(datos_tarjeta)
    return datos_tarjetas