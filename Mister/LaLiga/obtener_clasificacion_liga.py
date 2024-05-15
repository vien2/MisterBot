from selenium import webdriver
from bs4 import BeautifulSoup
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException

def obtener_datos_liga(driver):
    wait = WebDriverWait(driver, 10)
    
    # Navega a la sección correcta (asegúrate de ajustar los XPaths según sea necesario)
    try:
        enlace_mas = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Más')]/parent::li/a")))
        enlace_mas.click()
        time.sleep(2)  # Es importante darle tiempo a la página para que cargue completamente los elementos
        
        enlace_jugadores = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'LaLiga')]")))
        enlace_jugadores.click()
        time.sleep(2)  # Espera a que la tabla de clasificación se cargue completamente
        
        # Una vez que estás en la página correcta, extrae el HTML de la tabla
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Suponiendo que tu tabla esté dentro de un div con la clase 'box box-table', ajusta si es necesario
        table = soup.find('div', class_='box box-table')
        rows = table.find_all('tr')[1:]  # Excluye el encabezado de la tabla
        
        data_list = []  # Aquí almacenaremos los datos de cada equipo
        
        for row in rows:
            cols = row.find_all('td')
            equipo_data = {
                "Posicion": cols[0].text.strip(),
                "Escudo": cols[1].find('img')['src'],
                "Equipo": cols[2].text.strip(),
                "PTS": cols[3].text.strip(),
                "PJ": cols[4].text.strip(),
                "PG": cols[5].text.strip(),
                "PE": cols[6].text.strip(),
                "PP": cols[7].text.strip(),
                "DG": cols[8].text.strip() if len(cols) > 8 else 'N/A'  # Asume que DG puede no estar presente para todos
            }
            data_list.append(equipo_data)
        return data_list
    except TimeoutException:
        print("Se ha producido un error de tiempo de espera al intentar navegar.")
        return None
    except NoSuchElementException:
        print("Algún elemento no fue encontrado durante la navegación.")
        return None
