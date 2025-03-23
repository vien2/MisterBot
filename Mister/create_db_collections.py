from pymongo import MongoClient
import configparser

def crear_base_datos_y_colecciones():
    config = configparser.ConfigParser()
    config.read('Mister\config.ini')

    client_uri = config['MongoDB']['client']
    nombre_base_datos = config['MongoDB']['nombre_base_datos']
    colecciones = config['MongoDB']['colecciones'].split(',')
    
    # Seleccionar o crear la base de datos
    client = MongoClient(client_uri)
    db = client[nombre_base_datos]
    
    # Crear las colecciones si no existen
    for nombre_coleccion in colecciones:
        if nombre_coleccion not in db.list_collection_names():
            db.create_collection(nombre_coleccion)
            print(f"Colección '{nombre_coleccion}' creada en la base de datos '{nombre_base_datos}'")
        else:
            print(f"La colección '{nombre_coleccion}' ya existe en la base de datos '{nombre_base_datos}'")