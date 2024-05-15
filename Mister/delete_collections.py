from pymongo import MongoClient
import configparser

def borrar_colecciones():
    # Conectar a MongoDB
    config = configparser.ConfigParser()
    config.read('Mister\config.ini')

    client_uri = config['MongoDB']['client']
    nombre_base_datos = config['MongoDB']['nombre_base_datos']

    client = MongoClient(client_uri)
    db = client[nombre_base_datos]
    
    # Obtener la lista de colecciones en la base de datos
    colecciones = db.list_collection_names()
    
    # Borrar cada colección
    for coleccion in colecciones:
        db.drop_collection(coleccion)
        print(f"Colección '{coleccion}' eliminada correctamente")
    
    # Cerrar conexión a MongoDB
    client.close()

# Ejemplo de uso
borrar_colecciones()
