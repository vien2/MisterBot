from pymongo import MongoClient
import configparser

def ver_coleccion(nombre_coleccion):
    config = configparser.ConfigParser()
    config.read('Mister\config.ini')

    client_uri = config['MongoDB']['client']
    nombre_base_datos = config['MongoDB']['nombre_base_datos']

    try:
        # Conectar a MongoDB
        client = MongoClient(client_uri)
        db = client[nombre_base_datos]

        # Obtener la colección especificada
        coleccion = db[nombre_coleccion]

        # Imprimir todos los documentos en la colección
        for documento in coleccion.find():
            print(documento)
        
        # Cerrar conexión a MongoDB
        client.close()

    except Exception as e:
        print(f"Error al conectar a la colección {nombre_coleccion}: {e}")

# Ejemplo de uso
ver_coleccion('Jugadores')
