import mysql.connector

def connect_to_mysql():
    # Crear una conexión a la base de datos
    try:
        cnn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="J1f2a3o4**",
            database="chavalitos"
        )
        print("Conexión a la base de datos exitosa.")
    except mysql.connector.Error as error:
        print("Error al conectarse a la base de datos:", error)
    return cnn