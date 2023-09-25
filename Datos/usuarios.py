import mysql.connector
from datetime import datetime
"""
def insertar_datos_usuarios(cnn, datos_usuarios):
    # Obtener la fecha de carga actual en formato date
    fecha_de_carga = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        cursor = cnn.cursor()

        for usuario_data in datos_usuarios:
            usuario = usuario_data["Usuario"]
            # Consulta SQL con marcadores de posición
            sql = "INSERT INTO usuarios (Nombre, OtrosAtributos, f_carga) VALUES (%s, %s, %s)"
            # Datos a insertar
            data = (usuario,None, fecha_de_carga)
            # Ejecutar la consulta con los datos proporcionados
            cursor.execute(sql, data)
            # Hacer commit para guardar los cambios
            cnn.commit()
        print("Datos de Usuarios insertados correctamente.")
    except mysql.connector.Error as error:
        print("Error al insertar datos en la tabla usuarios:", error)
"""
def insertar_datos_usuarios(cnn, datos_usuarios):
    fecha_de_carga = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        cursor = cnn.cursor()

        for usuario_data in datos_usuarios:
            usuario = usuario_data["Usuario"]

            # Consultar si el usuario ya existe en la tabla
            sql_select = "SELECT * FROM usuarios WHERE Nombre = %s"
            cursor.execute(sql_select, (usuario,))
            existing_user = cursor.fetchone()

            if existing_user:
                # Si el usuario ya existe, comprobar si los datos son diferentes antes de actualizar
                if (existing_user[1] != usuario):
                    # Los datos son diferentes, realizar un UPDATE
                    sql_update = "UPDATE usuarios SET Nombre = %s, f_carga = %s WHERE Nombre = %s"
                    data_update = (usuario, fecha_de_carga, usuario)
                    cursor.execute(sql_update, data_update)
                    print(f"Datos del usuario {usuario} actualizados correctamente.")
                else:
                    # Los datos son iguales, no hacer nada
                    print(f"Datos del usuario {usuario} son iguales, no se realizaron cambios.")
            else:
                # Si el usuario no existe, realizar un INSERT
                sql_insert = "INSERT INTO usuarios (Nombre, OtrosAtributos, f_carga) VALUES (%s, %s, %s)"
                data_insert = (usuario, None, fecha_de_carga)
                cursor.execute(sql_insert, data_insert)
                print(f"Usuario {usuario} insertado correctamente.")

            # Hacer commit para guardar los cambios
            cnn.commit()

        print("Datos de Usuarios insertados/actualizados correctamente.")
    except mysql.connector.Error as error:
        print("Error al insertar/actualizar datos en la tabla usuarios:", error)