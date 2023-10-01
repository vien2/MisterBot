import mysql.connector
from datetime import datetime

def insertar_datos_clasificacion_jornada(cnn):
  fecha_de_carga = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  # Temporada actual (estática)
  temporada_actual = "23/24"

  # Definir los datos a insertar
  datos = [
      (13, 1, temporada_actual, 2, 1, 57, fecha_de_carga, "11", "71.511.000"),
      (14, 3, temporada_actual, 2, 5, 44, fecha_de_carga, "11", "40.130.000"),
      (15, 4, temporada_actual, 2, 6, 40, fecha_de_carga, "11", "61.059.000"),
      (16, 14, temporada_actual, 2, 12, 8, fecha_de_carga, "9", "37.917.000"),
      (17, 6, temporada_actual, 2, 10, 30, fecha_de_carga, "11", "46.012.000"),
      (18, 7, temporada_actual, 2, 4, 44, fecha_de_carga, "11", "48.793.000"),
      (19, 8, temporada_actual, 2, 8, 35, fecha_de_carga, "11", "43.492.000"),
      (20, 9, temporada_actual, 2, 2, 48, fecha_de_carga, "11", "53.149.000"),
      (21, 10, temporada_actual, 2, 11, 29, fecha_de_carga, "11", "46.187.000"),
      (22, 11, temporada_actual, 2, 9, 33, fecha_de_carga, "11", "57.661.000"),
      (23, 12, temporada_actual, 2, 7, 36, fecha_de_carga, "11", "54.709.000"),
      (24, 13, temporada_actual, 2, 3, 47, fecha_de_carga, "11", "53.149.000"),
      (25, 1, temporada_actual, 3, 1, 45, fecha_de_carga, "11", "78.379.000"),
      (26, 3, temporada_actual, 3, 11, 26, fecha_de_carga, "11", "53.009.000"),
      (27, 4, temporada_actual, 3, 6, 38, fecha_de_carga, "11", "68.808.000"),
      (28, 14, temporada_actual, 3, 12, 15, fecha_de_carga, "9", "36.533.000"),
      (29, 6, temporada_actual, 3, 10, 28, fecha_de_carga, "11", "48.838.000"),
      (30, 7, temporada_actual, 3, 7, 37, fecha_de_carga, "11", "45.197.000"),
      (31, 8, temporada_actual, 3, 5, 40, fecha_de_carga, "11", "45.818.000"),
      (32, 9, temporada_actual, 3, 9, 32, fecha_de_carga, "11", "63.558.000"),
      (33, 10, temporada_actual, 3, 4, 42, fecha_de_carga, "11", "49.237.000"),
      (34, 11, temporada_actual, 3, 2, 45, fecha_de_carga, "11", "66.846.000"),
      (35, 12, temporada_actual, 3, 3, 45, fecha_de_carga, "11", "55.139.000"),
      (36, 13, temporada_actual, 3, 8, 35, fecha_de_carga, "11", "55.229.000"),
      (37, 1, temporada_actual, 4, 8, 35, fecha_de_carga, "11", "89.160.000"),
      (38, 3, temporada_actual, 4, 6, 33, fecha_de_carga, "11", "56.048.000"),
      (39, 4, temporada_actual, 4, 4, 38, fecha_de_carga, "10", "69.587.000"),
      (40, 14, temporada_actual, 4, 12, 14, fecha_de_carga, "10", "38.153.000"),
      (41, 6, temporada_actual, 4, 7, 29, fecha_de_carga, "9", "61.315.000"),
      (42, 7, temporada_actual, 4, 3, 40, fecha_de_carga, "11", "58.810.000"),
      (43, 8, temporada_actual, 4, 5, 37, fecha_de_carga, "10", "46.439.000"),
      (44, 9, temporada_actual, 4, 1, 52, fecha_de_carga, "11", "66.557.000"),
      (45, 10, temporada_actual, 4, 2, 42, fecha_de_carga, "11", "57.294.000"),
      (46, 11, temporada_actual, 4, 11, 17, fecha_de_carga, "6", "64.404.000"),
      (47, 12, temporada_actual, 4, 10, 20, fecha_de_carga, "10", "56.336.000"),
      (48, 13, temporada_actual, 4, 9, 25, fecha_de_carga, "10", "62.281.000")
  ]

  # Definir la consulta SQL para insertar los datos
  sql = "INSERT INTO clasificacion_jornada (ID, UsuarioId, Temporada, Jornada, Posicion, Puntos, f_carga, Jugadores, valor) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"

  try:
    cursor = cnn.cursor()
      
    # Ejecutar la consulta para insertar los datos
    cursor.executemany(sql, datos)
  
    # Confirmar los cambios en la base de datos
    cnn.commit()

    # Cerrar el cursor y la conexión a la base de datos
    cursor.close()
    cnn.close()

    print("Registros insertados.")
              
  except mysql.connector.Error as error:
      print("Error al insertar datos en la tabla Clasificacion_general:", error)