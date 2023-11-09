import mysql.connector
from datetime import datetime

def insertar_datos_clasificacion_jornada(cnn):
  fecha_de_carga = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  # Temporada actual (estática)
  temporada_actual = "23/24"

  # Definir los datos a insertar
  datos = [
      (49, 1, temporada_actual, 5, 2, 56, fecha_de_carga, "11", "85.670.000"),
      (50, 3, temporada_actual, 5, 6, 48, fecha_de_carga, "11", "60.868.000"),
      (51, 4, temporada_actual, 5, 8, 36, fecha_de_carga, "11", "84.300.000"),
      (52, 14, temporada_actual, 5, 11, 17, fecha_de_carga, "11", "48.820.000"),
      (53, 6, temporada_actual, 5, 10, 24, fecha_de_carga, "11", "57.312.000"),
      (54, 7, temporada_actual, 5, 3, 55, fecha_de_carga, "11", "59.606.000"),
      (55, 8, temporada_actual, 5, 9, 34, fecha_de_carga, "11", "47.147.000"),
      (56, 9, temporada_actual, 5, 5, 52, fecha_de_carga, "11", "88.607.000"),
      (57, 10, temporada_actual, 5, 1, 57, fecha_de_carga, "11", "68.022.000"),
      (58, 11, temporada_actual, 5, 4, 53, fecha_de_carga, "11", "63.949.000"),
      (59, 12, temporada_actual, 5, 7, 48, fecha_de_carga, "11", "58.244.000"),
      (60, 13, temporada_actual, 5, 12, 0, fecha_de_carga, "0", "0.000.000"),
      (61, 1, temporada_actual, 6, 5, 49, fecha_de_carga, "11", "85.908.000"),
      (62, 3, temporada_actual, 6, 7, 47, fecha_de_carga, "11", "70.832.000"),
      (63, 4, temporada_actual, 6, 1, 62, fecha_de_carga, "11", "92.213.000"),
      (64, 14, temporada_actual, 6, 11, 29, fecha_de_carga, "11", "64.342.000"),
      (65, 6, temporada_actual, 6, 9, 40, fecha_de_carga, "11", "62.997.000"),
      (66, 7, temporada_actual, 6, 2, 58, fecha_de_carga, "11", "72.981.000"),
      (67, 8, temporada_actual, 6, 10, 34, fecha_de_carga, "11", "77.903.000"),
      (68, 9, temporada_actual, 6, 4, 55, fecha_de_carga, "11", "90.774.000"),
      (69, 10, temporada_actual, 6, 6, 47, fecha_de_carga, "11", "71.965.000"),
      (70, 11, temporada_actual, 6, 3, 56, fecha_de_carga, "11", "74.263.000"),
      (71, 12, temporada_actual, 6, 12, 26, fecha_de_carga, "11", "60.842.000"),
      (72, 13, temporada_actual, 6, 8, 43, fecha_de_carga, "11", "62.077.000"),
      (74, 1, temporada_actual, 7, 3, 40, fecha_de_carga, "11", "78.146.000"),
      (75, 3, temporada_actual, 7, 6, 33, fecha_de_carga, "11", "62.592.000"),
      (76, 4, temporada_actual, 7, 10, 19, fecha_de_carga, "11", "96.360.000"),
      (77, 14, temporada_actual, 7, 12, 11, fecha_de_carga, "11", "30.223.000"),
      (78, 6, temporada_actual, 7, 4, 34, fecha_de_carga, "9", "55.402.000"),
      (79, 7, temporada_actual, 7, 5, 33, fecha_de_carga, "11", "81.367.000"),
      (80, 8, temporada_actual, 7, 7, 33, fecha_de_carga, "11", "39.026.000"),
      (81, 9, temporada_actual, 7, 2, 45, fecha_de_carga, "11", "83.368.000"),
      (82, 10, temporada_actual, 7, 1, 48, fecha_de_carga, "11", "78.842.000"),
      (83, 11, temporada_actual, 7, 11, 18, fecha_de_carga, "11", "70.534.000"),
      (84, 12, temporada_actual, 7, 8, 32, fecha_de_carga, "11", "59.639.000"),
      (85, 13, temporada_actual, 7, 9, 20, fecha_de_carga, "11", "54.923.000"),
      (86, 1, temporada_actual, 8, 7, 41, fecha_de_carga, "11", "89.195.000"),
      (87, 3, temporada_actual, 8, 4, 48, fecha_de_carga, "11", "63.008.000"),
      (88, 4, temporada_actual, 8, 3, 58, fecha_de_carga, "11", "95.253.000"),
      (89, 14, temporada_actual, 8, 12, 8, fecha_de_carga, "11", "28.452.000"),
      (90, 6, temporada_actual, 8, 5, 44, fecha_de_carga, "11", "64.185.000"),
      (91, 7, temporada_actual, 8, 2, 67, fecha_de_carga, "11", "87.610.000"),
      (92, 8, temporada_actual, 8, 6, 44, fecha_de_carga, "11", "56.648.000"),
      (93, 9, temporada_actual, 8, 9, 34, fecha_de_carga, "11", "91.895.000"),
      (94, 10, temporada_actual, 8, 1, 78, fecha_de_carga, "11", "78.397.000"),
      (95, 11, temporada_actual, 8, 8, 36, fecha_de_carga, "11", "68.316.000"),
      (96, 12, temporada_actual, 8, 10, 33, fecha_de_carga, "11", "67.517.000"),
      (85, 13, temporada_actual, 8, 11, 26, fecha_de_carga, "11", "51.309.000")
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