import sys
import os
import json
import pandas as pd
from utils import (
    log, guardar_en_csv, aplanar_datos, get_base_path_from_ini,
    añadir_temporada, añadir_f_carga, añadir_hash, conexion_db
)
from configuracion import get_filename_config, get_funciones_disponibles
from patterns.csv_postgresql import cargar_csv_postgresql
from iniciar_sesion import iniciar_sesion


def ejecutar_proceso(id_load):
    log(f"Inicio de proceso para ID Load: {id_load}")
    base_path = get_base_path_from_ini()

    with conexion_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT fichero AS nombre_fichero, tabla, tipo_carga, "schema", incremental_field, clave_conflicto, usa_hash
                FROM dbo.load 
                WHERE idload = %s
            """, (id_load,))
            row = cur.fetchone()

    if not row:
        log(f"No se encontró configuración para idload {id_load}")
        return

    nombre_fichero, tabla, tipo_carga, schema, incremental_field, clave_conflicto, usa_hash = row
    FUNCIONES_DISPONIBLES = get_funciones_disponibles()

    if nombre_fichero not in FUNCIONES_DISPONIBLES:
        log(f"No se encontró una función mapeada para '{nombre_fichero}'")
        return

    log(f"Ejecutando extracción: {nombre_fichero}")
    driver = iniciar_sesion(schema=schema)
    funcion = FUNCIONES_DISPONIBLES[nombre_fichero]

    # Ejecutar la función de scraping
    datos = funcion(driver, schema=schema)

    # Añadir temporada
    datos = añadir_temporada(datos)

    # Convertimos a DataFrame para añadir hash si aplica
    df = pd.DataFrame(datos)
    if usa_hash:
        df = añadir_hash(df, schema=schema, tabla=tabla)
    df = añadir_f_carga(df)

    # Pasamos de nuevo a lista si la función final lo requiere
    datos_lista = aplanar_datos(df.to_dict(orient="records"))

    # Guardar CSV
    filename_config = get_filename_config(nombre_fichero, schema=schema)
    ruta_csv = os.path.join(base_path, filename_config["archivo"])
    guardar_en_csv(datos_lista, base_path, filename_config)

    # Clave conflicto como lista
    if clave_conflicto:
        if isinstance(clave_conflicto, str):
            clave_conflicto = json.loads(clave_conflicto)

    # Carga en PostgreSQL
    cargar_csv_postgresql(
        ruta_csv=ruta_csv,
        schema=schema,
        tabla=tabla,
        tipo_carga=tipo_carga,
        incremental_field=incremental_field,
        clave_conflicto=clave_conflicto,
        hash_field="hash" if tipo_carga == "diferencial" else None
    )

    driver.quit()
    log("Proceso finalizado correctamente.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        log("Debes proporcionar un ID de carga como argumento.")
    else:
        try:
            ejecutar_proceso(int(sys.argv[1]))
        except Exception as e:
            log(f"Error en la ejecución: {e}")
