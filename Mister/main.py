import sys
import os
import json
import pandas as pd
from utils import (
    log, guardar_en_csv, aplanar_datos, get_base_path_from_ini,
    añadir_temporada, añadir_f_carga, añadir_hash, conexion_db,
    registrar_ejecucion_carga
)
from configuracion import get_filename_config, get_funciones_disponibles
from patterns.csv_postgresql import cargar_csv_postgresql
from iniciar_sesion import iniciar_sesion


def ejecutar_proceso(id_load: int):
    estado = 'KO'
    try:
        log(f"Inicio de proceso para ID Load: {id_load}")
        base_path = get_base_path_from_ini()

        # --- Lee la configuración del load ---
        with conexion_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT fichero AS nombre_fichero, tabla, tipo_extraccion, tipo_carga, "schema",
                           incremental_field, clave_conflicto, usa_hash, usa_driver, parametros
                    FROM dbo.load
                    WHERE idload = %s
                """, (id_load,))
                row = cur.fetchone()

        if not row:
            log(f"No se encontró configuración para idload {id_load}")
            return

        (nombre_fichero, tabla, tipo_extraccion, tipo_carga,
         schema, incremental_field, clave_conflicto,
         usa_hash, usa_driver, parametros) = row

        # --- Parseo seguro de 'parametros' ---
        params = {}
        if parametros:
            if isinstance(parametros, dict):
                params = parametros
            else:
                try:
                    params = json.loads(parametros)
                except Exception:
                    log("WARN: 'parametros' no es JSON válido. Usando {}.")

        FUNCIONES_DISPONIBLES = get_funciones_disponibles()
        if nombre_fichero not in FUNCIONES_DISPONIBLES:
            log(f"No se encontró una función mapeada para '{nombre_fichero}'")
            return
        funcion = FUNCIONES_DISPONIBLES[nombre_fichero]

        # ---------------------------
        # CASO POSTPROCESO
        # ---------------------------
        if tipo_extraccion == "postproceso":
            log(f"Ejecutando postproceso: {nombre_fichero}")

            if tipo_carga == "psql":
                with conexion_db() as conn:
                    funcion(conn, schema=schema, **params)
                log("Postproceso SQL finalizado correctamente.")
                estado = "OK"
                return

            elif tipo_carga == "accion":
                # PRODUCCIÓN: headless=True
                driver = iniciar_sesion(schema=schema, headless=True)
                try:
                    funcion(driver, schema=schema, **params)
                    log("Postproceso Selenium finalizado correctamente.")
                    estado = "OK"
                finally:
                    try:
                        driver.quit()
                    except Exception as e:
                        log(f"Warning al cerrar driver: {e}")
                return

            else:
                log(f"Tipo de carga '{tipo_carga}' no reconocido para postproceso.")
                return

        # ---------------------------
        # CASO EXTRACCIÓN (scraping normal)
        # ---------------------------
        log(f"Ejecutando extracción: {nombre_fichero}")
        if usa_driver:
            # PRODUCCIÓN: headless=True
            driver = iniciar_sesion(schema=schema, headless=True)
            try:
                datos = funcion(driver, schema=schema, **params)
            finally:
                try:
                    driver.quit()
                except Exception as e:
                    log(f"Warning al cerrar driver: {e}")
        else:
            datos = funcion(None, schema=schema, **params)

        # Añadir temporada y f_carga / hash
        datos = añadir_temporada(datos)
        df = pd.DataFrame(datos)
        if usa_hash:
            df = añadir_hash(df, schema=schema, tabla=tabla)
        df = añadir_f_carga(df)
        datos_lista = aplanar_datos(df.to_dict(orient="records"))

        # Guardar CSV
        filename_config = get_filename_config(nombre_fichero, schema=schema)
        ruta_csv = os.path.join(base_path, filename_config["archivo"])
        guardar_en_csv(datos_lista, base_path, filename_config)

        # clave_conflicto -> lista (si viene como JSON)
        if clave_conflicto and isinstance(clave_conflicto, str):
            try:
                clave_conflicto = json.loads(clave_conflicto)
            except Exception:
                log("WARN: 'clave_conflicto' no es JSON válido. Ignorando…")
                clave_conflicto = None

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
        estado = "OK"

    except Exception as e:
        log(f"Error en la ejecución: {e}")
    finally:
        try:
            registrar_ejecucion_carga(id_load, estado)
        except Exception as e:
            log(f"registrar_ejecucion_carga: fallo al registrar → {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        log("Debes proporcionar un ID de carga como argumento.")
    else:
        try:
            ejecutar_proceso(int(sys.argv[1]))
        except Exception as e:
            log(f"Error en la ejecución: {e}")
