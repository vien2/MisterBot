from utils import log, obtener_temporada_actual

def migrar_datos_chavalitos(conn, schema, **kwargs):
    """
    Postproceso para copiar datos de validación (historico chavalitos) al nuevo esquema.
    Tablas: jornadas_liga, datos_jornadas, best_xi_jornadas.
    Filtra por la temporada actual obtenida de utils.
    """
    temporada = obtener_temporada_actual()
    source_schema = "chavalitos"
    tables = ["jornadas_liga", "datos_jornadas", "best_xi_jornadas"]
    
    log(f"Inicio migración desde {source_schema} a {schema} para temporada {temporada}")

    with conn.cursor() as cur:
        for table in tables:
            try:
                # 1. Borrar datos existentes de esa temporada en destino para evitar duplicados (Idempotencia)
                log(f"Limpiando {schema}.{table} para temporada {temporada}...")
                cur.execute(f'DELETE FROM "{schema}"."{table}" WHERE temporada = %s', (temporada,))
                
                # 2. Insertar selectivo desde origen
                log(f"Copiando datos de {source_schema}.{table}...")
                query = f'''
                    INSERT INTO "{schema}"."{table}"
                    SELECT * FROM "{source_schema}"."{table}"
                    WHERE temporada = %s
                '''
                cur.execute(query, (temporada,))
                rows_copied = cur.rowcount
                log(f"Migración {table}: {rows_copied} registros copiados.")
                
            except Exception as e:
                log(f"Error migrando tabla {table}: {e}")
                raise e
    
    log("Fin migración de datos chavalitos.")
