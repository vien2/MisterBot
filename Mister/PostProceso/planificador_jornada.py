import sys
import os
from datetime import datetime

# === RUTA DE PROYECTO: sube 1 nivel para pillar utils.py ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

import pandas as pd
try:
    from utils import log, conexion_db
except Exception as e:
    print("ERROR importando utils:", e)
    raise

# ---- helper: escribe en log y también en consola ----
def slog(msg: str):
    try:
        log(msg)
    finally:
        print(msg)

def planificar_jornada(conn=None, schema="chavalitos", usuario="vien2", temporada="25/26"):
    slog("planificar_jornada: Inicio")

    # Si no se pasa conexión, usar el context manager
    if conn is None:
        with conexion_db() as conn_interna:
            return planificar_jornada(conn_interna, schema, usuario, temporada)

    # 1) Saldo
    saldo_df = pd.read_sql(
        f"SELECT saldo FROM {schema}.saldos WHERE usuario = %s AND temporada = %s",
        conn, params=(usuario, temporada)
    )
    if saldo_df.empty:
        slog(f"planificar_jornada: No se encontró saldo para {usuario}")
        return

    saldo = float(saldo_df.iloc[0]['saldo'])
    slog(f"Saldo disponible: {saldo:,.0f} €")

    # 2) Plantilla propia
    plantilla = pd.read_sql(
        f"SELECT * FROM {schema}.v_datos_jugador WHERE propietario = %s AND temporada = %s",
        conn, params=(usuario, temporada)
    )
    if plantilla.empty:
        slog("No hay jugadores en tu plantilla para esta temporada.")
        return

    for col in ("media", "puntos", "clausula", "valor"):
        if col in plantilla.columns:
            plantilla[col] = pd.to_numeric(plantilla[col], errors="coerce")
    plantilla["alerta"] = plantilla.get("alerta", "").fillna("")

    # 2b) Rendimiento reciente (últimas 3 jornadas)
    try:
        recientes = pd.read_sql(
            f"""SELECT id_jugador, AVG(puntuacion) AS media_3j 
                FROM (
                    SELECT id_jugador, puntuacion,
                           ROW_NUMBER() OVER (PARTITION BY id_jugador ORDER BY jornada DESC) AS rn
                    FROM {schema}.v_datos_jornadas
                    WHERE temporada = %s
                ) t
                WHERE rn <= 3
                GROUP BY id_jugador""",
            conn, params=(temporada,)
        )
        plantilla = plantilla.merge(recientes, on="id_jugador", how="left")
    except Exception as e:
        slog(f"ERROR calculando medias recientes: {type(e).__name__}: {e}")
        plantilla["media_3j"] = None

    # 2c) Filtrar lesionados/sancionados/dudas
    alineables = plantilla[~plantilla['alerta'].str.contains("Lesionado|Duda|Sancionado", case=False, na=False)]

    # XI según formación 1-3-3-4 usando media_3j si existe
    posiciones = {"PT": 1, "DF": 3, "MC": 3, "DL": 4}
    once = []
    for pos, n in posiciones.items():
        candidatos = alineables[alineables['posicion'] == pos].copy()
        if "media_3j" in candidatos.columns and candidatos["media_3j"].notna().any():
            candidatos["ranking"] = candidatos["media_3j"]
        else:
            candidatos["ranking"] = candidatos["media"]
        candidatos = candidatos.sort_values(by=["ranking", "puntos"], ascending=[False, False]).head(n)
        once.append(candidatos)
    xi_recomendado = pd.concat(once) if once else pd.DataFrame()

    slog("XI recomendado (1-3-3-4) [ponderado últimas 3 jornadas]:")
    if xi_recomendado.empty:
        slog("  *No hay suficientes jugadores alineables para completar el XI.*")
    else:
        for _, row in xi_recomendado.iterrows():
            slog(f" - {row.get('nombre','?')} {row.get('apellido','?')} ({row.get('posicion','?')}) "
                 f"| Media {row.get('media',0):.2f} | Media3J {row.get('media_3j',0):.2f} "
                 f"| Puntos {row.get('puntos',0):.0f}")

    # 3) Jugadores a vender
    vender = plantilla[
        (plantilla['media'].fillna(0) < 2)
        | (plantilla['alerta'].str.contains("Lesionado|Sancionado", case=False, na=False))
    ]
    if not vender.empty:
        slog("Jugadores a vender:")
        for _, row in vender.sort_values(by=["media"], ascending=True).iterrows():
            slog(f" - {row.get('nombre','?')} {row.get('apellido','?')} "
                 f"| Media {row.get('media',0):.2f} | Alerta: {row.get('alerta','')}")
    else:
        slog("Sin candidatos claros a vender por media/alertas.")

    # 4) Opciones de compra en mercado
    try:
        mercado = pd.read_sql(
            f"""SELECT m.*,
                v.tendencia
            FROM {schema}.v_mercado m
            LEFT JOIN (
                SELECT CONCAT(LEFT(nombre,1), '. ', apellido) AS nombre_abreviado,
                    MAX(tendencia) AS tendencia
                FROM {schema}.v_valores
                WHERE temporada = %s
                GROUP BY CONCAT(LEFT(nombre,1), '. ', apellido)
            ) v ON m.nombre = v.nombre_abreviado
            WHERE m.temporada = %s
            ORDER BY m.puntuacion_media DESC""",
            conn, params=(temporada, temporada)
        )
    except Exception as e:
        slog(f"ERROR leyendo mercado: {type(e).__name__}: {e}")
        mercado = pd.DataFrame()
    if not mercado.empty:
        for col in ("precio", "puntuacion_media"):
            if col in mercado.columns:
                mercado[col] = pd.to_numeric(mercado[col], errors="coerce")

        slog("Opciones de compra en mercado (<= saldo y media > 4):")
        candidatos_compra = mercado[
            (mercado["precio"].fillna(10**18) <= saldo) & (mercado["puntuacion_media"].fillna(0) > 4)
        ]
        if candidatos_compra.empty:
            slog("  *Ninguna opción que cumpla criterios ahora mismo.*")
        else:
            for _, row in candidatos_compra.iterrows():
                tend = row.get("tendencia", "")
                slog(f" - {row.get('nombre','?')} | Precio {row.get('precio',0):,.0f} | "
                     f"Media {row.get('puntuacion_media',0):.2f} | Tendencia: {tend}")

    # 5) Oportunidades de cláusula (rivales en negativo y positivos)
    rivales = pd.read_sql(
        f"SELECT usuario, saldo::numeric AS saldo FROM {schema}.saldos WHERE temporada = %s",
        conn, params=(temporada,)
    )

    if not rivales.empty:
        for _, rival in rivales.iterrows():
            jugadores_rival = pd.read_sql(
                f"""SELECT * FROM {schema}.v_datos_jugador
                    WHERE propietario = %s AND temporada = %s AND media > 6""",
                conn, params=(rival['usuario'], temporada)
            )
            if "clausula" in jugadores_rival.columns:
                jugadores_rival["clausula"] = pd.to_numeric(jugadores_rival["clausula"], errors="coerce")

            for _, jug in jugadores_rival.iterrows():
                clausula = float(jug.get("clausula")) if pd.notna(jug.get("clausula")) else None
                if clausula is not None:
                    saldo_post = saldo - clausula
                    tipo = "NEGATIVO" if rival['saldo'] < 0 else "POSITIVO"
                    slog(f"Posible cláusula a {rival['usuario']} ({tipo}): "
                         f"{jug.get('nombre','?')} {jug.get('apellido','?')} "
                         f"por {clausula:,.0f} (te quedarías con {saldo_post:,.0f} €)")
    else:
        slog("No hay rivales en la liga para analizar cláusulas.")

    slog("planificar_jornada: Fin")


if __name__ == "__main__":
    with conexion_db() as conn:
        planificar_jornada(conn, schema="chavalitos", usuario="vien2", temporada="25/26")
