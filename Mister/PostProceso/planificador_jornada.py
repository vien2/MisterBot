import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import warnings
import pandas as pd
from utils import log, conexion_db

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# ---- helper: escribe en log y también en consola ----
def slog(msg: str):
    try:
        log(msg)
    finally:
        print(msg)

def planificar_jornada(conn=None, schema="chavalitos", usuario="vien2", temporada="25/26"):
    slog("planificar_jornada: Inicio")

    if conn is None:
        with conexion_db() as conn_interna:
            return planificar_jornada(conn_interna, schema, usuario, temporada)

    # 1) Saldo
    saldo_df = pd.read_sql(
        f"SELECT saldo FROM {schema}.saldos WHERE usuario = %s AND temporada = %s",
        conn, params=(usuario, temporada)
    )
    if saldo_df.empty:
        slog(f"No se encontró saldo para {usuario}")
        return
    saldo = float(saldo_df.iloc[0]['saldo'])
    slog(f"Saldo disponible: {saldo:,.0f} €")

    # 2) Plantilla propia
    plantilla = pd.read_sql(
        f"SELECT * FROM {schema}.v_datos_jugador WHERE propietario = %s AND temporada = %s",
        conn, params=(usuario, temporada)
    )
    for col in ("media", "puntos", "clausula", "valor"):
        if col in plantilla.columns:
            plantilla[col] = pd.to_numeric(plantilla[col], errors="coerce")
    plantilla["alerta"] = plantilla["alerta"].fillna("")

    # 3) Jugadores a vender (explicado)
    vender, proteger = [], []

    for _, row in plantilla.iterrows():
        motivos, protegido = [], False
        nombre, apellido = row["nombre"], row["apellido"]
        media = float(row.get("media", 0) or 0)
        alerta = str(row.get("alerta", ""))
        id_jugador = row.get("id_jugador")

        # --- Criterios de protección ---
        if "Top" in alerta and "robado" in alerta:
            protegido = True
            motivos.append("alerta de top robado")

        # histórico de puntos
        try:
            hist = pd.read_sql(
                f"SELECT media_puntuacion FROM {schema}.v_puntos "
                f"WHERE id_jugador = %s ORDER BY temporada_mister DESC LIMIT 3",
                conn, params=(id_jugador,)
            )
            if not hist.empty and hist["media_puntuacion"].max() >= 6:
                protegido = True
                motivos.append(f"histórico con media {hist['media_puntuacion'].max():.1f}")
        except Exception:
            pass

        # evolución de valor
        try:
            val = pd.read_sql(
                f"""
                SELECT v.periodo, v.cambio_valor
                FROM {schema}.v_valores v
                JOIN (
                    SELECT id_jugador, periodo, MAX(f_carga) AS f_carga
                    FROM {schema}.v_valores
                    WHERE temporada = %s AND periodo IN ('Un día','Una semana')
                    GROUP BY id_jugador, periodo
                ) ult
                ON v.id_jugador = ult.id_jugador
                AND v.periodo = ult.periodo
                AND v.f_carga = ult.f_carga
                WHERE v.id_jugador = %s AND v.temporada = %s
                """,
                conn, params=(temporada, row['id_jugador'], temporada)
            )
            if not val.empty and (val.iloc[0]["subida"] or 0) > 100000:
                protegido = True
                motivos.append(f"subida de valor {val.iloc[0]['subida']:,}")
        except Exception:
            pass

        # --- Decisión final ---
        if protegido:
            proteger.append((nombre, apellido, media, alerta, motivos))
        else:
            if media < 2:
                motivos.append("media baja")
            if any(x in alerta for x in ["Lesion", "Duda", "Sancion"]):
                motivos.append("alerta médica/sanción")
            if not motivos:
                motivos.append("sin factores positivos")
            vender.append((nombre, apellido, media, alerta, motivos))

    if vender:
        slog("Jugadores a vender:")
        for n, a, m, al, motivos in vender:
            slog(f" - {n} {a} | Media {m:.2f} | Alerta: {al} → Motivo: {', '.join(motivos)}")
    else:
        slog("Sin jugadores claros a vender.")

    if proteger:
        slog("Jugadores protegidos (no vender):")
        for n, a, m, al, motivos in proteger:
            slog(f" - {n} {a} | Media {m:.2f} | Alerta: {al} → Protegido por: {', '.join(motivos)}")

    # 4) XI recomendado (1-3-3-4), excluyendo los a vender
    ids_vender = [row[0] for row in vender]
    alineables = plantilla[~plantilla["id_jugador"].isin(ids_vender)]
    posiciones = {"PT": 1, "DF": 3, "MC": 3, "DL": 4}
    once, faltantes = [], {}
    for pos, n in posiciones.items():
        candidatos = alineables[alineables['posicion'] == pos].sort_values(
            by=["media", "puntos"], ascending=[False, False]
        ).head(n)
        once.append(candidatos)
        if len(candidatos) < n:
            faltantes[pos] = n - len(candidatos)
    xi_recomendado = pd.concat(once) if once else pd.DataFrame()

    slog("XI recomendado (1-3-3-4) [excluyendo jugadores a vender]:")
    if xi_recomendado.empty:
        slog("  *No hay suficientes jugadores para alinear*")
    else:
        for _, row in xi_recomendado.iterrows():
            slog(f" - {row['nombre']} {row['apellido']} ({row['posicion']}) "
                 f"| Media {row['media']:.2f} | Puntos {row['puntos']:.0f}")

    if faltantes:
        for pos, n in faltantes.items():
            slog(f"⚠️ Te faltan {n} jugador(es) en la posición {pos} para completar el XI")

    # 5) Opciones de compra en mercado (incluye tendencia)
    mercado = pd.read_sql(f"""
        SELECT m.*, v.tendencia
        FROM {schema}.v_mercado m
        LEFT JOIN (
            SELECT nombre, MAX(tendencia) AS tendencia
            FROM {schema}.v_valores
            WHERE temporada = %s
            GROUP BY nombre
        ) v ON m.nombre = v.nombre
        WHERE m.temporada = %s
        ORDER BY m.puntuacion_media DESC
        LIMIT 50
    """, conn, params=(temporada, temporada))

    for col in ("precio", "puntuacion_media"):
        if col in mercado.columns:
            mercado[col] = pd.to_numeric(mercado[col], errors="coerce")

    candidatos_compra = mercado[
        (mercado["precio"].fillna(10**18) <= saldo) &
        (mercado["puntuacion_media"].fillna(0) > 4)
    ]
    if not candidatos_compra.empty:
        slog("Opciones de compra en mercado (<= saldo y media > 4):")
        for _, row in candidatos_compra.iterrows():
            slog(f" - {row['nombre']} | Precio {row['precio']:,} | "
                 f"Media {row['puntuacion_media']:.2f} | Tendencia: {row['tendencia']}")
    else:
        slog("Ningún jugador en mercado cumple criterios de compra.")

    # 6) Sugerencias de fichajes para posiciones faltantes
    if faltantes:
        slog("Sugerencias de fichajes para cubrir posiciones faltantes:")
        for pos, n in faltantes.items():
            candidatos_pos = mercado[
                (mercado["precio"] <= saldo) &
                (mercado["puntuacion_media"] > 4)
            ]
            if "posicion" in candidatos_pos.columns:
                candidatos_pos = candidatos_pos[candidatos_pos["posicion"] == pos]
            sugerencias = candidatos_pos.head(n)
            for _, row in sugerencias.iterrows():
                slog(f" - {row['nombre']} ({pos}) | Precio {row['precio']:,} | "
                     f"Media {row['puntuacion_media']:.2f} | Tendencia: {row['tendencia']}")

    slog("planificar_jornada: Fin")

if __name__ == "__main__":
    with conexion_db() as conn:
        planificar_jornada(conn, schema="chavalitos", usuario="vien2", temporada="25/26")
