import pandas as pd
import matplotlib.pyplot as plt
from utils import conexion_db  # Asumiendo que ya tienes esto definido

# Conexión usando tu context manager
with conexion_db() as conn:
    # 1. Usuarios que más cláusulas HAN HECHO (robado)
    query_realizadas = """
        SELECT usuario_destino AS usuario, COUNT(*) AS clausulas_realizadas
        FROM dbo.v_transferencias
        WHERE tipo_operacion = 'Cláusula'
          AND fecha_transferencia BETWEEN '2024-08-15' AND '2025-06-30'
        GROUP BY usuario_destino
        ORDER BY clausulas_realizadas DESC
        LIMIT 5;
    """
    df_realizadas = pd.read_sql(query_realizadas, conn)

    # 2. Usuarios que más cláusulas HAN SUFRIDO
    query_sufridas = """
        SELECT usuario_origen AS usuario, COUNT(*) AS clausulas_sufridas
        FROM dbo.v_transferencias
        WHERE tipo_operacion = 'Cláusula'
          AND fecha_transferencia BETWEEN '2024-08-15' AND '2025-06-30'
        GROUP BY usuario_origen
        ORDER BY clausulas_sufridas DESC
        LIMIT 5;
    """
    df_sufridas = pd.read_sql(query_sufridas, conn)

# --- GRAFICAR ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico de cláusulas realizadas
df_realizadas.plot(kind='bar', x='usuario', y='clausulas_realizadas', legend=False, ax=axes[0], color='green')
axes[0].set_title('Top usuarios que HAN HECHO cláusulas')
axes[0].set_xlabel('Usuario')
axes[0].set_ylabel('Nº Cláusulas')

# Gráfico de cláusulas sufridas
df_sufridas.plot(kind='bar', x='usuario', y='clausulas_sufridas', legend=False, ax=axes[1], color='red')
axes[1].set_title('Top usuarios que HAN SUFRIDO cláusulas')
axes[1].set_xlabel('Usuario')
axes[1].set_ylabel('Nº Cláusulas')

plt.tight_layout()
plt.show()
