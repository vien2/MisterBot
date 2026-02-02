from utils import conexion_db

def debug_ghosts():
    slug = "LEC 2026 Versus Season"
    print(f"--- Debugging Teams for {slug} ---")
    
    with conexion_db() as conn:
        with conn.cursor() as cur:
            # 1. Get Tournament ID
            cur.execute("SELECT id, region, slug FROM lol_stats.tournaments WHERE slug = %s", (slug,))
            res = cur.fetchone()
            if not res:
                print(f"ERROR: Tournament '{slug}' not found!")
                print("Searching for similar tournaments...")
                cur.execute("SELECT slug FROM lol_stats.tournaments WHERE slug ILIKE '%LEC%' OR slug ILIKE '%Versus%'")
                found = cur.fetchall()
                print("Did you mean one of these?")
                for f in found:
                    print(f" - {f[0]}")
                return
            t_id, region, real_slug = res
            print(f"Tournament ID: {t_id} | Region: {region} | Slug: {real_slug}")

            # 2. Get Teams linked via Matches -> TeamGameStats
            query = """
                SELECT DISTINCT t.id, t.name, t.code, t.region, m.id as match_id
                FROM lol_stats.team_game_stats tgs
                JOIN lol_stats.matches m ON tgs.match_id = m.id
                LEFT JOIN lol_stats.teams t ON tgs.team_id = t.id
                WHERE m.tournament_id = %s
                ORDER BY t.name
            """
            cur.execute(query, (t_id,))
            rows = cur.fetchall()

            print(f"\nFound {len(rows)} linked team records:")
            seen_teams = set()
            for r in rows:
                tid, name, code, t_reg, mid = r
                
                # Highlight suspicious ones
                status = "OK"
                if tid is None or name is None:
                    status = "!!! GHOST/MISSING !!!"
                elif region and t_reg != region:
                    status = f"WARN: Region Mismatch ({t_reg})"

                if tid not in seen_teams:
                    print(f"[{status}] ID: {tid} | Name: {name} | Code: {code} | Match source: {mid}")
                    seen_teams.add(tid)
                
                if status.startswith("!!!"):
                    print(f"    -> Suspicious Record Details: TeamGameStat linked to Match {mid}")

if __name__ == "__main__":
    debug_ghosts()
