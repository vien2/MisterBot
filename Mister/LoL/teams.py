from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, Integer, case, or_, text
import logging

logger = logging.getLogger("uvicorn.error")

from ..db import get_db, get_lol_db
from ..models_lol import Team, TeamGameStat, Match, Player, Tournament, GameStat
from ..dependencies import get_current_tournament_slug

router = APIRouter(
    prefix="/teams",
    tags=["teams"]
)

templates = Jinja2Templates(directory="templates")

def champion_image_filter(name):
    if not name: return ""
    mapping = {
        "Wukong": "MonkeyKing",
        "K'Sante": "KSante",
        "Renata Glasc": "Renata",
        "Nunu & Willump": "Nunu",
        "LeBlanc": "Leblanc",
        "Kog'Maw": "KogMaw",
        "Rek'Sai": "RekSai",
        "Dr. Mundo": "DrMundo",
        "Bel'Veth": "Belveth",
        "Vel'Koz": "Velkoz",
        "Kha'Zix": "Khazix",
        "Cho'Gath": "Chogath",
        "Kai'Sa": "Kaisa",
        "Yunara": "Yunara",
        "Ambessa": "Ambessa",
        "K": "KSante",
        "Rek": "RekSai",
        "RenataGlasc": "Renata",
        "ChoGath": "Chogath",
        "Nunu": "Nunu",
        "Wukong": "MonkeyKing",
        "LeBlanc": "Leblanc"
    }
    clean_name = mapping.get(name.strip(), name.strip().replace(" ", "").replace("'", "").replace(".", ""))

    if clean_name == "Yunara":
        return "https://gol.gg/_img/champions_icon/Yunara.png"
    if clean_name == "Zaahen":
        return "https://gol.gg/_img/champions_icon/Zaahen.png"
    if clean_name == "KSante":
        return "https://gol.gg/_img/champions_icon/KSante.png"

    return f"https://ddragon.leagueoflegends.com/cdn/15.1.1/img/champion/{clean_name}.png"

templates.env.filters["champion_image"] = champion_image_filter


@router.get("/")
def read_teams(request: Request, 
               current_tournament: str = Depends(get_current_tournament_slug),
               db: Session = Depends(get_lol_db)):
    
    # 1. Base Team Query (Filtered by Tournament)
    t_obj = None
    if current_tournament != "All":
        t_obj = db.query(Tournament).filter(Tournament.slug == current_tournament).first()
    
    query = db.query(Team)
    if t_obj:
        teams_with_matches = db.query(TeamGameStat.team_id)\
                               .join(Match, TeamGameStat.match_id == Match.id)\
                               .filter(Match.tournament_id == t_obj.id)
        
        query = query.filter(
            or_(
                Team.region == t_obj.region,
                Team.id.in_(teams_with_matches)
            )
        )
    
    teams = query.all()
    
    if not teams and t_obj and t_obj.region:
        teams = db.query(Team).filter(Team.region.ilike(f"%{t_obj.region}%")).all()
    
    t_id = t_obj.id if t_obj else None
    
    # 2. Aggregated Stats
    blue_wins = func.sum(case((TeamGameStat.side == 'Blue', func.cast(TeamGameStat.win, Integer)), else_=0))
    blue_games = func.sum(case((TeamGameStat.side == 'Blue', 1), else_=0))
    red_wins = func.sum(case((TeamGameStat.side == 'Red', func.cast(TeamGameStat.win, Integer)), else_=0))
    red_games = func.sum(case((TeamGameStat.side == 'Red', 1), else_=0))

    q = db.query(
        TeamGameStat.team_id,
        # FIX: Count match_id instead of id
        func.count(TeamGameStat.match_id).label("games_played"),
        func.sum(func.cast(TeamGameStat.win, Integer)).label("wins"),
        
        func.avg(case((TeamGameStat.game_duration > 0, TeamGameStat.game_duration), else_=None)).label("avg_duration"),
        func.min(case((TeamGameStat.game_duration > 0, TeamGameStat.game_duration), else_=None)).label("min_duration"),
        func.max(TeamGameStat.game_duration).label("max_duration"),
        
        func.sum(TeamGameStat.towers_destroyed).label("total_towers"),
        func.sum(TeamGameStat.dragons_killed).label("total_dragons"),
        func.sum(TeamGameStat.barons_killed).label("total_barons"),
        
        func.sum(func.cast(TeamGameStat.first_blood, Integer)).label("fb_count"),
        func.sum(func.cast(TeamGameStat.first_tower, Integer)).label("ft_count"),
        
        func.avg(TeamGameStat.first_blood_time).label("avg_fb_time"),
        func.avg(TeamGameStat.first_tower_time).label("avg_ft_time"),
        func.avg(TeamGameStat.first_dragon_time).label("avg_fd_time"),
        func.sum(TeamGameStat.elder_dragons_killed).label("total_elders"),
        
        blue_wins.label("blue_wins"),
        blue_games.label("blue_games"),
        red_wins.label("red_wins"),
        red_games.label("red_games"),
        
        func.sum(TeamGameStat.total_kills).label("team_total_kills"),
        func.sum(TeamGameStat.total_deaths).label("team_total_deaths"),
        func.sum(TeamGameStat.total_assists).label("team_total_assists")
    )
    
    if t_id:
        q = q.join(Match, TeamGameStat.match_id == Match.id).filter(Match.tournament_id == t_id)
    
    stats_query = q.group_by(TeamGameStat.team_id).all()
    
    # 3. KDA Aggregation
    kda_q = db.query(
        Player.team_id,
        func.sum(GameStat.kills).label("total_kills"),
        func.sum(GameStat.deaths).label("total_deaths"),
        func.sum(GameStat.assists).label("total_assists")
    ).join(Player, GameStat.player_id == Player.id)
    
    if t_id:
        kda_q = kda_q.join(Match, GameStat.match_id == Match.id).filter(Match.tournament_id == t_id)
        
    kda_query = kda_q.group_by(Player.team_id).all()
    kda_map = {row.team_id: row for row in kda_query}
    
    from sqlalchemy.orm import aliased
    
    # 4. Opponent Stats
    T1 = aliased(TeamGameStat)
    T2 = aliased(TeamGameStat)
    
    opp_q = db.query(
        T1.team_id,
        func.sum(T2.dragons_killed).label("dragons_against"),
        func.sum(T2.towers_destroyed).label("towers_against"),
        func.sum(T2.elder_dragons_killed).label("elders_against"),
        func.sum(T2.barons_killed).label("barons_against")
    ).join(T2, T1.match_id == T2.match_id).filter(T1.team_id != T2.team_id)
    
    if t_id:
        opp_q = opp_q.join(Match, T1.match_id == Match.id).filter(Match.tournament_id == t_id)
    
    opp_query = opp_q.group_by(T1.team_id).all()
    opp_map = {row.team_id: row for row in opp_query}

    stats_map = {}
    for row in stats_query:
        wins = int(row.wins) if row.wins else 0
        games = row.games_played
        
        losses = games - wins
        win_rate = round((wins / games * 100), 1) if games > 0 else 0
        avg_dur = int(row.avg_duration) if row.avg_duration else 0
        min_dur = int(row.min_duration) if row.min_duration else 0
        max_dur = int(row.max_duration) if row.max_duration else 0
        
        b_games = int(row.blue_games) if row.blue_games else 0
        b_wins = int(row.blue_wins) if row.blue_wins else 0
        b_wr = round((b_wins / b_games * 100), 1) if b_games > 0 else 0
        
        r_games = int(row.red_games) if row.red_games else 0
        r_wins = int(row.red_wins) if row.red_wins else 0
        r_wr = round((r_wins / r_games * 100), 1) if r_games > 0 else 0
        
        opp_stats = opp_map.get(row.team_id)
        
        total_dragons = row.total_dragons if row.total_dragons else 0
        dragons_against = opp_stats.dragons_against if opp_stats and opp_stats.dragons_against else 0
        avg_dragons_for = round(total_dragons / games, 2) if games > 0 else 0
        avg_dragons_against = round(dragons_against / games, 2) if games > 0 else 0
        avg_dragons_total = round((total_dragons + dragons_against) / games, 2) if games > 0 else 0

        total_towers = row.total_towers if row.total_towers else 0
        towers_against = opp_stats.towers_against if opp_stats and opp_stats.towers_against else 0
        avg_towers_for = round(total_towers / games, 2) if games > 0 else 0
        avg_towers_against = round(towers_against / games, 2) if games > 0 else 0
        avg_towers_total = round((total_towers + towers_against) / games, 2) if games > 0 else 0

        total_elders = row.total_elders if row.total_elders else 0
        elders_against = opp_stats.elders_against if opp_stats and opp_stats.elders_against else 0
        avg_elders_for = round(total_elders / games, 2) if games > 0 else 0

        total_barons = row.total_barons if row.total_barons else 0
        barons_against = opp_stats.barons_against if opp_stats and opp_stats.barons_against else 0
        avg_barons_for = round(total_barons / games, 2) if games > 0 else 0
        avg_barons_against = round(barons_against / games, 2) if games > 0 else 0
        
        avg_fb_time = int(row.avg_fb_time) if row.avg_fb_time else 0
        avg_ft_time = int(row.avg_ft_time) if row.avg_ft_time else 0
        avg_fd_time = int(row.avg_fd_time) if row.avg_fd_time else 0

        k, d, a = 0, 0, 0
        kda_row = kda_map.get(row.team_id)
        if kda_row and kda_row.total_kills:
            k = kda_row.total_kills
            d = kda_row.total_deaths
            a = kda_row.total_assists
            
        if k == 0 and d == 0:
             k = row.team_total_kills if row.team_total_kills else 0
             d = row.team_total_deaths if row.team_total_deaths else 0
             a = row.team_total_assists if row.team_total_assists else 0

        kda_ratio = round((k + a) / d, 2) if d > 0 else (k + a)

        stats_map[row.team_id] = {
            "games_played": games,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_kills": k,
            "total_deaths": d,
            "avg_duration": avg_dur,
            "min_duration": min_dur,
            "max_duration": max_dur,
            "avg_towers_for": avg_towers_for,
            "avg_towers_against": avg_towers_against,
            "avg_towers_total": avg_towers_total,
            "avg_dragons_for": avg_dragons_for,
            "avg_dragons_against": avg_dragons_against,
            "avg_dragons_total": avg_dragons_total,
            "avg_fd_time": avg_fd_time,
            "total_barons": total_barons,
            "barons_against": barons_against,
            "avg_barons_for": avg_barons_for,
            "avg_barons_against": avg_barons_against,
            "fb_count": row.fb_count if row.fb_count else 0,
            "ft_count": row.ft_count if row.ft_count else 0,
            "total_elders": total_elders,
            "elders_against": elders_against,
            "avg_elders_for": avg_elders_for,
            "avg_fb_time": avg_fb_time,
            "avg_ft_time": avg_ft_time,
            "avg_fd_time": avg_fd_time,
            "blue_wr": b_wr,
            "red_wr": r_wr,
            "kda": kda_ratio,
        }
        
    teams_with_stats = []
    for team in teams:
        s = stats_map.get(team.id, {
            "games_played": 0, "wins": 0, "losses": 0, "win_rate": 0, 
            "total_kills": 0, "total_deaths": 0,
            "avg_duration": 0, "min_duration": 0, "max_duration": 0,
            "avg_towers_for": 0, "avg_towers_against": 0, "avg_towers_total": 0,
            "avg_dragons_for": 0, "avg_dragons_against": 0, "avg_dragons_total": 0,
            "total_barons": 0, "barons_against": 0, "avg_barons_for": 0, "avg_barons_against": 0,
            "total_elders": 0, "elders_against": 0, "avg_elders_for": 0,
            "fb_count": 0, "ft_count": 0,
            "avg_fb_time": 0, "avg_ft_time": 0, "avg_fd_time": 0,
            "blue_wr": 0, "red_wr": 0,
            "kda": 0
        })
        t_obj = team
        t_obj.stats = s
        teams_with_stats.append(t_obj)
        
    teams_with_stats.sort(key=lambda x: (x.stats["wins"], x.stats["win_rate"]), reverse=True)
    
    return templates.TemplateResponse("teams.html", {
        "request": request, 
        "teams": teams_with_stats, 
        "title": "Teams"
    })

@router.get("/{team_id}")
def read_team_details(request: Request, team_id: int, 
                      current_tournament: str = Depends(get_current_tournament_slug),
                      db: Session = Depends(get_lol_db)):
    
    team = db.query(Team).filter(Team.id == team_id).first()
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    # FIX: Count match_id instead of id
    q = db.query(
        func.count(TeamGameStat.match_id).label("games_played"),
        func.sum(func.cast(TeamGameStat.win, Integer)).label("wins"),
        func.sum(TeamGameStat.total_kills).label("kills"),
        func.sum(TeamGameStat.towers_destroyed).label("towers")
    ).filter(TeamGameStat.team_id == team_id)

    t_obj = db.query(Tournament).filter(Tournament.slug == current_tournament).first()
    t_id = t_obj.id if t_obj else None

    if t_id:
        q = q.join(Match, TeamGameStat.match_id == Match.id).filter(Match.tournament_id == t_id)

    agg_stats = q.first()
    
    summary = {
        "games_played": agg_stats.games_played if agg_stats and agg_stats.games_played else 0,
        "wins": int(agg_stats.wins) if agg_stats and agg_stats.wins else 0,
        "kills": agg_stats.kills if agg_stats and agg_stats.kills else 0,
        "towers": agg_stats.towers if agg_stats and agg_stats.towers else 0
    }
    summary["losses"] = summary["games_played"] - summary["wins"]
    summary["win_rate"] = round((summary["wins"] / summary["games_played"] * 100), 1) if summary["games_played"] > 0 else 0

    series_date_sub = db.query(
        func.coalesce(Match.series_id, Match.id).label('s_id'),
        func.max(Match.match_date).label('series_max_date')
    ).group_by(text('s_id')).subquery()

    history_q = db.query(TeamGameStat, Match)\
                  .join(Match, TeamGameStat.match_id == Match.id)\
                  .join(series_date_sub, func.coalesce(Match.series_id, Match.id) == series_date_sub.c.s_id)\
                  .filter(TeamGameStat.team_id == team_id)
    
    if current_tournament != "All":
        history_q = history_q.filter(Match.tournament_id == current_tournament)
        
    history_query = history_q.order_by(
        desc(series_date_sub.c.series_max_date), 
        series_date_sub.c.s_id, 
        Match.game_number
    ).all()
    
    match_history = []
    for tgs, match in history_query:
        if match.blue_team_id == team.id:
            opponent_id = match.red_team_id
        else:
            opponent_id = match.blue_team_id
            
        opponent_team = db.query(Team).filter(Team.id == opponent_id).first()
        
        opponent_name = opponent_team.name if opponent_team else "Unknown"
        opponent_code = opponent_team.code if opponent_team else "UNK"
        opponent_image = opponent_team.image_url if opponent_team else ""
        
        match_history.append({
            "match_id": match.id,
            # FIX: Removed invalid .id reference. Used fake unique ID for template if needed
            "game_id": f"{tgs.match_id}_{team_id}", 
            "game_number": match.game_number,
            "date": match.match_date,
            "opponent_name": opponent_name,
            "opponent_code": opponent_code,
            "opponent_image": opponent_image,
            "win": tgs.win,
            "stats": tgs, 
            "side": tgs.side
        })
        
    return templates.TemplateResponse("team_details.html", {
        "request": request, 
        "team": team, 
        "summary": summary,
        "match_history": match_history,
        "title": team.name
    })
