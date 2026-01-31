from sqlalchemy import Column, Integer, String, ForeignKey, Date, DateTime, Boolean, Float
from sqlalchemy.orm import relationship
from .database import Base

# Schema configuration for all tables
SCHEMA_ARGS = {"schema": "LoL_Stats"}

class Tournament(Base):
    __tablename__ = "tournaments"
    __table_args__ = SCHEMA_ARGS

    id = Column(String, primary_key=True, index=True) # e.g., "LEC 2024 Winter"
    name = Column(String)
    slug = Column(String)
    start_date = Column(Date)
    end_date = Column(Date)
    season = Column(String) # e.g. S15, S16
    region = Column(String)

    matches = relationship("Match", back_populates="tournament")

class Team(Base):
    __tablename__ = "teams"
    __table_args__ = SCHEMA_ARGS

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    code = Column(String) # e.g. G2, FNC
    image_url = Column(String)
    region = Column(String)

    players = relationship("Player", back_populates="team")

class Player(Base):
    __tablename__ = "players"
    __table_args__ = SCHEMA_ARGS

    id = Column(Integer, primary_key=True, index=True)
    handle = Column(String, index=True) # In-game name
    real_name = Column(String)
    role = Column(String) # Top, Jungle...
    team_id = Column(Integer, ForeignKey("LoL_Stats.teams.id")) # Explicit schema in ForeignKey
    image_url = Column(String)

    team = relationship("Team", back_populates="players")
    stats = relationship("GameStat", back_populates="player")

class Match(Base):
    __tablename__ = "matches"
    __table_args__ = SCHEMA_ARGS

    id = Column(String, primary_key=True, index=True) # gol.gg Game ID
    tournament_id = Column(String, ForeignKey("LoL_Stats.tournaments.id"))
    match_date = Column(Date)
    patch = Column(String)
    week = Column(String)
    
    best_of = Column(Integer, default=1)
    series_id = Column(String, index=True, nullable=True)
    game_number = Column(Integer, default=1)
    
    blue_team_id = Column(Integer, ForeignKey("LoL_Stats.teams.id"))
    red_team_id = Column(Integer, ForeignKey("LoL_Stats.teams.id"))
    winner_id = Column(Integer, ForeignKey("LoL_Stats.teams.id"))
    
    score = Column(String)
    
    tournament = relationship("Tournament", back_populates="matches")
    blue_team = relationship("Team", foreign_keys=[blue_team_id])
    red_team = relationship("Team", foreign_keys=[red_team_id])
    winner_team = relationship("Team", foreign_keys=[winner_id])

    stats = relationship("GameStat", back_populates="match")
    team_stats = relationship("TeamGameStat", back_populates="match_")

class GameStat(Base):
    __tablename__ = "game_stats"
    __table_args__ = SCHEMA_ARGS

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(String, ForeignKey("LoL_Stats.matches.id"))
    player_id = Column(Integer, ForeignKey("LoL_Stats.players.id"))
    champion_name = Column(String)
    
    # KDA
    kills = Column(Integer, default=0)
    deaths = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    
    # Eco
    total_gold = Column(Integer, default=0)
    cs = Column(Integer, default=0)
    gold_per_min = Column(Float, default=0.0)
    cs_per_min = Column(Float, default=0.0)
    
    match = relationship("Match", back_populates="stats")
    player = relationship("Player", back_populates="stats")
    
    # Damage
    damage_dealt = Column(Integer, default=0)
    damage_taken = Column(Integer, default=0)
    damage_share = Column(Float, default=0.0)
    
    # Vision
    vision_score = Column(Integer, default=0)
    wards_placed = Column(Integer, default=0)
    wards_killed = Column(Integer, default=0)
    control_wards_purchased = Column(Integer, default=0)
    
    # Context
    side = Column(String)
    win = Column(Boolean)

    # Multi-kills
    double_kills = Column(Integer, default=0)
    triple_kills = Column(Integer, default=0)
    quadra_kills = Column(Integer, default=0)
    penta_kills = Column(Integer, default=0)

    # Detailed Stats
    level = Column(Integer)
    vision_score_per_minute = Column(Float)
    wards_per_minute = Column(Float)
    vision_wards_per_minute = Column(Float)
    wards_cleared_per_minute = Column(Float)
    vision_share = Column(Float)
    detector_wards_placed = Column(Integer)
    
    physical_damage_dealt_to_champions = Column(Integer)
    magic_damage_dealt_to_champions = Column(Integer)
    true_damage_dealt_to_champions = Column(Integer)
    damage_per_minute = Column(Integer)
    
    damage_dealt_to_turrets = Column(Integer)
    damage_dealt_to_buildings = Column(Integer)
    
    ka_per_minute = Column(Float)
    kill_participation = Column(Float)
    solo_kills = Column(Integer)
    time_ccing_others = Column(Integer)
    total_time_cc_dealt = Column(Integer)
    
    total_heal = Column(Integer)
    total_heals_on_teammates = Column(Integer)
    damage_self_mitigated = Column(Integer)
    total_damage_shielded_on_teammates = Column(Integer)
    total_time_spent_dead = Column(Integer)
    
    cs_in_team_jungle = Column(Integer)
    cs_in_enemy_jungle = Column(Integer)
    
    gold_share = Column(Float)
    gold_diff_15 = Column(Integer)
    cs_diff_15 = Column(Integer)
    xp_diff_15 = Column(Integer)
    level_diff_15 = Column(Integer)
    
    consumables_purchased = Column(Integer)
    items_purchased = Column(Integer)
    shutdown_bounty_collected = Column(Integer)
    shutdown_bounty_lost = Column(Integer)
    objectives_stolen = Column(Integer)

class TeamGameStat(Base):
    __tablename__ = "team_game_stats"
    __table_args__ = SCHEMA_ARGS

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(String, ForeignKey("LoL_Stats.matches.id")) 
    team_id = Column(Integer, ForeignKey("LoL_Stats.teams.id"))
    side = Column(String)
    win = Column(Boolean)
    
    first_blood = Column(Boolean)
    first_tower = Column(Boolean)
    first_dragon = Column(Boolean)
    first_baron = Column(Boolean)
    
    towers_destroyed = Column(Integer)
    dragons_killed = Column(Integer)
    barons_killed = Column(Integer)
    void_grubs = Column(Integer)
    rift_heralds = Column(Integer)
    
    game_duration = Column(Integer)
    patch = Column(String)

    elder_dragons_killed = Column(Integer, default=0)
    
    first_blood_time = Column(Integer)
    first_tower_time = Column(Integer)
    first_dragon_time = Column(Integer)
    first_baron_time = Column(Integer)
    
    dragon_events = Column(String) 
    
    bans = Column(String)
    picks = Column(String)
    
    plates_total = Column(Integer)
    plates_top = Column(Integer)
    plates_mid = Column(Integer)
    plates_bot = Column(Integer)
    
    wards_placed = Column(Integer)
    wards_destroyed = Column(Integer)
    
    jungle_share_15 = Column(Float)
    jungle_share_end = Column(Float)

    gold_at_15 = Column(Integer)
    total_gold = Column(Integer)
    total_kills = Column(Integer)
    total_deaths = Column(Integer)
    total_assists = Column(Integer)

    match_ = relationship("Match", back_populates="team_stats")
    team = relationship("Team")

class ScrapeLog(Base):
    __tablename__ = "scrape_logs"
    __table_args__ = SCHEMA_ARGS

    id = Column(Integer, primary_key=True, index=True)
    entity_type = Column(String, index=True) # "MATCH", "TOURNAMENT", "TEAM"
    entity_id = Column(String, index=True)
    status = Column(String)
    last_scraped = Column(DateTime)
    details = Column(String, nullable=True)
