from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from utils import leer_config_db, log
import urllib.parse

# Construct Database URL from utils logic
try:
    config = leer_config_db()
    # SQL Alchemy URL format: postgresql://user:password@host:port/database
    
    # Handle special characters in password
    password = urllib.parse.quote_plus(config['password'])
    
    SQLALCHEMY_DATABASE_URL = f"postgresql://{config['user']}:{password}@{config['host']}:{config['port']}/{config['database']}"
    
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

except Exception as e:
    log(f"LoL Database Init Error: {e}")
    # Fallback to prevent import crash, though functionality will break
    SessionLocal = None
    Base = declarative_base()
    engine = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
