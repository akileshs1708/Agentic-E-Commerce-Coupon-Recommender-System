from shared.database import init_db
from shared.models import Base
from sqlalchemy import create_engine

print("Running DB initialization...")

init_db()

print("DB initialized successfully")