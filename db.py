from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field, create_engine, Session

# SQLite file in project root
DATABASE_URL = "sqlite:///vdr.db"
engine = create_engine(DATABASE_URL, echo=False)

class Selection(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    gender: str
    top_url: Optional[str] = None
    bottom_url: Optional[str] = None
    liked: bool = Field(default=False)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session() -> Session:
    return Session(engine)
