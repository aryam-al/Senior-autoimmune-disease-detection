"""SQLite database: users and analysis history."""

import json
import os
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

import config

engine = create_engine(
    config.DATABASE_URL,
    connect_args={"check_same_thread": False} if config.DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=utcnow)


class ReportHistory(Base):
    __tablename__ = "report_history"

    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, index=True)
    # XGBoost result (original columns; old rows hold the earlier Random Forest result)
    disease = Column(String)
    confidence = Column(Float)
    date = Column(String)
    created_at = Column(DateTime, default=utcnow)
    # Added in v2
    qwen_disease = Column(String, nullable=True)
    qwen_confidence = Column(Float, nullable=True)
    source_filename = Column(String, nullable=True)
    input_json = Column(Text, nullable=True)
    result_json = Column(Text, nullable=True)


def _add_missing_columns():
    """create_all() does not alter existing tables, so add new columns by hand."""
    insp = inspect(engine)
    if "report_history" not in insp.get_table_names():
        return
    existing = {c["name"] for c in insp.get_columns("report_history")}
    new_cols = {
        "qwen_disease": "VARCHAR",
        "qwen_confidence": "FLOAT",
        "source_filename": "VARCHAR",
        "input_json": "TEXT",
        "result_json": "TEXT",
    }
    with engine.begin() as conn:
        for name, sql_type in new_cols.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE report_history ADD COLUMN {name} {sql_type}"))


def _import_legacy_users():
    """Copy accounts from the old users.json into the users table (once).
    The old file is left in place; it can be deleted after checking the import."""
    if not os.path.exists(config.LEGACY_USERS_FILE):
        return
    with open(config.LEGACY_USERS_FILE, encoding="utf-8") as f:
        legacy = json.load(f)
    db = SessionLocal()
    try:
        for email, u in legacy.items():
            email = email.strip().lower()
            if not db.query(User).filter(User.email == email).first():
                db.add(User(email=email, name=u.get("name", email), password_hash=u["password"]))
        db.commit()
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()
    _import_legacy_users()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
