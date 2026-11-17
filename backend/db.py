"""SQLite orqali tahlil tarixini saqlash (SQLAlchemy)."""
import datetime
import json
import time
import uuid

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, func, update, or_, and_
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DB_PATH

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    image_path = Column(String)

    final_signal = Column(String)
    final_confidence = Column(Float)
    agreement = Column(Boolean)

    custom_signal = Column(String)
    custom_confidence = Column(Float)
    vision_signal = Column(String)
    vision_confidence = Column(Float)

    entry_price = Column(Float, nullable=True)
    tp_price = Column(Float, nullable=True)
    sl_price = Column(Float, nullable=True)

    raw_result = Column(Text)  # to'liq consensus natijasi JSON sifatida

    # evaluator.py to'ldiradi. outcome: TP / SL / OPEN / EXPIRED / SKIP / NO_DATA / None (hali tekshirilmagan)
    # outcome_kind: "signal" (haqiqiy Buy/Sell) yoki "hold-gipotetik" (Hold, Gemini yo'nalishi bo'yicha)
    outcome = Column(String, nullable=True)
    outcome_kind = Column(String, nullable=True)
    outcome_checked_at = Column(DateTime, nullable=True)
    outcome_version = Column(Integer, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() + "Z",
            "image_path": self.image_path,
            "final_signal": self.final_signal,
            "final_confidence": self.final_confidence,
            "agreement": self.agreement,
            "custom_signal": self.custom_signal,
            "custom_confidence": self.custom_confidence,
            "vision_signal": self.vision_signal,
            "vision_confidence": self.vision_confidence,
            "entry_price": self.entry_price,
            "tp_price": self.tp_price,
            "sl_price": self.sl_price,
            "outcome": self.outcome,
            "outcome_kind": self.outcome_kind,
            "outcome_version": self.outcome_version,
            "detail": json.loads(self.raw_result),
        }


def _add_missing_columns():
    """create_all() mavjud jadvalga yangi ustun qo'shmaydi -- eski bazani buzmasdan qo'shamiz."""
    with engine.begin() as conn:
        existing = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(analyses)")}
        for col in ("outcome", "outcome_kind"):
            if col not in existing:
                conn.exec_driver_sql(f"ALTER TABLE analyses ADD COLUMN {col} VARCHAR")
        if "outcome_checked_at" not in existing:
            conn.exec_driver_sql("ALTER TABLE analyses ADD COLUMN outcome_checked_at DATETIME")
        if "outcome_version" not in existing:
            conn.exec_driver_sql("ALTER TABLE analyses ADD COLUMN outcome_version INTEGER")


Base.metadata.create_all(engine)
_add_missing_columns()


def save_analysis(image_path: str, consensus: dict) -> Analysis:
    # Refuse invalid provider/indicator values before opening a transaction.
    raw_result = json.dumps(consensus, allow_nan=False)
    session = SessionLocal()
    try:
        record = Analysis(
            image_path=image_path,
            final_signal=consensus["final_signal"],
            final_confidence=consensus["final_confidence"],
            agreement=consensus["agreement"],
            custom_signal=consensus["custom_model"]["signal"],
            custom_confidence=consensus["custom_model"]["confidence"],
            vision_signal=consensus["vision_ai"]["signal"],
            vision_confidence=consensus["vision_ai"]["confidence"],
            entry_price=consensus.get("entry_price"),
            tp_price=consensus.get("tp_price"),
            sl_price=consensus.get("sl_price"),
            raw_result=raw_result,
        )
        if consensus.get("published_at"):
            published = datetime.datetime.fromisoformat(consensus["published_at"].replace("Z", "+00:00"))
            record.timestamp = published.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        session.add(record)
        session.commit()
        session.refresh(record)
        return record
    finally:
        session.close()


def patch_analysis_result(session, record_id: int, fields: dict, qwen_request_id: str | None = None):
    """Patch JSON paths atomically so model evaluation cannot erase a new Qwen result."""
    expression = Analysis.raw_result
    for path, value in fields.items():
        expression = func.json_set(expression, path, func.json(json.dumps(value, allow_nan=False)))
    statement = update(Analysis).where(Analysis.id == record_id)
    if qwen_request_id is not None:
        statement = statement.where(func.json_extract(Analysis.raw_result, "$.qwen_vision.request_id") == qwen_request_id)
    result = session.execute(statement.values(raw_result=expression), execution_options={"synchronize_session": False})
    return bool(result.rowcount)


def claim_qwen_analysis(record_id: int, model: str, lease_seconds: float = 150):
    """Only one request can start the optional analysis, including across workers."""
    session = SessionLocal()
    try:
        now = time.time()
        running = json.dumps({"source": "qwen", "model": model, "status": "running",
                              "request_id": uuid.uuid4().hex, "lease_expires_at": now + lease_seconds})
        result = session.execute(update(Analysis).where(
            Analysis.id == record_id,
            or_(func.json_extract(Analysis.raw_result, "$.qwen_vision.status").in_(("not_requested", "error")),
                and_(func.json_extract(Analysis.raw_result, "$.qwen_vision.status") == "running",
                     func.coalesce(func.json_extract(Analysis.raw_result, "$.qwen_vision.lease_expires_at"), 0) < now)),
        ).values(raw_result=func.json_set(Analysis.raw_result, "$.qwen_vision", func.json(running))),
            execution_options={"synchronize_session": False})
        session.commit()
        row = session.get(Analysis, record_id)
        return (row.to_dict() if row else None), bool(result.rowcount)
    finally:
        session.close()
