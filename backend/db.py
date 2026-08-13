"""SQLite orqali tahlil tarixini saqlash (SQLAlchemy)."""
import datetime
import json

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text
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

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
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
            "detail": json.loads(self.raw_result),
        }


Base.metadata.create_all(engine)


def save_analysis(image_path: str, consensus: dict) -> Analysis:
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
            raw_result=json.dumps(consensus),
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record
    finally:
        session.close()
