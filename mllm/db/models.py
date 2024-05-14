import time

from sqlalchemy import Boolean, Column, String, Float, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class PromptRecord(Base):
    __tablename__ = "prompts"

    id = Column(String, primary_key=True)
    namespace = Column(String, default="default")
    thread = Column(String)
    response = Column(Text)
    response_schema = Column(Text, nullable=True)
    metadata_ = Column(Text, default=dict)
    approved = Column(Boolean, default=False)
    flagged = Column(Boolean, default=False)
    agent_id = Column(String, nullable=True)
    model = Column(String, nullable=True)
    created = Column(Float, default=time.time)
    owner_id = Column(String, nullable=True)
