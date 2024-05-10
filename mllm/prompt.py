import uuid
import time
import logging
import json
from typing import Dict, Any, List, Optional, Type

from pydantic import create_model, Field, ValidationError, BaseModel
from threadmem import RoleThread, RoleMessage
from threadmem.server.models import V1RoleMessage, V1RoleThread

from .db.models import PromptRecord
from .db.conn import WithDB
from .models import V1Prompt

logger = logging.getLogger(__name__)


class Prompt(WithDB):
    """An LLM prompt"""

    def __init__(
        self,
        thread: RoleThread,
        response: RoleMessage,
        response_schema: Optional[Type[BaseModel]] = None,
        namespace: str = "default",
        metadata: Dict[str, Any] = {},
        approved: bool = False,
        flagged: bool = False,
        owner_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self._id = str(uuid.uuid4())
        self._namespace = namespace
        self._thread = thread
        self._response = response
        self._response_schema = response_schema
        self._metadata = metadata
        self._created = time.time()
        self._approved = approved
        self._flagged = flagged
        self._owner_id = owner_id
        self._agent_id = agent_id
        self._model = model

        self.save()

    @property
    def id(self) -> str:
        return self._id

    @property
    def namespace(self) -> str:
        return self._namespace

    @namespace.setter
    def namespace(self, value: str):
        self._namespace = value

    @property
    def thread(self) -> RoleThread:
        return self._thread

    @thread.setter
    def thread(self, value: RoleThread):
        self._thread = value

    @property
    def response(self) -> RoleMessage:
        return self._response

    @response.setter
    def response(self, value: RoleMessage):
        self._response = value

    @property
    def response_schema(self) -> Optional[Type[BaseModel]]:
        return self._response_schema

    @response_schema.setter
    def response_schema(self, value: Optional[Type[BaseModel]]):
        self._response_schema = value

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        self._metadata = value

    @property
    def created(self) -> float:
        return self._created

    @created.setter
    def created(self, value: float):
        self._created = value

    @property
    def approved(self) -> bool:
        return self._approved

    @approved.setter
    def approved(self, value: bool):
        self._approved = value

    @property
    def flagged(self) -> bool:
        return self._flagged

    @flagged.setter
    def flagged(self, value: bool):
        self._flagged = value

    @property
    def owner_id(self) -> Optional[str]:
        return self._owner_id

    @owner_id.setter
    def owner_id(self, value: str):
        self._owner_id = value

    @property
    def agent_id(self) -> Optional[str]:
        return self._agent_id

    @agent_id.setter
    def agent_id(self, value: str):
        self._agent_id = value

    @property
    def model(self) -> Optional[str]:
        return self._model

    @model.setter
    def model(self, value: str):
        self._model = value

    def to_record(self) -> PromptRecord:
        # Serialize the response using RoleMessageModel's json() method
        if not self.metadata:
            self.metadata = {}

        return PromptRecord(
            id=self._id,
            namespace=self._namespace,
            thread=self._thread.to_v1().model_dump_json(),
            response=self._response.to_v1().model_dump_json(),
            response_schema=(
                json.dumps(self._response_schema.model_json_schema())
                if self._response_schema
                else None
            ),
            metadata_=json.dumps(self._metadata),
            created=self._created,
            approved=self._approved,
            flagged=self._flagged,
            agent_id=self._agent_id,
            model=self._model,
        )

    @classmethod
    def from_record(cls, record: PromptRecord) -> "Prompt":
        # Deserialize thread_id into a RoleThreadModel using a suitable method or lookup
        thread_model = V1RoleThread.model_validate_json(str(record.thread))
        thread = RoleThread.from_v1(thread_model)

        # Deserialize the response
        response_model = V1RoleMessage.model_validate_json(str(record.response))
        response = RoleMessage.from_v1(response_model)

        # Dynamically create Pydantic model from JSON schema if available
        response_schema_model = None
        if record.response_schema:  # type: ignore
            schema = json.loads(record.response_schema)  # type: ignore
            response_schema_model = cls._schema_to_model(schema)

        # Load metadata
        metadata = json.loads(record.metadata_) if record.metadata_ else {}  # type: ignore

        obj = cls.__new__(cls)
        obj._id = record.id
        obj._namespace = record.namespace
        obj._thread = thread
        obj._response = response
        obj._response_schema = (
            response_schema_model  # Assign the dynamically created model
        )
        obj._metadata = metadata
        obj._created = record.created
        obj._approved = record.approved
        obj._flagged = record.flagged
        obj._agent_id = record.agent_id
        obj._model = record.model

        return obj

    def to_v1(self) -> V1Prompt:
        return V1Prompt(
            id=self._id,
            namespace=self._namespace,
            thread=self._thread.to_v1(),
            response=self._response.to_v1(),
            response_schema=(
                self._response_schema.model_json_schema()
                if self._response_schema
                else None
            ),
            metadata=self._metadata,
            created=self._created,
            approved=self._approved,
            flagged=self._flagged,
            agent_id=self._agent_id,
            model=self._model,
        )

    @classmethod
    def from_v1(cls, v1: V1Prompt) -> "Prompt":
        obj = cls.__new__(cls)

        response_schema_model = None
        if v1.response_schema:
            response_schema_model = cls._schema_to_model(v1.response_schema)

        obj._id = v1.id
        obj._namespace = v1.namespace
        obj._thread = RoleThread.from_v1(v1.thread)
        obj._response = RoleMessage.from_v1(v1.response)
        obj._response_schema = response_schema_model
        obj._metadata = v1.metadata
        obj._created = v1.created
        obj._approved = v1.approved
        obj._flagged = v1.flagged
        obj._agent_id = v1.agent_id
        obj._model = v1.model

        return obj

    @staticmethod
    def _schema_to_model(schema: Dict[str, Any]) -> Type[BaseModel]:
        model_name = schema.get("title", "DynamicResponseSchema")
        fields = {}
        for name, spec in schema.get("properties", {}).items():
            pydantic_type = {
                "string": str,
                "integer": int,
                "boolean": bool,
                "number": float,
            }.get(spec["type"], str)
            default = ... if name in schema.get("required", []) else None
            fields[name] = (pydantic_type, Field(default, **spec))

        response_schema_model = create_model(model_name, **fields)
        return response_schema_model

    @classmethod
    def find(cls, **kwargs) -> List["Prompt"]:
        for db in cls.get_db():
            records = (
                db.query(PromptRecord)
                .filter_by(**kwargs)
                .order_by(PromptRecord.created.desc())
                .all()
            )
            return [cls.from_record(record) for record in records]
        raise ValueError("No session")

    def save(self) -> None:
        logger.debug("saving prompt", self._id)
        for db in self.get_db():
            db.merge(self.to_record())
            db.commit()

    @classmethod
    def delete(cls, id: str) -> None:
        for db in cls.get_db():
            record = db.query(PromptRecord).filter_by(id=id).first()
            if record:
                db.delete(record)
                db.commit()

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.save()
