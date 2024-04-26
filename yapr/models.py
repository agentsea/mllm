from typing import Optional, Dict, Any, List

from pydantic import BaseModel
from threadmem.server.models import RoleThreadModel, RoleMessageModel


class PromptModel(BaseModel):
    """An LLM prompt model"""

    id: Optional[str] = None
    thread: RoleThreadModel
    response: RoleMessageModel
    namespace: str = "default"
    metadata: Dict[str, Any] = {}
    created: Optional[float] = None
    approved: bool = False
    flagged: bool = False


class EnvVarOptModel(BaseModel):
    name: str
    description: Optional[str] = None
    required: bool = False
    default: Optional[str] = None
    secret: bool = False
    options: List[str] = []


class LLMProviders(BaseModel):
    preference: List[str] = []


class LLMProviderOption(BaseModel):
    model: str
    env_var: EnvVarOptModel


class LLMProviderModel(BaseModel):
    options: List[LLMProviderOption]
