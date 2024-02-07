from pydantic import BaseModel
from typing import Any, Optional

class InputDataModel(BaseModel):
    user_id: str
    limit: int
	
class OutputDataModel(BaseModel):
    recommendations: Any

class UserDataModel(BaseModel):
    users: Any
