from pydantic import BaseModel

class InputDataModel(BaseModel):
    username: str
    limit: int
	
class OutputDataModel(BaseModel):
    message: str
    recommendations: list = []

class UserDataModel(BaseModel):
    users: list = []