from pydantic import BaseModel,Field,ConfigDict,ValidationError
from datetime import datetime

class PostBase(BaseModel):
    title: str = Field(min_length=1, max_length=50,strip_whitespace=True)
    content: str = Field(min_length=1,max_length=500,strip_whitespace=True)
    author: str = Field(min_length=1, max_length=50,strip_whitespace=True)

class PostCreate(PostBase):
    pass
   


class PostResponse(PostBase):
    model_config= ConfigDict(from_attributes=True)
    id: int
    date_posted: datetime