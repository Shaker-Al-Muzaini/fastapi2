from pydantic import BaseModel, EmailStr, Field, ConfigDict 
from datetime import datetime

class PostBase(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=50)
    content: str = Field(min_length=1, max_length=500)

class PostCreate(PostBase):
    user_id: int # يرسل المستخدم رقم الـ id الخاص به فقط

class PostResponse(PostBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    date_posted: datetime
    user_id: int
    author: UserResponse
    
class UserBase(BaseModel):
    username: str = Field(max_length=50)
    email: EmailStr = Field(max_length=120)


class UserCreate(UserBase):
    pass

class UserResponse(UserBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    image_file: str | None
    image_path: str
   
