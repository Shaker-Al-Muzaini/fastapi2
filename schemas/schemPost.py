from pydantic import BaseModel, EmailStr, Field, ConfigDict 
from datetime import datetime

# ==================== كلاسات المستخدم (User) ====================
class UserBase(BaseModel):
    username: str = Field(max_length=50)
    email: EmailStr = Field(max_length=120)

class UserCreate(UserBase):
    pass

class UserResponse(UserBase):
    model_config = ConfigDict(from_attributes=True)

    id: int  # التعديل: تغيير المعرف إلى رقمي ليتوافق مع serial4
    image_file: str | None
    image_path: str

# ==================== كلاسات المنشورات (Post) ====================
class PostBase(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=50)
    content: str = Field(min_length=1, max_length=500)

class PostCreate(PostBase):
    user_id: int  # التعديل: قبول الرقم 1 وربطه بالمستخدم رقمياً

class PostResponse(PostBase):
    model_config = ConfigDict(from_attributes=True)
    
    id: int           # التعديل: معرف المنشور أصبح رقماً تسلسلياً
    date_posted: datetime
    user_id: int      # التعديل: معرف المستخدم المرتبط أصبح رقماً
    author: UserResponse    
