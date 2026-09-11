from fastapi import APIRouter, Depends, HTTPException, Request, status, Form, UploadFile, File
from typing import Annotated
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from schemas.schemPost import UserResponse, UserCreate, PostResponse
from schemas.database import get_db
import schemas.model
import shutil
import os

# إنشاء كائن راوتر موحد ومتوافق مع إعدادات التضمين في main.py
router = APIRouter()
templates = Jinja2Templates(directory="templates")

# ==================== واجهات برمجة التطبيقات (APIs) ====================

@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate, db: Annotated[AsyncSession, Depends(get_db)]):
    existing_user = (await db.execute(
        select(schemas.model.User).where(schemas.model.User.username == user.username)
    )).scalar_one_or_none()

    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
        
    existing_email = (await db.execute(
        select(schemas.model.User).where(schemas.model.User.email == user.email)
    )).scalar_one_or_none()

    if existing_email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already exists")

    new_user = schemas.model.User(**user.model_dump())
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user

@router.get("/{user_id}", response_model=UserResponse)
async def get_user_api(user_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(schemas.model.User).where(schemas.model.User.id == user_id))
    user = result.scalars().first()
    if user:
        return user
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

@router.get("/{user_id}/posts", response_model=list[PostResponse])
async def get_user_posts_api(user_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(schemas.model.User).where(schemas.model.User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    result = await db.execute(select(schemas.model.Post).where(schemas.model.Post.user_id == user_id))
    posts = result.scalars().all()
    return posts

# ✨ واجهة الـ API الاحترافية لتحديث البيانات بالخلفية وتخزين الصورة في المجلد الصحيح
@router.put("/{user_id}/update-json", response_model=UserResponse)
async def update_user_json(
    user_id: int,
    username: Annotated[str, Form()],
    email: Annotated[str, Form()],
    db: Annotated[AsyncSession, Depends(get_db)],
    image: UploadFile = File(None)
):
    result = await db.execute(select(schemas.model.User).where(schemas.model.User.id == user_id))
    db_user = result.scalars().first()
    if not db_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    # التأكد من عدم تكرار الاسم لمستخدم آخر
    username_check = await db.execute(
        select(schemas.model.User).where(schemas.model.User.username == username, schemas.model.User.id != user_id)
    )
    if username_check.scalars().first():
        raise HTTPException(status_code=400, detail="Username already taken")

    db_user.username = username
    db_user.email = email

    # 🖼️ معالجة وحفظ الصورة الشخصية داخل مجلد media/profile_pics الصحيح
    if image and image.filename:
        upload_dir = "media/profile_pics"
        os.makedirs(upload_dir, exist_ok=True)
        
        # استخراج الامتداد الأصلي (مثل .jpg) وتوليد اسم فريد بدون مسافات وأقواس
        _, ext = os.path.splitext(image.filename)
        filename = f"user_{user_id}{ext.lower()}"
        filepath = os.path.join(upload_dir, filename)
        
        # حفظ الملف في المجلد الجديد المخصص
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            
        # تخزين الاسم النظيف الجديد في قاعدة البيانات
        db_user.image_file = filename

    await db.commit()
    await db.refresh(db_user)
    return db_user

# ==================== صفحات الـ HTML (Frontend) ====================

@router.get("/{user_id}/profile", include_in_schema=False, name="user_profile")
async def user_profile_page(request: Request, user_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(schemas.model.User).where(schemas.model.User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
    return templates.TemplateResponse(
        request, "profile.html", {"user": user, "title": f"{user.username}'s Profile"}
    )
