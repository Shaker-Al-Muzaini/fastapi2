from fastapi import APIRouter, Depends, HTTPException, Request, status, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from typing import Annotated
from fastapi.templating import Jinja2Templates
import shutil
import os
import random 

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# 1. تحديث مسارات الاستيراد لتتطابق مع بقية أجزاء المشروع
from schemas.schemPost import UserResponse, UserCreate, PostResponse
from schemas.database import get_db
from schemas import model as models

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# ===========================================================================
# 🔌 واجهات ودوال البروفايل (Profile & HTMX Actions)
# ===========================================================================

# 1. دالة عرض صفحة البروفايل الشاملة (تعديل نوع user_id إلى int)
@router.get("/{user_id}/profile", include_in_schema=False, name="user_profile")
async def user_profile_page(request: Request, user_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(models.User).where(models.User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
    posts_result = await db.execute(select(models.Post).where(models.Post.user_id == user_id))
    posts = posts_result.scalars().all()
        
    return templates.TemplateResponse(
        request, "profile.html", {"user": user, "posts": posts, "title": f"{user.username}'s Profile"}
    )


# 2. دالة معالجة واستلام تحديث الـ HTMX المباشر (تعديل نوع user_id إلى int)
@router.put("/{user_id}/update-html", response_class=HTMLResponse)
async def update_user_html(
    request: Request,
    user_id: int,
    username: Annotated[str, Form()],
    email: Annotated[str, Form()],
    db: Annotated[AsyncSession, Depends(get_db)],
    image: UploadFile = File(None)
):
    result = await db.execute(select(models.User).where(models.User.id == user_id))
    db_user = result.scalars().first()
    if not db_user:
        return HTMLResponse(content="المستخدم غير موجود!", status_code=404)
    
    # التحقق من عدم تكرار الاسم لحساب آخر
    username_check = await db.execute(
        select(models.User).where(models.User.username == username, models.User.id != user_id)
    )
    if username_check.scalars().first():
        return HTMLResponse(content="اسم المستخدم هذا مأخوذ بالفعل من قِبل حساب آخر!", status_code=400)

    db_user.username = username
    db_user.email = email

    # حفظ وتخزين الصورة في الميديا
    if image and image.filename:
        upload_dir = "media/profile_pics"
        os.makedirs(upload_dir, exist_ok=True)
        
        _, ext = os.path.splitext(image.filename)
        # تعديل التسمية البرمجية لتعتمد على رقم المعرف الـ int بشكل نصي
        filename = f"user_{str(user_id)}{ext.lower()}"
        filepath = os.path.join(upload_dir, filename)
        
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            
        db_user.image_file = filename

    await db.commit()
    await db.refresh(db_user) 

    # إعادة كتلة الـ HTML الفرعية المحدثة مباشرة لحقنها في الـ id="profile-card"
    content = f"""
    <div id="profile-card" class="d-flex align-items-center gap-4 border-bottom pb-4 flex-row-reverse justify-content-end" style="direction: rtl; text-align: right;">
        <img class="rounded-circle account-img border" 
             src="{db_user.image_path}?t={random.randint(1, 100000)}" 
             alt="{db_user.username}'s profile" 
             width="125" height="125" style="object-fit: cover;">
        <div>
            <h2 class="account-heading fw-bold mb-1">{db_user.username}</h2>
            <p class="text-secondary mb-3">{db_user.email}</p>
            <button type="button" class="btn btn-outline-info btn-sm fw-bold" data-bs-toggle="modal" data-bs-target="#editProfileModal">
                تعديل بيانات الحساب
            </button>
        </div>
    </div>
    """
    return HTMLResponse(content=content)

# ===========================================================================
# 🌐 واجهة إنشاء مستخدم جديد (JSON API) بمعرف رقمي تلقائي
# ===========================================================================

@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate, 
    db: Annotated[AsyncSession, Depends(get_db)]
):
    # 1. التحقق من عدم تكرار اسم المستخدم
    username_check = await db.execute(
        select(models.User).where(models.User.username == user_data.username)
    )
    if username_check.scalars().first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="اسم المستخدم هذا مأخوذ بالفعل!"
        )
        
    # 2. التحقق من عدم تكرار البريد الإلكتروني
    email_check = await db.execute(
        select(models.User).where(models.User.email == user_data.email)
    )
    if email_check.scalars().first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="البريد الإلكتروني مستخدم بالفعل!"
        )

    # 3. إنشاء كائن المستخدم الجديد (قاعدة البيانات ستعطيه الرقم 1 وتزيد تلقائياً)
    new_user = models.User(
        username=user_data.username,
        email=user_data.email
    )

    # 4. حفظ البيانات (Async Commit)
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    # 5. إرجاع استجابة الـ JSON
    return new_user
