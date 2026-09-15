from fastapi import APIRouter, Depends, Request, HTTPException, status, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from typing import Annotated
# تم حذف مكتبة uuid بالكامل للتوافق مع معرفات int التلقائية

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

# 1. تحديث مسارات الاستيراد لتتطابق مع بقية أجزاء المشروع
from schemas.schemPost import PostResponse, PostCreate
from schemas.database import get_db
from schemas import model as models

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# ===========================================================================
# 🔌 واجهات برمجة التطبيقات (APIs) - لإنشاء المنشور كـ JSON
# ===========================================================================

@router.post("", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
async def create_post(post: PostCreate, db: Annotated[AsyncSession, Depends(get_db)]):
    # التحقق من توافق المعرف الرقمي للمخدم
    result = await db.execute(select(models.User).where(models.User.id == post.user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
    new_post = models.Post(title=post.title, content=post.content, user_id=post.user_id)
    db.add(new_post)
    await db.commit()
    await db.refresh(new_post) 
    
    # تحميل بيانات الكاتب لترجع مع الـ response_model بشكل سليم وتجنب الأخطاء
    post_result = await db.execute(
        select(models.Post).options(selectinload(models.Post.author)).where(models.Post.id == new_post.id)
    )
    return post_result.scalars().first()


# ===========================================================================
# 📝 عمليات تعديل وحذف المنشورات وجلب الصفحات (Frontend HTML Actions)
# ===========================================================================

# 1. دالة عرض صفحة التعديل (تم تحويل post_id إلى int)
@router.get("/{post_id}/update", include_in_schema=False, name="update_post")
async def update_post_page(request: Request, post_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(models.Post).where(models.Post.id == post_id))
    post = result.scalars().first()
    
    if not post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
        
    return templates.TemplateResponse(
        request,
        "create_post.html",  
        {"post": post, "title": "Update Post", "legend": "Update Post"}
    )

# 2. دالة استقبال ومعالجة التعديل الفعلي عبر HTMX (تم تحويل post_id إلى int)
@router.put("/{post_id}/update", include_in_schema=False)
async def update_post(
    post_id: int, 
    title: Annotated[str, Form()], 
    content: Annotated[str, Form()], 
    db: Annotated[AsyncSession, Depends(get_db)]
):
    result = await db.execute(select(models.Post).where(models.Post.id == post_id))
    post = result.scalars().first()
    
    if not post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
    
    post.title = title
    post.content = content
    await db.commit()
    
    response = HTMLResponse(content="تم التحديث بنجاح")
    response.headers["HX-Redirect"] = f"/posts/{post_id}"
    return response

# 3. دالة حذف المنشور والتحويل التلقائي (تم تحويل post_id إلى int)
@router.post("/{post_id}/delete", include_in_schema=False, name="delete_post")
async def delete_post(post_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(models.Post).where(models.Post.id == post_id))
    post = result.scalars().first()
    
    if not post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
        
    await db.delete(post)
    await db.commit()
    
    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
