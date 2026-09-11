from fastapi import APIRouter, Depends, Request, HTTPException, status, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from typing import Annotated
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from schemas.schemPost import PostResponse, PostCreate
from schemas.database import get_db
import schemas.model

# ترك الراوتر فارغاً بدون بادئة يدوية هنا بناءً على صورتك
router = APIRouter()
templates = Jinja2Templates(directory="templates")

# ===========================================================================
# 🔌 واجهات برمجة التطبيقات (APIs)
# ===========================================================================

# تم ترك المسار الأساسي فارغاً "" ليكتسب البادئة كاملة من ملف main.py
@router.post("", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
async def create_post(post: PostCreate, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(schemas.model.User).where(schemas.model.User.id == post.user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
    new_post = schemas.model.Post(title=post.title, content=post.content, user_id=post.user_id)
    db.add(new_post)
    await db.commit()
    await db.refresh(new_post) 
    return new_post


# ===========================================================================
# 📝 عمليات تعديل وحذف المنشورات (Frontend Actions)
# ===========================================================================

@router.get("/{post_id}/update", include_in_schema=False, name="update_post")
async def update_post_page(request: Request, post_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(schemas.model.Post).where(schemas.model.Post.id == post_id))
    post = result.scalars().first()
    
    if not post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
        
    return templates.TemplateResponse(
        request,
        "create_post.html",  
        {"post": post, "title": "Update Post", "legend": "Update Post"}
    )

@router.post("/{post_id}/update", include_in_schema=False)
async def update_post(
    post_id: int, 
    title: Annotated[str, Form()], 
    content: Annotated[str, Form()], 
    db: Annotated[AsyncSession, Depends(get_db)]
):
    result = await db.execute(select(schemas.model.Post).where(schemas.model.Post.id == post_id))
    post = result.scalars().first()
    
    if not post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
    
    post.title = title
    post.content = content
    await db.commit()
    
    # تحويل آمن باستخدام المتغير post_id لمنع أخطاء السيرفر
    return RedirectResponse(url=f"/posts/{post_id}", status_code=status.HTTP_302_FOUND)

@router.post("/{post_id}/delete", include_in_schema=False, name="delete_post")
async def delete_post(post_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(schemas.model.Post).where(schemas.model.Post.id == post_id))
    post = result.scalars().first()
    
    if not post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
        
    await db.delete(post)
    await db.commit()
    
    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
