from fastapi import Depends, FastAPI, Request, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.templating import Jinja2Templates
from typing import Annotated
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload  
import schemas.model
from schemas.database import Base, get_db, engine

# 1. استيراد الراوترات الفرعية من مجلد routers
from routers.users import router as user_router
from routers.posts import router as post_router

app = FastAPI(title="My Professional Blog")

# إنشاء الجداول تلقائياً في قاعدة البيانات بشكل غير متزامن عند تشغيل التطبيق
@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/media", StaticFiles(directory="media"), name="media")
templates = Jinja2Templates(directory="templates")


# ===========================================================================
# 🔌 تضمين الراوترات الفرعية وتشكيل مساراتها مركزياً (مثل Laravel)
# ===========================================================================

# تشكيل مسار مستخدمي الـ API ليصبح: /api/users
app.include_router(user_router, prefix="/api/users", tags=["API Users"])

# تشكيل مسار منشورات الـ API لتبدأ بـ: /api/posts وعمليات التعديل والحذف لتبدأ بـ: /posts
app.include_router(post_router, prefix="/api/posts", tags=["API Posts"])
# لتضمين مسارات التعديل والحذف التي تبدأ بـ /posts بدون كلمة api، نضمن الراوتر مرة أخرى ببادئة مختلفة
app.include_router(post_router, prefix="/posts")


# ===========================================================================
# 🌐 صفحات الـ HTML الأساسية (محتفظ بها في الـ main بناءً على طلبك)
# ===========================================================================

# 1. الصفحة الرئيسية
@app.get("/", include_in_schema=False, name="home")
@app.get("/posts", include_in_schema=False, name="posts")
async def home(request: Request, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(
        select(schemas.model.Post).options(selectinload(schemas.model.Post.author))
    )
    posts = result.scalars().all()
    return templates.TemplateResponse(
        request,
        "home.html",
        {"posts": posts, "title": "Home"},
    )

# 2. صفحة منشورات مستخدم معين
@app.get("/users/{user_id}/posts/page", include_in_schema=False, name="user_posts")
async def user_posts_page(request: Request, user_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(schemas.model.User).where(schemas.model.User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
    result = await db.execute(
        select(schemas.model.Post)
        .where(schemas.model.Post.user_id == user_id)
        .options(selectinload(schemas.model.Post.author))
    )
    posts = result.scalars().all()
    return templates.TemplateResponse(
        request,
        "user_posts.html",
        {"posts": posts, "user": user, "title": f"{user.username}'s Posts"},
    )

# 3. صفحة تفاصيل منشور فردي
@app.get("/posts/{post_id}", include_in_schema=False, name="post_page")
async def get_post_page(request: Request, post_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(
        select(schemas.model.Post)
        .where(schemas.model.Post.id == post_id)
        .options(selectinload(schemas.model.Post.author))
    )
    post = result.scalars().first()
    
    if not post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
        
    return templates.TemplateResponse(
        request,
        "index.html", 
        {"post": post, "title": post.title}
    )
