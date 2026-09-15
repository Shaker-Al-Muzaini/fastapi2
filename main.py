from fastapi import Depends, FastAPI, Request, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.templating import Jinja2Templates
from typing import Annotated
from contextlib import asynccontextmanager

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from schemas import model
from schemas.database import Base, get_db, engine

from routers import user_router, post_router
from routers.video import router as video_router
# --- السطر الجديد المطلوب (1) ---
from routers.video import tts_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="My Professional Blog", lifespan=lifespan)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/media", StaticFiles(directory="media"), name="media")
templates = Jinja2Templates(directory="templates")

# ===========================================================================
# 🔌 تضمين الراوترات الفرعية
# ===========================================================================

app.include_router(user_router, prefix="/api/users", tags=["API Users"])
app.include_router(user_router, prefix="/users", tags=["Users"])
app.include_router(post_router, prefix="/api/posts", tags=["API Posts"])
app.include_router(post_router, prefix="/posts")

# ===========================================================================
# 🌐 صفحات الـ HTML الأساسية (Main View Routes)
# ===========================================================================

@app.get("/", include_in_schema=False, name="home")
@app.get("/posts", include_in_schema=False, name="posts")
async def home(request: Request, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(
        select(model.Post).options(selectinload(model.Post.author))
    )
    posts = result.scalars().all()
    return templates.TemplateResponse(
        request, "home.html", {"posts": posts, "title": "Home"}
    )


@app.get("/users/{user_id}/posts/page", include_in_schema=False, name="user_posts")
async def user_posts_page(request: Request, user_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(model.User).where(model.User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    result = await db.execute(
        select(model.Post)
        .where(model.Post.user_id == user_id)
        .options(selectinload(model.Post.author))
    )
    posts = result.scalars().all()
    return templates.TemplateResponse(
        request, "user_posts.html", {"posts": posts, "user": user, "title": f"{user.username}'s Posts"}
    )


@app.get("/posts/{post_id}", include_in_schema=False, name="post_page")
async def get_post_page(request: Request, post_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(
        select(model.Post)
        .where(model.Post.id == post_id)
        .options(selectinload(model.Post.author))
    )
    post = result.scalars().first()

    if not post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")

    return templates.TemplateResponse(
        request, "index.html", {"post": post, "title": post.title}
    )


@app.get("/test", )
async def test(request: Request):
    return templates.TemplateResponse(request, "test.html", {"title": "Test"})

@app.get("/dub_preview", include_in_schema=False)
async def dub_preview(request: Request):
    return templates.TemplateResponse(request, "dub_preview.html", {"title": "معاينة الفيديو المدمج"})

app.include_router(video_router, prefix="/api/video", tags=["Video Downloader"])
# --- السطر الجديد المطلوب (2) ---
app.include_router(tts_router, prefix="/api", tags=["Text-to-Speech"])