from fastapi import Depends, FastAPI , Request ,HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from schemas.schemPost import PostResponse, PostCreate, UserResponse, UserCreate
from datetime import datetime
from typing import Annotated
from sqlalchemy import select
from sqlalchemy.orm import Session
import schemas.model
from schemas.database import Base, get_db, engine

Base.metadata.create_all(bind=engine)

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/media", StaticFiles(directory="media"), name="media")
templates = Jinja2Templates(directory="templates")

# ==================== صفحات الـ HTML (Frontend) ====================

# 1. الصفحة الرئيسية
@app.get("/", include_in_schema=False, name="home")
@app.get("/posts", include_in_schema=False, name="posts")
def home(request: Request, db: Annotated[Session, Depends(get_db)]):
    result = db.execute(select(schemas.model.Post))
    posts = result.scalars().all()
    return templates.TemplateResponse(
        request,
        "home.html",
        {"posts": posts, "title": "Home"},
    )

# 2. صفحة منشورات مستخدم معين (تم تعديل المسار لمنع التضارب مع الـ API)
@app.get("/users/{user_id}/posts/page", include_in_schema=False, name="user_posts")
def user_posts_page(request: Request, user_id: int, db: Annotated[Session, Depends(get_db)]):
    result = db.execute(select(schemas.model.User).where(schemas.model.User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
    result = db.execute(select(schemas.model.Post).where(schemas.model.Post.user_id == user_id))
    posts = result.scalars().all()
    return templates.TemplateResponse(
        request,
        "user_posts.html",
        {"posts": posts, "user": user, "title": f"{user.username}'s Posts"},
    )

# 3. صفحة تفاصيل منشور فردي (تمت إضافة المائلة / وتوجيهها لملف تفاصيل المنشور)
@app.get("/posts/{post_id}", include_in_schema=False, name="post_page")
def get_post_page(request: Request, post_id: int, db: Annotated[Session, Depends(get_db)]):
    result = db.execute(select(schemas.model.Post).where(schemas.model.Post.id == post_id))
    post = result.scalars().first()
    
    if not post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
        
    # هنا نقوم باستدعاء قالب تفاصيل المنشور الفردي وتمريره كـ post مفرد
    return templates.TemplateResponse(
        request,
        "index.html", 
        {"post": post, "title": post.title}
    )


# ==================== واجهات برمجة التطبيقات (APIs - JSON) ====================

@app.post("/api/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, db: Annotated[Session, Depends(get_db)]):
    existing_user = db.execute(
        select(schemas.model.User).where(schemas.model.User.username == user.username)
    ).scalar_one_or_none()

    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
        
    existing_email = db.execute(
        select(schemas.model.User).where(schemas.model.User.email == user.email)
    ).scalar_one_or_none()

    if existing_email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already exists")

    new_user = schemas.model.User(**user.model_dump())
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.get("/api/users/{user_id}", response_model=UserResponse)
def get_user_api(user_id: int, db: Annotated[Session, Depends(get_db)]):
    result = db.execute(select(schemas.model.User).where(schemas.model.User.id == user_id))
    user = result.scalars().first()
    if user:
        return user
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

@app.get("/api/users/{user_id}/posts", response_model=list[PostResponse])
def get_user_posts_api(user_id: int, db: Annotated[Session, Depends(get_db)]):
    result = db.execute(select(schemas.model.User).where(schemas.model.User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    result = db.execute(select(schemas.model.Post).where(schemas.model.Post.user_id == user_id))
    posts = result.scalars().all()
    return posts
@app.get("/posts/{post_id}", include_in_schema=False, name="post_page")
def get_post(request: Request, post_id: int, db: Annotated[Session, Depends(get_db)]):
    result = db.execute(
        select(schemas.model.Post).where(schemas.model.Post.id == post_id)
    )
    post = result.scalars().first()
    
    if not post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
        
    # التعديل هنا: وجهنا الكود لفتح ملف index.html وتمرير المنشور ككائن مفرد
    return templates.TemplateResponse(
        request,
        "index.html", 
        {"post": post, "title": post.title}
    )



@app.post("/api/posts", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
def create_post(post: PostCreate, db: Annotated[Session, Depends(get_db)]):
    result = db.execute(select(schemas.model.User).where(schemas.model.User.id == post.user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
    new_post = schemas.model.Post(title=post.title, content=post.content, user_id=post.user_id)
    db.add(new_post)
    db.commit()
    db.refresh(new_post) 
    return new_post



from fastapi.responses import RedirectResponse
from fastapi import Form

# 📝 1. عرض صفحة تعديل المنشور
@app.get("/posts/{post_id}/update", include_in_schema=False, name="update_post")
def update_post_page(request: Request, post_id: int, db: Annotated[Session, Depends(get_db)]):
    result = db.execute(select(schemas.model.Post).where(schemas.model.Post.id == post_id))
    post = result.scalars().first()
    
    if not post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
        
    # هنا نقوم بفتح قالب مخصص لتعديل البيانات ونمرر المنشور الحالي
    return templates.TemplateResponse(
        request,
        "create_post.html",  # سنعيد استخدام قالب إنشاء المنشور بعد تعديله ليدعم التعديل أيضاً
        {"post": post, "title": "Update Post", "legend": "Update Post"}
    )

# 💾 2. معالجة بيانات التعديل القادمة من الفورم
@app.post("/posts/{post_id}/update", include_in_schema=False)
def update_post(
    post_id: int, 
    title: Annotated[str, Form()], 
    content: Annotated[str, Form()], 
    db: Annotated[Session, Depends(get_db)]
):
    result = db.execute(select(schemas.model.Post).where(schemas.model.Post.id == post_id))
    post = result.scalars().first()
    
    if not post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
    
    # تحديث الحقول في قاعدة البيانات
    post.title = title
    post.content = content
    db.commit()
    
    # بعد النجاح، يتم توجيهه إلى صفحة تفاصيل المنشور المعدّل
    return RedirectResponse(url=f"/posts/{post.id}", status_code=status.HTTP_302_FOUND)

# ❌ 3. تنفيذ عملية الحذف
@app.post("/posts/{post_id}/delete", include_in_schema=False, name="delete_post")
def delete_post(post_id: int, db: Annotated[Session, Depends(get_db)]):
    result = db.execute(select(schemas.model.Post).where(schemas.model.Post.id == post_id))
    post = result.scalars().first()
    
    if not post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
        
    db.delete(post)
    db.commit()
    
    # بعد الحذف بنجاح، يتم توجيهه إلى الصفحة الرئيسية
    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
