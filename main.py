posts: list[dict] = [
    {
        "id": 1,
        "author": "Corey Schafer",
        "title": "FastAPI is Awesome",
        "content": "This framework is really easy to use and super fast.",
        "date_posted": "2026-04-20T00:00:00",
    },
    {
        "id": 2,
        "author": "Jane Doe",
        "title": "Python is Great for Web Development",
        "content": "Python is a great language for web development, and FastAPI makes it even better.",
        "date_posted": "2026-07-20T00:00:00",
    },
]

from fastapi import FastAPI , Request ,HTTPException, status
from httpx import post, request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from schemas.schemPost import PostResponse,PostCreate
app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
@app.get("/",response_class=HTMLResponse, name="home",response_model=list[PostResponse])
def read_root(request: Request):
    return templates.TemplateResponse(request,"home.html",{"posts": posts, "title": "Home Page"})
# عندما نقوم بارجع قيمه josn api يتم تطبيق  PostResponse ام في ارجاع html لا يتم التطبيق  

# @app.get("/",response_model=list[PostResponse])
# def read_root():
#     return posts

@app.get("/sn/{id}",
         name="sn",
         response_model=PostResponse
        )
def read_sn( id: str,request: Request):
    post = next((p for p in posts if str(p["id"]) == id), None)
    accept_header = request.headers.get("accept", "")
    if "application/json" in accept_header:
        if post:
            return JSONResponse(content={"success": True, "post": post, "errors": []}, status_code=200)

        # إذا لم يجد المنشور يعيد خطأ JSON برقم 404
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not exist.")

    if post:
        return templates.TemplateResponse(
            request, "index.html",
            {"posts": [post], "title": post["title"], "errors": []},
            status_code=200
        )

    return templates.TemplateResponse(
        request, "index.html",
        {"posts": [], "title": "Post Not Found", "errors": ["Post not exist."]},
        status_code=404
    )

@app.post(
    "/posts",
    response_model=PostResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_post(post: PostCreate):
    new_id = max(p["id"] for p in posts) + 1 if posts else 1
    new_post = {
        "id": new_id,
        **post.model_dump(),
    }
    posts.append(new_post)
    return new_post

