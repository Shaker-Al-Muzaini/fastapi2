posts: list[dict] = [
    {
        "id": 1,
        "author": "Corey Schafer",
        "title": "FastAPI is Awesome",
        "content": "This framework is really easy to use and super fast.",
        "date_posted": "April 20, 2026",
    },
    {
        "id": 2,
        "author": "Jane Doe",
        "title": "Python is Great for Web Development",
        "content": "Python is a great language for web development, and FastAPI makes it even better.",
        "date_posted": "April 21, 2026",
    },
]



from fastapi import FastAPI , Request , HTTPException, status
from httpx import post, request
app = FastAPI()
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse, name="home")
def read_root(request: Request):
    return templates.TemplateResponse(request,"home.html", {"posts": posts, "title": "Home Page"})

@app.get("/sn/{id}",name="sn")
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
