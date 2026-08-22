from fastapi import FastAPI
from pydantic import BaseModel ,HttpUrl
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World shaker"}

class tesigg(BaseModel):
    name: str
    age: int
    website: HttpUrl 

@app.post("/items/")
def view(item: tesigg):
    return {"name": item.name, "age": item.age, "website": item.website}
