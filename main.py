from fastapi import FastAPI
from pydantic import BaseModel ,HttpUrl
import psycopg2
from psycopg2.extras import RealDictCursor
import time
app = FastAPI()

@app.get("/")
def read_root():
    cursor.execute("SELECT * FROM studant")
    data = cursor.fetchall()
    return {"Hello": "World shaker", "studant": data}

class tesigg(BaseModel):
    name: str
    age: int

while True:
    try:
        conn = psycopg2.connect(host='localhost', database='test', user='postgres', password='', 
            cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        print('Successfully connected Database')
        break
    except Exception as error:
        print('Database connection failed')
        print("Error:", error)
        time.sleep(2)
   

@app.post("/items/")
def view(item: tesigg):
    return {"name": item.name, "age": item.age}
