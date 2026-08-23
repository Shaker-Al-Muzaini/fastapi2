from fastapi import FastAPI, HTTPException
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import asynccontextmanager
from pydantic import BaseModel

db_connection = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_connection
    try:
        db_connection = psycopg2.connect(
            host='localhost', 
            database='test', 
            user='postgres', 
            password='',
            cursor_factory=RealDictCursor
        )
        print('Successfully connected to Database permanently!')
    except Exception as error:
        print("Database connection failed:", error)
    
    yield
    if db_connection:
        db_connection.close()

app = FastAPI(lifespan=lifespan)

# الـ Model الخاص بالبيانات
class tesigg(BaseModel):
    name: str
    age: int

@app.get("/")
def get_students():
    if db_connection is None:
        raise HTTPException(status_code=500, detail="Database connection is offline")
    
    cursor = db_connection.cursor()
    # تم تعديل اسم الجدول هنا إلى studant ليطابق قاعدة بياناتك
    cursor.execute("SELECT name, age FROM studant;") 
    students = cursor.fetchall()
    cursor.close() 
    
    return {"Hello": "World shaker", "student": students}

@app.post("/")
def view(item: tesigg):
    if db_connection is None:
        raise HTTPException(status_code=500, detail="Database connection is offline")
    
    try:
        cursor = db_connection.cursor()
        
        # تم تعديل اسم الجدول هنا أيضاً إلى studant ليتم الإدخال بنجاح
        cursor.execute("""INSERT INTO studant (name, age) VALUES (%s, %s) RETURNING * """, (item.name, item.age))
        new_post = cursor.fetchone()
        
        db_connection.commit()
        cursor.close()
        
        return {"data": new_post}
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"حدث خطأ أثناء الإدخال: {str(error)}")
