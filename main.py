<<<<<<< HEAD
# from fastapi import FastAPI, HTTPException ,status, Request
# import psycopg2
# from psycopg2.extras import RealDictCursor
# from contextlib import asynccontextmanager
# from pydantic import BaseModel
=======
from fastapi import FastAPI, HTTPException ,status, Request
import psycopg2 
from psycopg2.extras import RealDictCursor
from contextlib import asynccontextmanager
from pydantic import BaseModel
>>>>>>> 55a05f8b405d0c4400d42dacf1e37bba2457712b

# db_connection = None

<<<<<<< HEAD
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global db_connection
#     try:
#         db_connection = psycopg2.connect(
#             host='localhost', 
#             database='test', 
#             user='postgres', 
#             password='',
#             cursor_factory=RealDictCursor
#         )
#         print('Successfully connected to Database permanently!')
#     except Exception as error:
#         print("Database connection failed:", error)
=======
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_connection
    try:
        db_connection = psycopg2.connect(
            host='localhost', 
            database='test', 
            user='postgres', 
            user='postgres2', 
            password='',
            cursor_factory=RealDictCursor
        )
        print('Successfully connected to Database permanently!')
    except Exception as error:
        print("Database connection failed:", error)
>>>>>>> 55a05f8b405d0c4400d42dacf1e37bba2457712b
    
#     yield
#     if db_connection:
#         db_connection.close()

# app = FastAPI(lifespan=lifespan)

# # الـ Model الخاص بالبيانات
# class tesigg(BaseModel):
#     name: str
#     age: int

# @app.get("/")
# def get_students():
#     if db_connection is None:
#         raise HTTPException(status_code=500, detail="Database connection is offline")
    
#     cursor = db_connection.cursor()
#     # تم تعديل اسم الجدول هنا إلى studant ليطابق قاعدة بياناتك
#     cursor.execute("SELECT name, age FROM studant;") 
#     students = cursor.fetchall()
#     cursor.close() 
    
#     return {"Hello": "World shaker", "student": students}

# @app.post("/")
# def view(item: tesigg):
#     if db_connection is None:
#         raise HTTPException(status_code=500, detail="Database connection is offline")
    
#     try:
#         cursor = db_connection.cursor()
        
#         # تم تعديل اسم الجدول هنا أيضاً إلى studant ليتم الإدخال بنجاح
#         cursor.execute("""INSERT INTO studant (name, age) VALUES (%s, %s) RETURNING * """, (item.name, item.age))
#         new_post = cursor.fetchone()
        
#         db_connection.commit()
#         cursor.close()
        
#         return {"data": new_post}
#     except Exception as error:
#         raise HTTPException(status_code=400, detail=f"حدث خطأ أثناء الإدخال: {str(error)}")

# @app.get("/{id}")
# def get_student(id: int):
#     # 1. فحص أمان للتأكد من أن الاتصال بقاعدة البيانات يعمل
#     if db_connection is None:
#         raise HTTPException(status_code=500, detail="Database connection is offline")
    
#     try:
#         # 2. إنشاء الـ cursor من الاتصال المركزي الدائم
#         cursor = db_connection.cursor()
        
#         # 3. تنفيذ استعلام جلب الطالب بناءً على الـ id من جدول studant
#         cursor.execute("""SELECT * FROM studant WHERE id = %s """, (id,))
#         course = cursor.fetchone()
        
#         # 4. إغلاق الـ cursor بعد جلب البيانات
#         cursor.close()
        
#         # 5. التحقق إذا كان المعرّف (id) غير موجود في الجدول
#         if not course:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Student with id: {id} was not found"
#             )
        
#         # 6. إرجاع تفاصيل الطالب بنجاح
#         return {"Course_detail": course}
        
#     except HTTPException as http_err:
#         raise http_err
#     except Exception as error:
#         raise HTTPException(status_code=400, detail=f"حدث خطأ أثناء جلب البيانات: {str(error)}")
 
# posts: list[dict] = [
#     {
#         "id": 1,
#         "author": "Corey Schafer",
#         "title": "FastAPI is Awesome",
#         "content": "This framework is really easy to use and super fast.",
#         "date_posted": "April 20, 2026",
#     },
#     {
#         "id": 2,
#         "author": "Jane Doe",
#         "title": "Python is Great for Web Development",
#         "content": "Python is a great language for web development, and FastAPI makes it even b",
#         "date_posted": "April 21, 2026",
#     },
# ]