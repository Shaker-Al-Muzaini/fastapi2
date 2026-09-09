from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# 1. تحديد مسار قاعدة البيانات (هنا سننشئ قاعدة بيانات SQLite محلية باسم blog.db)
SQLALCHEMY_DATABASE_URL = "sqlite:///./blog.db"

# 2. إنشاء محرك الاتصال (Engine)
# الخاصية connect_args مطلوبة فقط مع SQLite لمنع مشاكل تعدد الخيوط (Threads)
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# 3. إنشاء جلسة اتصال مخصصة لقراءة وكتابة البيانات (Session)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. الكلاس الأساسي الذي سترث منه جداول قاعدة البيانات مستقبلاً (Models)
Base = declarative_base()

# 5. دالة مساعدة لفتح وإغلاق الاتصال تلقائياً مع FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
