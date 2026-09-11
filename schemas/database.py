from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# 1. تحديد مسار قاعدة البيانات (SQLite محلية باسم blog.db عبر محرك aiosqlite)
SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///./blog.db"

# 2. إنشاء محرك الاتصال غير المتزامن (Engine)
# الخاصية connect_args مطلوبة فقط مع SQLite لمنع مشاكل تعدد الخيوط (Threads)
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# 3. إنشاء مصنع جلسات الاتصال غير المتزامنة (Session Maker)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession)

# 4. الكلاس الأساسي الذي ترث منه جداول قاعدة البيانات (Models) في SQLAlchemy 2.0
class Base(DeclarativeBase):
    pass

# 5. الدالة المساعدة (Dependency) لفتح وإغلاق الاتصال تلقائياً مع FastAPI
# تم الاعتماد على أسلوب async with الأفضل والأكثر أماناً لإدارة الجالات وإغلاقها تلقائياً [1]
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
