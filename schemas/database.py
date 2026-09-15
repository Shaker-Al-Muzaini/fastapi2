from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# 1. مسار الاتصال بقاعدة بيانات PostgreSQL 18 في Laragon
SQLALCHEMY_DATABASE_URL = "postgresql+asyncpg://postgres:@localhost:5432/blog"

# 2. إنشاء محرك الاتصال غير المتزامن (Engine) بدون إعدادات SQLite القديمة
engine = create_async_engine(SQLALCHEMY_DATABASE_URL)

# 3. إنشاء مصنع جلسات الاتصال غير المتزامنة (Session Maker)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession)

# 4. الكلاس الأساسي الذي ترث منه جداول قاعدة البيانات
class Base(DeclarativeBase):
    pass

# 5. الدالة المساعدة (Dependency) لفتح وإغلاق الاتصال تلقائياً مع FastAPI
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
