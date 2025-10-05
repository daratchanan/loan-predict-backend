# app/database.py

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv

# โหลดค่า Environment Variables จากไฟล์ .env
load_dotenv()

# ดึงค่าการเชื่อมต่อจาก Environment Variables
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# สร้าง Connection String สำหรับ PostgreSQL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# สร้าง SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# สร้าง SessionLocal class สำหรับสร้าง Database Session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# สร้าง Base class สำหรับ ORM models (เพื่อให้ db_models.py นำไปใช้)
Base = declarative_base()


# --- ฟังก์ชัน Dependency สำหรับ FastAPI ---
def get_db():
    """
    ฟังก์ชันนี้จะสร้างและจัดการ Database Session สำหรับแต่ละ Request
    ที่เข้ามายัง API Endpoint และจะปิด Session อัตโนมัติเมื่อ Request จบลง
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()