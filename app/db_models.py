# app/db_models.py


import datetime
from sqlalchemy import Column, Integer, Numeric, Boolean, String, DateTime
from sqlalchemy.sql import func
from .database import Base  # Import Base class จากไฟล์ database.py
from sqlalchemy.dialects.postgresql import JSONB

# คลาสนี้คือการจำลองตาราง loan_data ในรูปแบบของ Python class
class LoanData(Base):
    __tablename__ = 'loan_data'  # ระบุว่าคลาสนี้เชื่อมโยงกับตารางชื่อ 'loan_data'

    # กำหนดคอลัมน์และประเภทข้อมูลให้ตรงกับใน PostgreSQL
    id = Column(Integer, primary_key=True, index=True)
    application_date = Column(DateTime(timezone=True), server_default=func.now())
    purpose = Column(String(100))
    credit_policy = Column(Integer)
    int_rate = Column(Numeric)
    installment = Column(Numeric)
    log_annual_inc = Column(Numeric)
    dti = Column(Numeric)
    fico = Column(Integer)
    days_with_cr_line = Column(Numeric)
    revol_bal = Column(Integer)
    revol_util = Column(Numeric)
    inq_last_6mths = Column(Integer)
    delinq_2yrs = Column(Integer)
    pub_rec = Column(Integer)
    not_fully_paid = Column(Integer)
    estimated_credit_limit = Column(Numeric)
    installment_to_income_ratio = Column(Numeric)
    high_interest = Column(Integer)
    model_prediction = Column(Integer, nullable=True)
    model_probability = Column(Numeric, nullable=True)
class ModelPerformanceLog(Base):
    __tablename__ = 'model_performance_logs'
    id = Column(Integer, primary_key=True)
    model_version = Column(String(50))
    training_date = Column(DateTime, default=datetime.now)
    accuracy = Column(Numeric)
    precision_class_1 = Column(Numeric)
    recall_class_1 = Column(Numeric)
    f1_score_class_1 = Column(Numeric)
    is_active = Column(Boolean, default=False)
    meta_model_importances = Column(JSONB)