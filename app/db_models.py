# app/db_models.py

from datetime import datetime
from sqlalchemy import Column, Integer, Numeric, Boolean, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from .database import Base  # Import Base class จากไฟล์ database.py
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy import Table

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
    lime_explanation = Column(JSONB, nullable=True)
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
    optimal_threshold = Column(Numeric, nullable=True)

    # ตารางสำหรับเก็บความสัมพันธ์ Many-to-Many ระหว่าง User และ Role
user_roles = Table('user_roles', Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('role_id', Integer, ForeignKey('roles.id'))
)

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    
    # ความสัมพันธ์: User หนึ่งคนสามารถมีได้หลาย Role
    roles = relationship("Role", secondary=user_roles, back_populates="users")

class Role(Base):
    __tablename__ = 'roles'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)
    description = Column(String(255))
    
    # ความสัมพันธ์: Role หนึ่งสามารถมีได้หลาย User
    users = relationship("User", secondary=user_roles, back_populates="roles")