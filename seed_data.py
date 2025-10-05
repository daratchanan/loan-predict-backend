# import_initial_data.py
import pandas as pd
import numpy as np
import joblib
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# --- 1. โหลดโมเดลที่ Train เสร็จแล้ว ---
print("Step 1: กำลังโหลดโมเดลจาก app/ml/model_pipeline.joblib...")
try:
    model = joblib.load("app/ml/model_pipeline.joblib")
except FileNotFoundError:
    print("Error: ไม่พบไฟล์ model_pipeline.joblib! กรุณารัน train_model.py ก่อน")
    exit()

# --- 2. โหลดและเตรียมข้อมูล (ETL Process) ---
print("\nStep 2: กำลังโหลดและเตรียมข้อมูลจาก loan_data.csv...")
try:
    df = pd.read_csv('data/loan_data.csv')
    df.columns = df.columns.str.replace('.', '_')
    
    # Feature Engineering (เหมือนกับใน train_script ทุกประการ)
    df['estimated_credit_limit'] = df['revol_bal'] / (df['revol_util'] + 0.001)
    df['annual_inc'] = np.exp(df['log_annual_inc'])
    df['installment_to_income_ratio'] = (df['installment'] * 12) / df['annual_inc']
    df['high_interest'] = (df['int_rate'] > df['int_rate'].median()).astype(int)

    dummies = pd.get_dummies(df['purpose'], prefix='purpose', drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    
    # เก็บ purpose เดิมไว้เพื่อใช้ในตาราง
    original_purpose = df['purpose'].copy()
    
    df.drop(['annual_inc'], axis=1, inplace=True)
    df['purpose'] = original_purpose # นำคอลัมน์ purpose ที่เป็น string กลับมา
    
    print(f"เตรียมข้อมูลสำเร็จ จำนวน {len(df)} แถว")
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
    exit()

# --- 3. Backfill - ทำนายผลข้อมูลทั้งหมด ---
print("\nStep 3: กำลังทำนายผล (Backfill) ข้อมูลทั้งหมดด้วยโมเดลที่โหลดมา...")

# เตรียม Features (X) ให้ตรงกับที่โมเดลคาดหวัง
model_features = model.steps[-1][1].feature_names_in_
X = df.drop(columns=['not_fully_paid', 'purpose']).reindex(columns=model_features)

predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]

# เพิ่มคอลัมน์ใหม่เข้าไปใน DataFrame หลัก
df['model_prediction'] = predictions
df['model_probability'] = probabilities
print("Backfill ข้อมูลสำเร็จ!")


# --- 4. สร้างตารางและนำเข้าข้อมูลทั้งหมดลง Database ---
print("\nStep 4: กำลังเชื่อมต่อและนำเข้าข้อมูลลง PostgreSQL...")
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

try:
    # ลบคอลัมน์ dummy ของ purpose ออกก่อน เพราะเรามีคอลัมน์ purpose หลักอยู่แล้ว
    df.drop(columns=dummies.columns, inplace=True)

    df.to_sql('loan_data', engine, if_exists='append', index=False)
    print("✅ นำเข้าข้อมูลเริ่มต้นพร้อมผลทำนายลงตาราง 'loan_data' สำเร็จ!")
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการเชื่อมต่อหรือนำเข้าข้อมูล: {e}")

print("\n--- [DATABASE SEEDING] กระบวนการเสร็จสิ้น ---")