# app/ml/retrain.py

import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.metrics import classification_report, accuracy_score

from app import db_models

def retrain_model_task():
    """
    ฟังก์ชันสำหรับ Retrain โมเดลทั้งหมด (ทำงานใน Background)
    จะดึงข้อมูลล่าสุด, ประเมินผล, บันทึก Log, Train ใหม่, และ Save ทับไฟล์เดิม
    """
    print("--- [BACKGROUND TASK] เริ่มกระบวนการ Retrain โมเดล ---")
    
    # --- 1. โหลดข้อมูลทั้งหมดจาก PostgreSQL ---
    print("Step 1: กำลังโหลดข้อมูลล่าสุดจาก PostgreSQL...")
    load_dotenv()
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_NAME = os.getenv("DB_NAME")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")

    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(DATABASE_URL)

    try:
        df = pd.read_sql("SELECT * FROM loan_data WHERE not_fully_paid IS NOT NULL", engine)
        print(f"โหลดข้อมูลสำหรับ Train สำเร็จ จำนวน {len(df)} แถว")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
        return

    # --- 2. แยก X (Features) และ y (Target) ---
    X = df.drop(['not_fully_paid', 'id'], axis=1)
    y = df['not_fully_paid']

    # --- 3. แบ่งข้อมูลเป็น Train และ Test Set (สำหรับการประเมินผลชั่วคราว) ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- 4. สร้าง Pipeline ---
    categorical_features = X.select_dtypes(include=np.bool_).columns.tolist()
    categorical_indices = [X.columns.get_loc(col) for col in categorical_features]

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    et = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    meta_xgb = XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_jobs=-1, random_state=42)
    
    stacking_model = StackingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("et", et)],
        final_estimator=meta_xgb,
        cv=5,
        n_jobs=-1
    )
    
    final_pipeline = ImblearnPipeline(steps=[
        ('smotenc', SMOTENC(categorical_features=categorical_indices, random_state=42)),
        ('classifier', stacking_model)
    ])

    # --- 5. ฝึกสอนและประเมินผลโมเดล ---
    print("Step 5: กำลังฝึกสอนโมเดลด้วย Train Set (80%) เพื่อการประเมิน...")
    final_pipeline.fit(X_train, y_train)

    print("Step 6: กำลังประเมินผลโมเดลด้วย Test Set (20%)...")
    y_pred = final_pipeline.predict(X_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    print("ผลการประเมินของโมเดลใหม่:")
    print(classification_report(y_test, y_pred))

    # --- 6. บันทึกผลลัพธ์ลง Database ---
    print("Step 7: กำลังบันทึกผลการประเมินลง Database...")
    Session = sessionmaker(bind=engine)
    db_session = Session()
    try:
        # ดึงโมเดล Stacking ที่ train เสร็จแล้วออกมา
        trained_stacking_model = final_pipeline.named_steps['classifier']
        
        # ดึงโมเดล RandomForest ที่อยู่ข้างในออกมา
        trained_rf_model = trained_stacking_model.named_estimators_['rf']
        
        # ดึงชื่อฟีเจอร์และค่าความสำคัญ
        feature_names = X.columns.tolist()
        importances = trained_rf_model.feature_importances_
        
        # สร้าง Dictionary และแปลงเป็น JSON string
        feature_importance_dict = dict(zip(feature_names, importances))

        # อัปเดตเวอร์ชันเก่าทั้งหมดให้ is_active = False
        db_session.query(db_models.ModelPerformanceLog).update({"is_active": False})

        # เพิ่ม Log ของเวอร์ชันใหม่
        model_version_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_log = db_models.ModelPerformanceLog(
            model_version=model_version_str,
            accuracy=report_dict['accuracy'],
            precision_class_1=report_dict['1']['precision'],
            recall_class_1=report_dict['1']['recall'],
            f1_score_class_1=report_dict['1']['f1-score'],
            feature_importances=feature_importance_dict,
            is_active=True
        )
        db_session.add(new_log)
        db_session.commit()
        print("✅ บันทึกผลลง Database สำเร็จ!")
    except Exception as e:
        db_session.rollback()
        print(f"เกิดข้อผิดพลาดในการบันทึกผลลง Database: {e}")
    finally:
        db_session.close()

    # --- 7. ฝึกสอนโมเดลสุดท้ายด้วยข้อมูลทั้งหมด 100% ---
    print("Step 8: กำลังฝึกสอนโมเดลสุดท้ายด้วยข้อมูลทั้งหมด...")
    final_pipeline.fit(X, y)
    print("การฝึกสอนโมเดลสุดท้ายเสร็จสมบูรณ์!")

    # --- 8. บันทึก Pipeline สุดท้าย ---
    output_path = 'app/ml/model_pipeline.joblib'
    joblib.dump(final_pipeline, output_path)
    
    print(f"--- [BACKGROUND TASK] ✅ Retrain และบันทึกโมเดลใหม่สำเร็จที่: {output_path} ---")