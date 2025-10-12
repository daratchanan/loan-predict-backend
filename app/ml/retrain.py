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
from sklearn.compose import ColumnTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score

from app import db_models # Import จาก __init__.py ของ app

def retrain_model_task():
    """
    ฟังก์ชันสำหรับ Retrain โมเดลทั้งหมด (ทำงานใน Background)
    จะดึงข้อมูลล่าสุด, เตรียมข้อมูล, สร้าง Pipeline ที่สมบูรณ์,
    ประเมินผล, บันทึก Log, Train ใหม่ด้วยข้อมูลทั้งหมด, และ Save ทับไฟล์เดิม
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
        # ดึงข้อมูลเฉพาะส่วนที่จำเป็นและมี target (not_fully_paid)
        df = pd.read_sql("SELECT * FROM loan_data WHERE not_fully_paid IS NOT NULL", engine)
        print(f"โหลดข้อมูลสำหรับ Train สำเร็จ จำนวน {len(df)} แถว")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
        return

    # --- 2. เตรียมข้อมูล (Feature Engineering) ---
    print("Step 2: กำลังเตรียมข้อมูลและสร้าง Features (Feature Engineering)...")
    df_clean = df.copy()

    # สร้าง Feature เหมือนกับใน train_model.py
    df_clean['estimated_credit_limit'] = df_clean['revol_bal'] / (df_clean['revol_util'] + 0.001)
    df_clean['annual_inc'] = np.exp(df_clean['log_annual_inc'])
    df_clean['installment_to_income_ratio'] = (df_clean['installment'] * 12) / df_clean['annual_inc']
    df_clean.drop('annual_inc', axis=1, inplace=True)
    
    median_int_rate = df_clean['int_rate'].median()
    df_clean['high_interest'] = (df_clean['int_rate'] > median_int_rate).astype(int)
    
    # แปลง purpose เป็น One-Hot Encoding
    df_clean = pd.get_dummies(df_clean, columns=['purpose'], drop_first=True)


    # --- 3. แยก X (Features) และ y (Target) ---
    print("Step 3: กำลังกำหนด Features (X) และ Target (y)...")
    # ไม่เอาคอลัมน์ที่ไม่ใช่ feature ออก
    features_to_drop = ['not_fully_paid', 'id', 'application_date', 'model_prediction', 'model_probability', 'lime_explanation']
    X = df_clean.drop(columns=features_to_drop, errors='ignore')
    y = df_clean['not_fully_paid']

    # --- 4. แบ่งข้อมูลเป็น Train และ Test Set (สำหรับการประเมินผล) ---
    print("Step 4: กำลังแบ่งข้อมูลเป็น Train (80%) และ Test (20%) Set...")
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    
    # บันทึก X_train ที่ Clean แล้วสำหรับ LIME
    # ต้องแน่ใจว่า X_train มีคอลัมน์เหมือนกับที่โมเดลคาดหวัง
    X_train.to_csv('data/X_train_cleaned.csv', index=False)
    print("✅ บันทึกไฟล์ data/X_train_cleaned.csv สำหรับ LIME เรียบร้อยแล้ว")


    # --- 5. สร้าง Pipeline ที่สมบูรณ์ ---
    print("Step 5: กำลังสร้าง Preprocessing และ Modeling Pipeline...")
    
    # 5.1 ระบุประเภทคอลัมน์
    # คอลัมน์ bool มาจาก get_dummies
    bool_cols = [c for c in X.columns if X[c].dtype == "bool"]
    # คอลัมน์ตัวเลขคือคอลัมน์ที่เหลือ
    num_cols = [col for col in X.columns if col not in bool_cols]

    # 5.2 สร้าง Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), num_cols)],
        remainder='passthrough' # ปล่อยคอลัมน์ bool ผ่านไปเลย
    )

    # 5.3 กำหนด Base Models และ Meta Model (เหมือน train_model.py)
    rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=random_state)
    svm = SVC(kernel="rbf", probability=True, C=2.0, gamma="scale", random_state=random_state)
    et = ExtraTreesClassifier(n_estimators=600, n_jobs=-1, random_state=random_state)
    meta_xgb_wrapped = OneVsRestClassifier(
        XGBClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, objective="binary:logistic", eval_metric="logloss",
            n_jobs=-1, random_state=random_state
        )
    )
    
    # 5.4 สร้าง StackingClassifier
    stacking_model = StackingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("et", et)],
        final_estimator=meta_xgb_wrapped,
        cv=5,
        n_jobs=-1
    )
    
    # 5.5 รวมทุกอย่างใน Final Pipeline
    # ไม่จำเป็นต้องใช้ SMOTENC ใน Pipeline เพราะเราจะเทรนบนข้อมูลทั้งหมดตอนท้าย
    final_pipeline = Pipeline(steps=[
        ('prep', preprocessor),
        ('classifier', stacking_model)
    ])

    # --- 6. ฝึกสอนและประเมินผลโมเดลชั่วคราว ---
    print("Step 6: กำลังฝึกสอนโมเดลด้วย Train Set (80%) เพื่อการประเมิน...")
    final_pipeline.fit(X_train, y_train)

    print("Step 7: กำลังประเมินผลโมเดลด้วย Test Set (20%)...")
    proba_test = final_pipeline.predict_proba(X_test)[:, 1]
    
    # หาค่า Threshold ที่ดีที่สุดจาก F1-Score
    thresholds = np.linspace(0.00, 1.00, 101)
    f1_scores = [f1_score(y_test, (proba_test >= t).astype(int), zero_division=0) for t in thresholds]
    best_f1_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_index]
    y_pred_best = (proba_test >= best_threshold).astype(int)
    
    report_dict = classification_report(y_test, y_pred_best, output_dict=True)
    print("="*60)
    print(f"ผลการประเมินของโมเดลใหม่ (Optimal Threshold: {best_threshold:.2f}):")
    print(classification_report(y_test, y_pred_best))
    print("="*60)

    # --- 7. บันทึกผลลัพธ์ลง Database ---
    print("Step 8: กำลังบันทึกผลการประเมินลง Database...")
    Session = sessionmaker(bind=engine)
    db_session = Session()
    try:
        # อัปเดตเวอร์ชันเก่าทั้งหมดให้ is_active = False
        db_session.query(db_models.ModelPerformanceLog).update({"is_active": False})

        # คำนวณ Meta-Model Importances
        trained_stacking_model = final_pipeline.named_steps['classifier']
        meta_model = trained_stacking_model.final_estimator_.estimators_[0]
        base_model_names = [name for name, _ in trained_stacking_model.estimators]
        meta_importances = meta_model.feature_importances_
        meta_importance_dict = {name: float(imp) for name, imp in zip(base_model_names, meta_importances)}
        
        # เพิ่ม Log ของเวอร์ชันใหม่
        model_version_str = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        class_1_metrics = report_dict.get('1', {})
        new_log = db_models.ModelPerformanceLog(
            model_version=model_version_str,
            accuracy=report_dict['accuracy'],
            precision_class_1=class_1_metrics.get('precision'),
            recall_class_1=class_1_metrics.get('recall'),
            f1_score_class_1=class_1_metrics.get('f1-score'),
            meta_model_importances=meta_importance_dict, # แก้ไข: ใช้ meta importances
            optimal_threshold=best_threshold,      # เพิ่ม: บันทึก threshold
            is_active=True
        )
        db_session.add(new_log)
        db_session.commit()
        print("✅ บันทึกผลลง Database สำเร็จ!")
        print(f"   - Meta-Model Importances: {meta_importance_dict}")
    except Exception as e:
        db_session.rollback()
        print(f"เกิดข้อผิดพลาดในการบันทึกผลลง Database: {e}")
    finally:
        db_session.close()

    # --- 8. ฝึกสอนโมเดลสุดท้ายด้วยข้อมูลทั้งหมด 100% ---
    print("Step 9: กำลังฝึกสอนโมเดลสุดท้ายด้วยข้อมูลทั้งหมด (100%)...")
    final_pipeline.fit(X, y)
    print("การฝึกสอนโมเดลสุดท้ายเสร็จสมบูรณ์!")

    # --- 9. บันทึก Pipeline สุดท้าย ---
    output_path = 'app/ml/model_pipeline.joblib'
    joblib.dump(final_pipeline, output_path)
    
    print(f"--- [BACKGROUND TASK] ✅ Retrain และบันทึกโมเดลใหม่สำเร็จที่: {output_path} ---")