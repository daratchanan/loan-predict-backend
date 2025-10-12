# app/ml/predict.py
import joblib
import pandas as pd
import numpy as np
from app.models import LoanApplicationRequest
import lime
import lime.lime_tabular
from sqlalchemy.orm import Session
from app import db_models


# --- ส่วนที่ 1: Setup โมเดลและ LIME Explainer (ทำงานครั้งเดียวตอนเริ่ม API) ---

# 1.1 โหลดโมเดลที่ฝึกสอนเสร็จแล้ว
model = joblib.load("app/ml/model_pipeline.joblib")

# 1.2 โหลดข้อมูล Train set ที่ Clean แล้ว (จำเป็นสำหรับ LIME)
try:
    X_train_clean = pd.read_csv('data/X_train_cleaned.csv')
    feature_names = X_train_clean.columns.tolist()
except FileNotFoundError:
    print("FATAL ERROR: data/X_train_cleaned.csv not found. LIME explainer cannot be created.")
    # ในกรณีที่ไม่มีไฟล์นี้ ระบบจะไม่สามารถสร้าง explainer ได้
    # ควรทำให้แอปฯ หยุดทำงาน หรือมี fallback ที่เหมาะสม
    raise

# 1.3 สร้าง LIME Explainer object
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_clean.values,
    feature_names=feature_names,
    class_names=['ผ่านเกณฑ์', 'ไม่ผ่านเกณฑ์'],
    mode='classification'
)


# --- ส่วนที่ 2: ฟังก์ชันสำหรับทำนายผล (ทำงานทุกครั้งที่มี Request) ---

def get_prediction(data: LoanApplicationRequest, db: Session):
    """
    ทำนายผลสินเชื่อจากข้อมูลใบสมัคร โดยมีขั้นตอนเตรียมข้อมูลที่สมบูรณ์
    """
    
    # --- 1. เตรียมข้อมูล (Feature Engineering) ---
    df = pd.DataFrame([data.dict()])
    df['estimated_credit_limit'] = df['revol_bal'] / (df['revol_util'] + 0.001)
    df['annual_inc'] = np.exp(df['log_annual_inc'])
    df['installment_to_income_ratio'] = (df['installment'] * 12) / df['annual_inc']
    df.drop('annual_inc', axis=1, inplace=True)
    median_int_rate = 0.1221 
    df['high_interest'] = (df['int_rate'] > median_int_rate).astype(int)

    # --- 2. สร้าง DataFrame ให้มีโครงสร้างตรงกับที่โมเดลคาดหวัง ---
    df_processed = pd.DataFrame(columns=feature_names, index=df.index)
    df_processed.update(df)

    if f'purpose_{data.purpose}' in df_processed.columns:
        df_processed[f'purpose_{data.purpose}'] = 1

    df_processed.fillna(0, inplace=True)
    df_final = df_processed[feature_names]

    # --- 3. ดึง OPTIMAL THRESHOLD จาก DATABASE ---
    active_model_log = db.query(db_models.ModelPerformanceLog)\
                         .filter(db_models.ModelPerformanceLog.is_active == True)\
                         .first()
    # ถ้าหาไม่เจอ หรือค่าเป็น None, ให้ใช้ค่า Default ที่ปลอดภัย
    best_threshold = float(active_model_log.optimal_threshold) if active_model_log and active_model_log.optimal_threshold is not None else 0.5
    print(f"Using Optimal Threshold from DB: {best_threshold}")

    # --- 4. ทำนายผล (ใช้ Threshold ที่ดึงมา) ---
    probability = model.predict_proba(df_final)[0, 1]
    prediction_value = 1 if probability >= best_threshold else 0
    prediction_label = "ไม่ผ่านเกณฑ์" if prediction_value == 1 else "ผ่านเกณฑ์"

    # =========================================================================
    # =====> เพิ่มโค้ดส่วนนี้เข้าไป <=====

    # 4. สร้างฟังก์ชันตัวกลาง (Wrapper) สำหรับ LIME
    def predict_fn_for_lime(numpy_array):
        # แปลง NumPy array กลับเป็น DataFrame พร้อมชื่อคอลัมน์ที่ถูกต้อง
        data_as_df = pd.DataFrame(numpy_array, columns=feature_names)
        # ส่ง DataFrame นี้ไปให้โมเดลทำนาย
        return model.predict_proba(data_as_df)

    # =========================================================================

    # --- 5. คำนวณ LIME Explanation ---
    explanation_lime = explainer.explain_instance(
        data_row=df_final.iloc[0].values,
        # --- เปลี่ยนไปใช้ฟังก์ชันตัวกลางที่เราสร้างขึ้น ---
        predict_fn=predict_fn_for_lime,
        num_features=5 
    )
    explanation_dict = dict(explanation_lime.as_list())

    # --- 6. รวบรวมผลลัพธ์ ---
    prediction_result = {
        "prediction": prediction_label,
        "probability": float(probability),
        "fico_score": data.fico,
        "interest_rate": data.int_rate,
        "explanation": explanation_dict
    }
    
    db_record_data = df_final.reset_index(drop=True)

    return prediction_result, db_record_data, data.purpose