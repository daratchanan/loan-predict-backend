# app/ml/predict.py
import joblib
import pandas as pd
import numpy as np
from app.models import LoanApplicationRequest

# โหลด model pipeline
model = joblib.load("app/ml/model_pipeline.joblib")
median_int_rate = 0.1221 # ค่า median ที่เราใช้ตอน train

def get_prediction(data: LoanApplicationRequest):
    # 1. สร้าง DataFrame จากข้อมูลที่รับมา
    df = pd.DataFrame([data.dict()])

    # 2. ทำ Feature Engineering ให้เหมือนกับตอน Train
    df['estimated_credit_limit'] = df['revol_bal'] / (df['revol_util'] + 0.001)
    df['annual_inc'] = np.exp(df['log_annual_inc'])
    df['installment_to_income_ratio'] = (df['installment'] * 12) / df['annual_inc']
    df['high_interest'] = (df['int_rate'] > median_int_rate).astype(int)

    # 3. ทำ One-Hot Encoding สำหรับ 'purpose'
    # สร้างคอลัมน์ dummy ทั้งหมดให้มีค่าเป็น False ก่อน
    df['purpose_debt_consolidation'] = False
    df['purpose_educational'] = False
    df['purpose_small_business'] = False
    
    # เช็คค่า purpose ที่รับมาแล้วกำหนดค่า True ให้ถูกคอลัมน์
    if data.purpose == 'debt_consolidation':
        df['purpose_debt_consolidation'] = True
    elif data.purpose == 'educational':
        df['purpose_educational'] = True
    elif data.purpose == 'small_business':
        df['purpose_small_business'] = True
        
    # 4. จัดเรียงคอลัมน์ให้ตรงกับที่โมเดลคาดหวัง
    # (จำเป็นมาก เพราะ model.predict คาดหวังลำดับคอลัมน์ที่ถูกต้อง)
    model_features = model.named_steps['classifier'].feature_names_in_
    df_final = df[model_features]

    # 5. ทำนายผล
    probability = model.predict_proba(df_final)[0, 1]
    prediction_value = model.predict(df_final)[0]
    prediction_label = "ไม่ผ่านเกณฑ์" if prediction_value == 1 else "ผ่านเกณฑ์"

    prediction_result = {
        "prediction": prediction_label,
        "probability": float(probability),
        "fico_score": data.fico,
        "interest_rate": data.int_rate
    }
    
    # คืนค่า 2 อย่าง: ผลการทำนาย และ ข้อมูลที่แปลงแล้ว
    return prediction_result, df_final ,data.purpose