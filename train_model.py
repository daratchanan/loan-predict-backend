# train_model.py
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from dotenv import load_dotenv
# from app import db_models
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Numeric, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                            roc_curve, RocCurveDisplay, roc_auc_score, precision_recall_curve, PrecisionRecallDisplay,
                            precision_score, recall_score, f1_score)

print("--- เริ่มกระบวนการสร้าง, ประเมิน, และฝึกสอนโมเดลสุดท้าย ---")

# --- 1. โหลดข้อมูล ---
print("Step 1: กำลังโหลดข้อมูล...")
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

try:
    df = pd.read_csv("data/loan_data.csv")
    print(f"โหลดข้อมูลสำเร็จ จำนวน {len(df)} แถว")
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
    exit()

# --- 2. เตรียมข้อมูล (Data Preparation) ---
print("Step 2: กำลังเตรียมข้อมูล...")

# แก้ไข 1: แก้ไขชื่อคอลัมน์ เปลี่ยนจาก '.' เป็น '_'
df.columns = df.columns.str.replace('.', '_', regex=False)

df_clean = df.copy()

print('Create Financial Ratios สร้างอัตราส่วนทางการเงิน 3 คอลัมน์')
print('1. วงเงินสินเชื่อโดยประมาณ estimated_credit_limit')

if 'revol_bal' in df_clean.columns and 'revol_util' in df_clean.columns:
    df_clean['estimated_credit_limit'] = df_clean['revol_bal'] / (df_clean['revol_util'] + 0.001)

print('2. อัตราส่วนงวดต่อรายได้')
# ต้องแปลงค่า log_annual_inc กลับมาเป็น "รายได้ต่อปีจริงๆ" ก่อน เพื่อให้ได้อัตราส่วนที่ตีความได้ในเชิงเศรษฐศาสตร์ (ค่า log คือ แปลง scale แล้ว)
df_clean['annual_inc'] = np.exp(df_clean['log_annual_inc'])

# สร้างอัตราส่วนระหว่าง "ค่างวดต่อปี" กับ "รายได้ต่อปี"
if 'installment' in df_clean.columns and 'annual_inc' in df_clean.columns:
    df_clean['installment_to_income_ratio'] = (df_clean['installment'] * 12) / df_clean['annual_inc']

df_clean.drop('annual_inc', axis=1, inplace=True)

print('3. อัตราดอกเบี้ยสูงหรือไม่')

if 'int_rate' in df_clean.columns:
    df_clean['high_interest'] = (df_clean['int_rate'] > df_clean['int_rate'].median()).astype(int)


print('4. แปลงคอลัมน์ purpose (Categorical) เป็น One-Hot Encoding')

df_clean = pd.get_dummies(df_clean, columns=['purpose'], drop_first=True)

print("ตัวอย่างข้อมูลหลังเตรียมข้อมูล:")
print(df_clean.head())


random = 42
target = 'not_fully_paid'

# --- 3. กำหนด Features (X) และ Target (y) ---
print("Step 3: กำลังกำหนด Features และ Target...")

X = df_clean.drop(columns=target)
y = df_clean[target]

# ระบุคอลัมน์ "เชิงหมวดหมู่" สำหรับ SMOTENC
cat_cols = [c for c in X.columns if X[c].dtype == "bool"]


# --- 4. แบ่งข้อมูลเป็น Train และ Test Set ---
print("Step 4: กำลังแบ่งข้อมูลเป็น Train (80%) และ Test (20%) Set...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random, stratify=y)

# --- 5. Oversampling ด้วย SMOTE ใช้ SMOTENC ---
print("Step 5: Oversampling ด้วย SMOTE ใช้ SMOTENC...")

# 5.1 แปลงคอลัมน์ bool เป็น int 
X_train_int = X_train.assign(**{c: X_train[c].astype(int) for c in cat_cols})

X_test_int = X_test.assign(**{c: X_test[c].astype(int) for c in cat_cols})

# 5.2 หาตำแหน่งของคอลัมน์หมวดหมู่ 
categorical_indices = [X_train_int.columns.get_loc(c) for c in cat_cols]

# 5.3 ปรับ k_neighbors อัตโนมัติ 
minority_count = y_train.value_counts().min()
k_neighbors = 5 if minority_count > 5 else max(1, minority_count - 1)

# 5.4 ทำ SMOTENC และสร้าง DataFrame ใหม่
smote = SMOTENC(
    categorical_features=categorical_indices,
    random_state=random,
    k_neighbors=k_neighbors
)
X_train_sm, y_train_sm = smote.fit_resample(X_train_int, y_train)


# --- 6. Stacking Ensemble and Pipeling ---
print("Step 6: Stacking Ensemble and Pipeling...")

# num_cols คือคอลัมน์ที่เหลือทั้งหมดหลังจากหักคอลัมน์หมวดหมู่ออกไป
num_cols = [col for col in X_train_sm.columns if col not in cat_cols]

# 6.1 สร้าง Preprocessor
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), num_cols)],
    remainder='passthrough'  # ปล่อยคอลัมน์ bool ที่ทำ get_dummies แล้วผ่านไปเลย
)

# 6.2 กำหนด Base Models และ Meta Model ---
# Base Models
rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=random)
svm = SVC(kernel="rbf", probability=True, C=2.0, gamma="scale", random_state=random)
et = ExtraTreesClassifier(n_estimators=600, n_jobs=-1, random_state=random)

# Meta Model (XGBoost) 
meta_xgb_wrapped = OneVsRestClassifier(
    XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=random
    )
)

# 6.3 สร้าง StackingClassifier โดยใช้ Final Estimator ที่ถูกห่อหุ้มแล้ว 

stack_xgbmeta = StackingClassifier(
    estimators=[
        ("rf", rf),
        ("svm", svm),
        ("et", et),
    ],
    final_estimator=meta_xgb_wrapped, # <-- ใช้ตัวที่ถูกห่อแล้ว
    stack_method="auto",
    passthrough=False,
    cv=5,
    n_jobs=1
)

# 6.4 รวมทุกอย่างใน Pipeline 
model_pipeline = Pipeline(steps=[
    ("prep", preprocessor),
    ("stack", stack_xgbmeta),
])

# --- 7. เริ่ม Train โมเดล ---
print("🚀 กำลังเริ่ม Train Stacking Ensemble Model...")

model_pipeline.fit(X_train_sm, y_train_sm)
print("✅ Train โมเดลสำเร็จ!")

# --- 8. กำลังประเมินผลโมเดลด้วย Test Set ---
print("Step 8: กำลังประเมินผลโมเดลด้วย Test Set...")

# 8.1 ทำนายความน่าจะเป็น (Probability) บน Test Set ---
proba_test = model_pipeline.predict_proba(X_test_int)[:, 1]

# 8.2 หาค่า Threshold ที่ดีที่สุดโดยอิงจาก F1-Score 
thresholds = np.linspace(0.00, 1.00, 101)
f1_scores = [f1_score(y_test, (proba_test >= t).astype(int), zero_division=0) for t in thresholds]

# หาค่า F1 ที่สูงสุดและ Threshold ณ จุดนั้น
best_f1_index = np.argmax(f1_scores)
best_threshold = thresholds[best_f1_index]
best_f1 = f1_scores[best_f1_index]

# 8.3 สร้าง Prediction สุดท้ายโดยใช้ Threshold ที่ดีที่สุด 
y_pred_best = (proba_test >= best_threshold).astype(int)

# 8.4 คำนวณค่าต่างๆ และแสดงผลตามรูปแบบที่ต้องการ ---
accuracy = accuracy_score(y_test, y_pred_best)
cm = confusion_matrix(y_test, y_pred_best)
class_report = classification_report(y_test, y_pred_best)

print("="*60)
print("         ผลการประเมินโมเดล (Optimal Threshold)")
print("="*60)
print(f'threshold: {best_threshold:.2f}')
print(f'Accuracy: {accuracy * 100:.4f}%')
print(cm)
print(class_report)
print("="*60)

# --- 9. บันทึก Model Pipeline ---
print("Step 9: กำลังบันทึก Model Pipeline เป็นไฟล์ joblib...")

output_dir = 'app/ml'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

joblib.dump(model_pipeline, f'{output_dir}/model_pipeline.joblib')

print(f"✅ โมเดลถูกบันทึกเรียบร้อยแล้วที่: {output_dir}/model_pipeline.joblib")


# --- Step 10: บันทึกผลลัพธ์ลง Database ---
print("Step 10: กำลังบันทึกผลการประเมินลงฐานข้อมูล...")

# 10.1 กำหนด ORM Model ให้ตรงกับตาราง model_performance_logs
Base = declarative_base()

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
    feature_importances = Column(JSONB)

# 10.2 สร้าง Session สำหรับคุยกับ Database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

try:
    # 10.3 อัปเดตโมเดลเก่าทั้งหมดให้ is_active = false
    print("   - กำลังอัปเดตสถานะโมเดลเก่า...")
    db.query(ModelPerformanceLog).update({ModelPerformanceLog.is_active: False})
    db.commit()

    # 10.4 เตรียมข้อมูลสำหรับบันทึก
    # สร้าง version ของโมเดลจากวันที่และเวลาปัจจุบัน
    model_version_str = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # ดึงค่า precision, recall, f1-score ของ class 1 (fully_paid) จาก classification_report
    report_dict = classification_report(y_test, y_pred_best, output_dict=True)
    class_1_metrics = report_dict.get('1', {}) # ใช้ .get เพื่อความปลอดภัยหาก class 1 ไม่มีใน report

    # หมายเหตุ: feature_importances สำหรับ Stacking Model มีความซับซ้อนในการคำนวณ
    # และตีความ จึงขอไม่บันทึกในขั้นตอนนี้ (ใส่เป็น None)
    
    # 10.5 สร้าง Log object ใหม่
    new_log = ModelPerformanceLog(
        model_version=model_version_str,
        training_date=datetime.now(),
        accuracy=accuracy,
        precision_class_1=class_1_metrics.get('precision'),
        recall_class_1=class_1_metrics.get('recall'),
        f1_score_class_1=class_1_metrics.get('f1-score'),
        is_active=True,
        feature_importances=None 
    )

    # 10.6 เพิ่มและ commit ข้อมูลใหม่
    db.add(new_log)
    db.commit()
    db.refresh(new_log)
    print(f"   - ✅ บันทึกผลการประเมินสำหรับโมเดลเวอร์ชัน '{model_version_str}' เรียบร้อยแล้ว")

except Exception as e:
    print(f"   - ❌ เกิดข้อผิดพลาดในการบันทึกข้อมูลลง Database: {e}")
    db.rollback()
finally:
    db.close()


print("--- กระบวนการเสร็จสิ้น ---")

