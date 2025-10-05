# train_model.py
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from dotenv import load_dotenv
from app import db_models
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
# แก้ไข: เปลี่ยนจาก SMOTENC เป็น SMOTE เพราะเราจะแปลงข้อมูล Categorical เป็นตัวเลขก่อน
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.metrics import classification_report, accuracy_score

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

# แก้ไข 2: แปลงคอลัมน์ 'purpose' (Categorical) เป็น One-Hot Encoding
df = pd.get_dummies(df, columns=['purpose'], drop_first=True)
print("แปลงคอลัมน์ 'purpose' เป็น One-Hot Encoding เรียบร้อยแล้ว")
print("ตัวอย่างข้อมูลหลังเตรียมข้อมูล:")
print(df.head())


# --- 3. กำหนด Features (X) และ Target (y) ---
print("Step 3: กำลังกำหนด Features และ Target...")
# แก้ไข 3: ลบ 'id' ออกจาก list drop เพราะไม่มีในข้อมูล
X = df.drop('not_fully_paid', axis=1)
y = df['not_fully_paid']

# --- 4. แบ่งข้อมูลเป็น Train และ Test Set ---
print("Step 4: กำลังแบ่งข้อมูลเป็น Train (80%) และ Test (20%) Set...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 5. สร้าง Pipeline ---
print("Step 5: กำลังสร้าง Pipeline...")

# Base Learners
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
svm = SVC(kernel="rbf", probability=True, random_state=42)
et = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=42)
# Meta-Learner
meta_xgb = XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_jobs=-1, random_state=42)
# Stacking Model
stacking_model = StackingClassifier(
    estimators=[("rf", rf), ("svm", svm), ("et", et)],
    final_estimator=meta_xgb,
    cv=5,
    n_jobs=-1
)

# แก้ไข 4: เปลี่ยน Pipeline ให้ใช้ SMOTE แทน SMOTENC
# เนื่องจากข้อมูลทั้งหมดถูกแปลงเป็นตัวเลขแล้ว จึงไม่จำเป็นต้องระบุ categorical_features
final_pipeline = ImblearnPipeline(steps=[
    ('smote', SMOTE(random_state=42)),
    ('classifier', stacking_model)
])

# --- 6. ฝึกสอนและประเมินผลโมเดล ---
print("\n--- Phase 1: การประเมินผลโมเดล ---")
print("Step 6: กำลังฝึกสอนโมเดลด้วย Train Set...")
final_pipeline.fit(X_train, y_train)

print("\nStep 7: กำลังประเมินผลโมเดลด้วย Test Set...")
y_pred = final_pipeline.predict(X_test)

print("\nผลการประเมินประสิทธิภาพของโมเดล:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
report_dict = classification_report(y_test, y_pred, output_dict=True)
print("\nกำลังบันทึกผลการประเมินลง Database...")
db_session = None # กำหนดค่าเริ่มต้น
try:
    trained_stacking_model = final_pipeline.named_steps['classifier']
    trained_rf_model = trained_stacking_model.named_estimators_['rf']
    
    feature_names = X.columns.tolist()
    importances = trained_rf_model.feature_importances_
    
    feature_importance_dict = dict(zip(feature_names, importances))

    Session = sessionmaker(bind=engine)
    db_session = Session()

    db_session.query(db_models.ModelPerformanceLog).update({"is_active": False})

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
    print(f"เกิดข้อผิดพลาดในการบันทึกผลลง Database: {e}")
finally:
    if db_session:
        db_session.close()

# --- 7. ฝึกสอนโมเดลสุดท้ายเพื่อใช้งานจริง ---
print("\n--- Phase 2: การฝึกสอนโมเดลสุดท้ายเพื่อ Deployment ---")
print("Step 8: กำลังฝึกสอนโมเดลอีกครั้งด้วยข้อมูลทั้งหมด (Train + Test)...")
final_pipeline.fit(X, y)
print("การฝึกสอนโมเดลสุดท้ายเสร็จสมบูรณ์!")

# --- 8. บันทึก Pipeline สุดท้าย ---
print("\nStep 9: กำลังบันทึก Pipeline เป็นไฟล์ joblib...")
output_dir = 'app/ml'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

joblib.dump(final_pipeline, f'{output_dir}/model_pipeline.joblib')

print(f"✅ โมเดลถูกบันทึกเรียบร้อยแล้วที่: {output_dir}/model_pipeline.joblib")
print("--- กระบวนการเสร็จสิ้น ---")