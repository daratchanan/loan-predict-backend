# app/main.py
from fastapi import FastAPI, Depends,Query, BackgroundTasks, HTTPException
from typing import Optional
import math
from sqlalchemy.orm import Session
from sqlalchemy import func, cast, Float
from datetime import datetime, timedelta
import pandas as pd

from .models import LoanApplicationRequest, PredictionResponse, PerformanceResponse, DashboardResponse, RecentApplication, BreakdownItem
from .ml.predict import get_prediction, model # Import model มาด้วย
from .database import engine, get_db
from . import db_models
from .ml.retrain import retrain_model_task

from fastapi.middleware.cors import CORSMiddleware

db_models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Loan Approval Prediction API",
    description="API สำหรับสนับสนุนการตัดสินใจอนุมัติสินเชื่อ",
    version="1.0.0"
)

origins = [
    "http://localhost:3000", # URL ของ Next.js App
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Loan Approval API"}

@app.post("/applications", response_model=PredictionResponse, tags=["Applications"])
def create_application_and_predict(
    application_data: LoanApplicationRequest, 
    db: Session = Depends(get_db)
):
    # 1. get_prediction จะคืนค่า df_with_engineered_features มาด้วย
    prediction_result, df_with_engineered_features, original_purpose = get_prediction(application_data)
    
    # 2. เริ่มต้นสร้าง Dictionary สำหรับบันทึกลง DB จากข้อมูลดิบที่รับเข้ามา
    db_dict = application_data.dict()
    
    # 3. ดึงค่าฟีเจอร์ที่สร้างขึ้นใหม่ (Engineered Features) จาก DataFrame ที่ predict.py ส่งมา
    engineered_features = df_with_engineered_features[[
        'estimated_credit_limit', 
        'installment_to_income_ratio', 
        'high_interest'
    ]].to_dict('records')[0]

    # 4. รวมข้อมูลทั้งหมดเข้าด้วยกัน
    db_dict.update(engineered_features) # เพิ่ม engineered features
    
    # เพิ่มผลลัพธ์จากโมเดล
    prediction_value = 1 if prediction_result["prediction"] == "ไม่ผ่านเกณฑ์" else 0
    db_dict['model_prediction'] = prediction_value
    db_dict['model_probability'] = prediction_result["probability"]
    db_dict['lime_explanation'] = prediction_result["explanation"]
    
    # สร้าง ORM object จาก Dictionary ที่มีโครงสร้างถูกต้องแล้ว
    db_record = db_models.LoanData(**db_dict)
    
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    
    return PredictionResponse(**prediction_result)

@app.get("/dashboard", response_model=DashboardResponse, tags=["Dashboard"])
def get_dashboard_data(
    db: Session = Depends(get_db),
    prediction_filter: Optional[int] = Query(None, description="Filter by prediction (0=ผ่านเกณฑ์, 1=ไม่ผ่านเกณฑ์)"),
    purpose_filter: Optional[str] = Query(None, description="Filter by a specific purpose (e.g., 'debt_consolidation')"), # <--- 1. เพิ่มบรรทัดนี้
    sort_by: Optional[str] = Query(None, description="Sort by 'int_rate' or 'model_probability'"),
    sort_order: str = Query("desc", description="Sort order: 'asc' or 'desc'"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1)
):
    """
    ดึงข้อมูลทั้งหมดสำหรับหน้า Dashboard พร้อม Pagination, Filter, และ Sort
    """
    # --- 1 & 2. KPIs (เหมือนเดิม) ---
    total_applications = db.query(db_models.LoanData).count()
    thirty_days_ago = datetime.now() - timedelta(days=30)
    approval_rate_query = db.query(func.avg(cast(db_models.LoanData.credit_policy, Float))).filter(db_models.LoanData.application_date >= thirty_days_ago).scalar()
    approval_rate = round((approval_rate_query or 0.0) * 100, 2)

    # --- 3. Recent Applications Table (with dynamic query) ---
    query = db.query(db_models.LoanData)
    
    # Filtering
    if prediction_filter is not None:
        query = query.filter(db_models.LoanData.model_prediction == prediction_filter)
    
    if purpose_filter: # <--- 2. เพิ่มเงื่อนไข 2 บรรทัดนี้
        query = query.filter(db_models.LoanData.purpose == purpose_filter)
        
    # Sorting
    if sort_by in ["int_rate", "model_probability"]:
        column_to_sort = getattr(db_models.LoanData, sort_by)
        if sort_order == "desc":
            query = query.order_by(column_to_sort.desc())
        else:
            query = query.order_by(column_to_sort.asc())
    else: # Default sort
        query = query.order_by(db_models.LoanData.application_date.desc())
        
    # Pagination
    total_items = query.count() # <-- .count() จะนับรายการที่ผ่านการ filter แล้ว
    total_pages = math.ceil(total_items / page_size)
    paginated_apps = query.offset((page - 1) * page_size).limit(page_size).all()
    
    recent_applications_response = {
        "total_items": total_items,
        "total_pages": total_pages,
        "current_page": page,
        "items": paginated_apps
    }

    # --- 4 & 5. Breakdowns (เหมือนเดิม) ---
    prediction_counts = db.query(
        db_models.LoanData.model_prediction,
        func.count(db_models.LoanData.model_prediction)
    ).group_by(db_models.LoanData.model_prediction).all()
    
    prediction_breakdown = [
        BreakdownItem(label="ผ่านเกณฑ์", value=next((c for p, c in prediction_counts if p == 0), 0)),
        BreakdownItem(label="ไม่ผ่านเกณฑ์", value=next((c for p, c in prediction_counts if p == 1), 0))
    ]

    purpose_counts_query = db.query(
        db_models.LoanData.purpose,
        func.count(db_models.LoanData.purpose).label('count')
    ).group_by(db_models.LoanData.purpose).order_by(func.count(db_models.LoanData.purpose).desc()).all()
    
    purpose_breakdown = [BreakdownItem(label=row.purpose, value=row.count) for row in purpose_counts_query]

    return DashboardResponse(
        total_applications=total_applications,
        approval_rate_30_days=approval_rate,
        recent_applications=recent_applications_response,
        prediction_breakdown=prediction_breakdown,
        purpose_breakdown=purpose_breakdown
    )

@app.post("/retrain", tags=["Training"])
def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_model_task) # <--- ตรวจสอบให้แน่ใจว่าชื่อตรงนี้ถูกต้องด้วย
    return {"message": "กระบวนการ Retrain โมเดลได้เริ่มขึ้นแล้ว (ทำงานใน Background)"}

@app.get("/model-performance", response_model=PerformanceResponse, tags=["Performance"])
def get_active_model_performance(db: Session = Depends(get_db)):
    """
    ดึงข้อมูลประสิทธิภาพของโมเดลที่กำลังใช้งานอยู่ (is_active = True)
    """
    performance_log = db.query(db_models.ModelPerformanceLog)\
                        .filter(db_models.ModelPerformanceLog.is_active == True)\
                        .first()
    
    if not performance_log:
        raise HTTPException(status_code=404, detail="ไม่พบข้อมูลประสิทธิภาพของโมเดลที่ใช้งานอยู่")

    return performance_log