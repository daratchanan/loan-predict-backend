# app/main.py
import os
from fastapi import FastAPI, Depends,Query, BackgroundTasks, HTTPException, status
from typing import List, Optional
import math
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import func, cast, Float
from datetime import date, datetime, timedelta

from .models import LoanApplicationRequest, PredictionResponse, PerformanceResponse, DashboardResponse, BreakdownItem
from .ml.predict import get_prediction, model 
from .database import engine, get_db, SessionLocal
from . import db_models
from .ml.retrain import retrain_model_task

from . import auth, db_models, schemas
from .database import engine, get_db
from .auth import get_current_active_user, get_user 
from . import db_models

from fastapi.middleware.cors import CORSMiddleware

from app import models

def create_initial_data():
    """
    ฟังก์ชันสำหรับสร้างข้อมูลเริ่มต้น (Roles และ Users)
    จะถูกเรียกใช้งานตอน Startup
    """
    # สร้าง DB Session เฉพาะกิจสำหรับ Startup
    db = SessionLocal()
    try:
        print("--- Checking for initial data ---")

        # --- 1. ตรวจสอบและสร้าง Roles ---
        initial_roles = [
            {"name": "admin", "description": "Administrator with all permissions"},
            {"name": "loan_officer", "description": "Loan officer who can create applications"}
        ]
        
        for role_data in initial_roles:
            role = db.query(db_models.Role).filter(db_models.Role.name == role_data["name"]).first()
            if not role:
                db.add(db_models.Role(**role_data))
                print(f"Role '{role_data['name']}' created.")
        
        db.commit() # Commit เพื่อให้ Roles พร้อมใช้งาน

        # --- 2. ตรวจสอบและสร้าง Admin User ---
        admin_username = os.getenv("ADMIN_USERNAME")
        admin_password = os.getenv("ADMIN_PASSWORD")

        if admin_username and admin_password:
            admin_user = db.query(db_models.User).filter(db_models.User.username == admin_username).first()
            if not admin_user:
                hashed_password = auth.get_password_hash(admin_password)
                admin_role = db.query(db_models.Role).filter(db_models.Role.name == "admin").first()
                
                new_admin = db_models.User(
                    username=admin_username, 
                    hashed_password=hashed_password,
                    is_active=True
                )
                if admin_role:
                    new_admin.roles.append(admin_role)
                
                db.add(new_admin)
                print(f"Admin user '{admin_username}' created.")
        
        # --- 3. ตรวจสอบและสร้าง Loan Officer User ---
        officer_username = os.getenv("OFFICER_USERNAME")
        officer_password = os.getenv("OFFICER_PASSWORD")

        if officer_username and officer_password:
            officer_user = db.query(db_models.User).filter(db_models.User.username == officer_username).first()
            if not officer_user:
                hashed_password = auth.get_password_hash(officer_password)
                officer_role = db.query(db_models.Role).filter(db_models.Role.name == "loan_officer").first()

                new_officer = db_models.User(
                    username=officer_username,
                    hashed_password=hashed_password,
                    is_active=True
                )
                if officer_role:
                    new_officer.roles.append(officer_role)

                db.add(new_officer)
                print(f"Loan Officer user '{officer_username}' created.")

        db.commit()
        print("--- Initial data check complete ---")

    finally:
        db.close()

def role_checker(required_roles: List[str]):
    """
    นี่คือ Factory ที่สร้าง Dependency สำหรับตรวจสอบ Role
    """
    def check_user_role(current_user: db_models.User = Depends(get_current_active_user)):
        user_roles = {role.name for role in current_user.roles}
        
        # ตรวจสอบว่า user มี role ที่ต้องการอย่างน้อยหนึ่ง role หรือไม่
        if not any(role in user_roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to perform this action"
            )
        return current_user
    return check_user_role

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

# สร้าง Dependency สำหรับแต่ละ Role ที่ต้องการ
admin_only = role_checker(["admin"])
loan_officer_or_admin = role_checker(["loan_officer", "admin"])

@app.on_event("startup")
async def startup_event():
    """
    Event ที่จะทำงานตอนเริ่มรัน Application
    """
    # สร้างตารางทั้งหมดในฐานข้อมูล (ถ้ายังไม่มี)
    db_models.Base.metadata.create_all(bind=engine)
    # เรียกใช้ฟังก์ชันสร้างข้อมูลเริ่มต้น
    create_initial_data()

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Loan Approval API"}

# --- เพิ่ม Endpoint สำหรับ Authentication ---

@app.post("/token", response_model=schemas.Token, tags=["Authentication"])
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    db: Session = Depends(get_db)
):
    """
    รับ username และ password จากฟอร์ม, ตรวจสอบ, และคืน JWT Token
    """
    user = get_user(db, form_data.username)
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 1. ดึงรายชื่อ Role ของ User ออกมาเป็น List ของ string
    user_roles = [role.name for role in user.roles]
    
    # 2. สร้าง data dictionary ที่มีทั้ง sub (username) และ roles
    token_data = {"sub": user.username, "roles": user_roles}

    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)

    access_token = auth.create_access_token(
        data=token_data, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me", response_model=schemas.User, tags=["Users"])
def read_users_me(current_user: schemas.User = Depends(auth.get_current_active_user)):
    """
    Endpoint สำหรับทดสอบว่า Token ใช้งานได้หรือไม่
    จะคืนข้อมูลของ User ที่ Login อยู่
    """
    return current_user

# (แนะนำ) Endpoint สำหรับสร้าง User ใหม่
@app.post("/users/", response_model=schemas.User, tags=["Users"])
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = auth.get_user(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = auth.get_password_hash(user.password)
    new_user = db_models.User(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/applications", response_model=PredictionResponse, tags=["Applications"])
def create_application_and_predict(
    application_data: LoanApplicationRequest, 
    db: Session = Depends(get_db),
    current_user: db_models.User = Depends(loan_officer_or_admin)
):
    # 1. get_prediction จะคืนค่า df_with_engineered_features มาด้วย
    prediction_result, df_with_engineered_features, original_purpose = get_prediction(application_data, db)
    
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

# app/main.py (ส่วนเพิ่มเติม)

@app.get("/applications/{application_id}", response_model=models.RecentApplication, tags=["Applications"])
def get_application_by_id(
    application_id: int, 
    db: Session = Depends(get_db),
    current_user: db_models.User = Depends(get_current_active_user)
):
    """
    ดึงข้อมูลใบสมัครและผลการทำนายสำหรับ ID ที่ระบุ
    """
    application = db.query(db_models.LoanData).filter(db_models.LoanData.id == application_id).first()
    
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    # คืนค่าเป็น RecentApplication ซึ่งมีโครงสร้างผลการทำนายครบถ้วน
    return application

@app.get("/dashboard", response_model=DashboardResponse, tags=["Dashboard"])
def get_dashboard_data(
    db: Session = Depends(get_db),
    current_user: db_models.User = Depends(get_current_active_user),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    prediction_filter: Optional[int] = Query(None, description="Filter by prediction (0=ผ่านเกณฑ์, 1=ไม่ผ่านเกณฑ์)"),
    purpose_filter: Optional[str] = Query(None, description="Filter by a specific purpose"),
    sort_by: Optional[str] = Query(None, description="Sort by 'int_rate' or 'model_probability'"),
    sort_order: str = Query("desc", description="Sort order: 'asc' or 'desc'"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1)
):
    """
    ดึงข้อมูลทั้งหมดสำหรับหน้า Dashboard พร้อม Date Filter, Pagination, และ Sort
    """
    # --- 1. จัดการ Date Filter ---
    try:
        if start_date and end_date:
            start_date_obj = date.fromisoformat(start_date)
            # เพิ่มเวลา 23:59:59 ให้ end_date เพื่อให้รวมข้อมูลทั้งวัน
            end_date_obj = datetime.combine(date.fromisoformat(end_date), datetime.max.time())
        else:
            # ถ้าไม่ส่งมา, ใช้ค่า default ย้อนหลัง 30 วัน
            end_date_obj = datetime.now()
            start_date_obj = end_date_obj - timedelta(days=30)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")

    # --- 2. สร้าง Base Query ที่มี Date Filter ---
    # Query ทั้งหมดจะใช้ base_query นี้เพื่อให้ข้อมูลสอดคล้องกัน
    base_query = db.query(db_models.LoanData).filter(
        db_models.LoanData.application_date.between(start_date_obj, end_date_obj)
    )

    # --- 3. KPIs (ใช้ base_query) ---
    total_applications = base_query.count()
    # คำนวณ Approval Rate จากช่วงวันที่ที่เลือกเท่านั้น
    approval_rate_query = base_query.with_entities(
        func.avg(cast(1 - db_models.LoanData.model_prediction, Float))
    ).scalar()
    approval_rate = round((approval_rate_query or 0.0) * 100, 2)

    # --- 4. Recent Applications Table (ใช้ base_query) ---
    query = base_query # เริ่มจาก base query ที่กรองวันที่แล้ว
    
    # Filtering เพิ่มเติม
    if prediction_filter is not None:
        query = query.filter(db_models.LoanData.model_prediction == prediction_filter)
    if purpose_filter:
        query = query.filter(db_models.LoanData.purpose == purpose_filter)
        
    # Sorting
    if sort_by in ["int_rate", "model_probability"]:
        column_to_sort = getattr(db_models.LoanData, sort_by)
        query = query.order_by(column_to_sort.desc() if sort_order == "desc" else column_to_sort.asc())
    else: 
        query = query.order_by(db_models.LoanData.application_date.desc())
        
    # Pagination
    total_items = query.count()
    total_pages = math.ceil(total_items / page_size)
    paginated_apps = query.offset((page - 1) * page_size).limit(page_size).all()
    
    recent_applications_response = {
        "total_items": total_items,
        "total_pages": total_pages,
        "current_page": page,
        "items": paginated_apps
    }

    # --- 5. Breakdowns (ใช้ base_query) ---
    prediction_counts = base_query.with_entities(
        db_models.LoanData.model_prediction,
        func.count(db_models.LoanData.model_prediction)
    ).group_by(db_models.LoanData.model_prediction).all()
    
    prediction_breakdown = [
        BreakdownItem(label="ผ่านเกณฑ์", value=next((c for p, c in prediction_counts if p == 0), 0)),
        BreakdownItem(label="ไม่ผ่านเกณฑ์", value=next((c for p, c in prediction_counts if p == 1), 0))
    ]

    purpose_counts_query = base_query.with_entities(
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
def retrain_model(
    background_tasks: BackgroundTasks,
    current_user: db_models.User = Depends(admin_only)
    ):
    background_tasks.add_task(retrain_model_task) 
    return {"message": "Model retraining process has been started. (ทำงานใน Background)"}

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