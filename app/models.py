# app/models.py

from pydantic import BaseModel
from datetime import datetime 
from typing import List, Dict, Any, Optional

# --- Input Model for Prediction Endpoint ---

class LoanApplicationRequest(BaseModel):
    credit_policy: int
    purpose: str
    int_rate: float
    installment: float
    log_annual_inc: float
    dti: float
    fico: int
    days_with_cr_line: float
    revol_bal: int
    revol_util: float
    inq_last_6mths: int
    delinq_2yrs: int
    pub_rec: int

# --- Output Model for Prediction Endpoint ---
class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    fico_score: int
    interest_rate: float
    explanation: Dict[str, Any]

# --- Models for Dashboard Endpoint ---
class RecentApplication(BaseModel):
    id: int
    application_date: datetime
    fico: int
    purpose: str
    int_rate: float
    model_prediction: int
    model_probability: float
    lime_explanation: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True

class PaginatedApplicationsResponse(BaseModel):
    total_items: int
    total_pages: int
    current_page: int
    items: List[RecentApplication]

class BreakdownItem(BaseModel):
    label: str
    value: int

class DashboardResponse(BaseModel):
    recent_applications: PaginatedApplicationsResponse
    total_applications: int
    approval_rate_30_days: float
    prediction_breakdown: List[BreakdownItem]
    purpose_breakdown: List[BreakdownItem]

# --- Output Model for Performance Endpoint ---
class PerformanceResponse(BaseModel):
    model_version: str
    training_date: datetime  # <--- 2. ระบุ Type Hint เป็น 'datetime' ที่เรา import มา
    accuracy: float
    precision_class_1: float
    recall_class_1: float
    f1_score_class_1: float
    meta_model_importances: Dict[str, Any]
    optimal_threshold: float

    class Config:
        from_attributes = True # for SQLAlchemy ORM compatibility