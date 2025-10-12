# app/auth.py

from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from . import database, db_models, schemas

# --- Configuration ---
# ควรเก็บค่าเหล่านี้ไว้ใน Environment Variables เพื่อความปลอดภัย
SECRET_KEY = "YOUR_SUPER_SECRET_KEY" # << เปลี่ยนเป็นค่าของคุณ
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- Password Hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# --- JWT Token Creation ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- OAuth2 Scheme ---
# tokenUrl="token" คือการบอกให้ client ไปขอ token ที่ endpoint /token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# --- Database Interaction ---
def get_user(db: Session, username: str):
    return db.query(db_models.User).filter(db_models.User.username == username).first()

# --- The Core Dependency Function ---
def get_current_active_user(
    token: str = Depends(oauth2_scheme), 
    db: Session = Depends(database.get_db)
) -> schemas.User:
    """
    Dependency ที่จะถอดรหัส Token, ดึงข้อมูล User, และตรวจสอบว่าเป็น Active User หรือไม่
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = schemas.TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = get_user(db, username=token_data.username)
    
    if user is None:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
        
    return user