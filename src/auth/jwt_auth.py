'''
__author__ = -
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "FastAPI  JWT authentication, creation and verification"
'''

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta
from jose import JWTError, jwt
from typing import Optional

from src.core.logging_utils import get_logger
from src.core.constants import (
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

logger = get_logger(__name__)

## ============================
## Auth system
## ============================
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

## ============================
## JWT Auth Functions
## ============================
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
        Create JWT token from user data

        Args:
            data (dict): Data to encode (e.g., {'sub': 'username'})
            expires_delta (timedelta, optional): Expiry duration

        Returns:
            str: Encoded JWT token
    """
    
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def verify_token(token: str = Depends(oauth2_scheme)) -> str:
    """
        Validate JWT token

        Args:
            token (str): JWT token from Authorization header

        Returns:
            str: Username if valid

        Raises:
            HTTPException: If token invalid
    """
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return username
    except JWTError as e:
        logger.error(f"JWT decoding failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")