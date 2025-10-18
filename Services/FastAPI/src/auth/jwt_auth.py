'''
__author__ = "Georges Nassopoulos"
__contributors__ = "Mateo Villa Arias"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "FastAPI JWT authentication, creation and verification"
'''

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt

## Imports for "microservice" et "legacy" structures respectively
try:
    from Services.FastAPI.src.logging_utils import get_logger
    from Services.FastAPI.src.auth.roles import USERS_DB, UserInDB    
    from Services.FastAPI.src.constants import (
        SECRET_KEY,
        ALGORITHM,
        ACCESS_TOKEN_EXPIRE_MINUTES
    )
except:
    from src.logging_utils import get_logger
    from src.auth.roles import USERS_DB, UserInDB
    from src.constants import (
        SECRET_KEY,
        ALGORITHM,
        ACCESS_TOKEN_EXPIRE_MINUTES
    )
       
## ============================
## Setup logger
## ============================
logger = get_logger(__name__)

## ============================
## OAuth2 setup
## ============================
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

## ============================
## User authentication
## ============================
def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """
        Authenticate user against in-memory USERS_DB loaded from .env

        Args:
            username (str): Provided username
            password (str): Provided password

        Returns:
            UserInDB | None: User object if valid, otherwise None
    """
    
    ## Look up the user in the in-memory USERS_DB
    user = USERS_DB.get(username)

    ## If no user found or password does not match -> authentication fails
    if not user or user.password != password:
        logger.warning(f"Authentication failed for user: {username}")
        return None

    ## Otherwise return the user object (username, password, role)
    logger.info(f"User authenticated: {username} with role {user.role}")
    return user

## ============================
## JWT token creation
## ============================
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
        Create a JWT token with username and role included

        Args:
            data (dict): Payload data, should include 'sub' (username) and 'role'
            expires_delta (timedelta | None): Expiration time delta

        Returns:
            str: Encoded JWT token
    """
    
    ## Copy input data (contains at least sub=username, role=user_role)
    to_encode = data.copy()

    ## Compute expiration (default is ACCESS_TOKEN_EXPIRE_MINUTES from .env)
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})

    ## Encode the JWT using our SECRET_KEY and ALGORITHM
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    ## Debug log for tracking token creation
    logger.debug(f"JWT created for {data.get('sub')} with role {data.get('role')}")
    return token

## ============================
## JWT verification & user extraction
## ============================
async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    """
        Decode JWT token and return current user with role

        Args:
            token (str): JWT token from Authorization header

        Returns:
            UserInDB: Authenticated user

        Raises:
            HTTPException: If token invalid or role mismatch
    """
    
    try:
        ## Decode the token using our secret and algorithm
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        ## Extract username and role from the token payload
        username: str = payload.get("sub")
        role: str = payload.get("role")

        ## If missing information -> invalid token
        if username is None or role is None:
            logger.error("Token payload missing username or role")
            raise HTTPException(status_code=401, detail="Invalid token payload")

    except JWTError as e:
        ## If decoding fails (wrong signature, expired, corrupted, etc.)
        logger.error(f"JWT decoding failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")

    ## Retrieve the user from our USERS_DB
    user = USERS_DB.get(username)

    ## If user not found or role does not match -> reject
    if user is None or user.role != role:
        logger.warning(f"Invalid token for user {username} with role {role}")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    ## Success -> return the full user object (username, password, role)
    return user