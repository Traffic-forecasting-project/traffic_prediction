'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Role definitions and user store loaded from .env for JWT authentication"
'''

import os
from dataclasses import dataclass
from dotenv import load_dotenv

## Load environment variables
load_dotenv()

## ========================
## Role constants
## ========================
ROLE_ADMIN = "admin"
ROLE_USER = "user"
ROLE_GUEST = "guest"

## ========================
## Role descriptions (optional, for documentation/logging)
## ========================
ROLES = {
    ROLE_ADMIN: "Administrator with full access",
    ROLE_USER: "Standard user with limited access",
    ROLE_GUEST: "Guest with minimal access",
}

## ========================
## User model (in-memory representation)
## ========================
@dataclass
class UserInDB:
    username: str
    password: str   ## TO DO : to be hashed in real systems and stored in RDBS
    role: str

## ========================
## Load users from .env
## ========================
USERS_DB = {
    os.getenv("ADMIN_USER"): UserInDB(
        username=os.getenv("ADMIN_USER"),
        password=os.getenv("ADMIN_PASS"),
        role=os.getenv("ADMIN_ROLE", ROLE_ADMIN),
    ),
    os.getenv("USER_USER"): UserInDB(
        username=os.getenv("USER_USER"),
        password=os.getenv("USER_PASS"),
        role=os.getenv("USER_ROLE", ROLE_USER),
    ),
    os.getenv("GUEST_USER"): UserInDB(
        username=os.getenv("GUEST_USER"),
        password=os.getenv("GUEST_PASS"),
        role=os.getenv("GUEST_ROLE", ROLE_GUEST),
    ),
}
