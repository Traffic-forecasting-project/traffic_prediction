'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "FastAPI dependencies for role-based access control"
'''

from typing import List
from fastapi import Depends, HTTPException, status

from src.auth.jwt_auth import get_current_user
from src.auth.roles import UserInDB
from src.core.logging_utils import get_logger

## ============================
## Setup logger
## ============================
logger = get_logger(__name__)

## ============================
## Role-based access checker
## ============================
class RoleChecker:
    """
        Dependency class to enforce role-based access control on endpoints
    """

    def __init__(self, allowed_roles: List[str]) -> None:
        """
            Initialize role checker with allowed roles
            Args:
                allowed_roles (List[str]): List of roles that are authorized
        """
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
        """
            Check if current user has one of the allowed roles.

            Args:
                current_user (UserInDB): The authenticated user extracted from JWT

            Returns:
                UserInDB: Current user if authorized

            Raises:
                HTTPException: If user does not have required role
        """
        
        ## Log the attempt with user and required roles
        logger.debug(
            f"User {current_user.username} with role {current_user.role} "
            f"attempts to access endpoint requiring {self.allowed_roles}"
        )

        ## If user role is not in allowed roles -> deny access
        if current_user.role not in self.allowed_roles:
            logger.warning(
                f"Access denied for user {current_user.username} with role {current_user.role}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operation not permitted"
            )

        ## Success -> grant access
        logger.info(
            f"Access granted for user {current_user.username} with role {current_user.role}"
        )
        return current_user
