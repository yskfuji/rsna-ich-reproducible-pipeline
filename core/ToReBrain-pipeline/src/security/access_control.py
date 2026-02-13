"""Minimal token-based authorization stub."""
import os
from fastapi import HTTPException, status

API_TOKEN = os.getenv("API_TOKEN", "change-me")

def authorize(token: str = API_TOKEN):
    if token != API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized")
    return True
