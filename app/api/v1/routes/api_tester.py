# app/api/v1/routes/api_tester.py
"""
API Tester JavaScript endpoint
Serves the JavaScript code for the EXO API Tester page
"""
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from app.database.models import User
from app.api.v1.routes.admin import require_admin_user

router = APIRouter()

# Path to the JavaScript file
JS_FILE_PATH = Path(__file__).parent.parent.parent.parent / "static" / "js" / "api_tester.js"


@router.get("/api-tester/script.js", name="admin_api_tester_script")
async def admin_api_tester_script(admin_user: User = Depends(require_admin_user)):
    """
    Returns the JavaScript code for the API tester page.
    Reads the JavaScript from a separate file instead of embedding it in Python.
    """
    try:
        if not JS_FILE_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail=f"JavaScript file not found: {JS_FILE_PATH}"
            )
        
        js_code = JS_FILE_PATH.read_text(encoding="utf-8")
        return Response(content=js_code, media_type="application/javascript")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading JavaScript file: {str(e)}"
        )

