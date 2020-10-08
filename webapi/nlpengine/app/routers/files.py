from fastapi import APIRouter, File, UploadFile
from typing import List
from pydantic import BaseModel, Field
from ..utils import save_upload_file
import os

router = APIRouter()


@router.post("/files/uploadone", tags=["files"])
async def files_upload_one(file: UploadFile = File(...)):
    try:
        await save_upload_file(file, os.environ["UPLOAD_FOLDER"])
    except Exception as e:
        return {"file_name": file.filename, "success": False, "message": e}

    return {"file_name": file.filename, "success": True}


@router.post("/files/uploadmany", tags=["files"])
async def files_upload_many(files: List[UploadFile] = File(...)):
    return {"file_names": [file.filename for file in files]}
