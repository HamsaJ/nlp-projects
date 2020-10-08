from fastapi import APIRouter, File, UploadFile, Path
from typing import List
from pydantic import BaseModel, Field
from ..utils import (
    save_upload_file,
    get_files_in_dir,
    get_job_and_cvs,
    get_cvranks_for_job,
)
import os

router = APIRouter()


@router.post("/job/listing", tags=["job"])
async def job_upload_one(id: int, file: UploadFile = File(...)):
    try:
        directory = os.environ["UPLOAD_FOLDER"] + f"/jobs/{id}"
        await save_upload_file(file, directory)
    except Exception as e:
        return {"file_name": file.filename, "success": False, "message": e}

    return {"file_name": file.filename, "success": True}


@router.get("/job/listing/{id}", tags=["job"])
def job_listing_one(
    id: int = Path(..., title="The ID of the item to get"),
):
    directory = os.environ["UPLOAD_FOLDER"] + f"/jobs/{id}"
    result, message = get_files_in_dir(directory)
    if result:
        return {"file_name": message, "success": result}
    else:
        return {"message": message, "success": result}


@router.post("/job/cv", tags=["job"])
async def job_upload_cv_one(id: int, job_id: int, file: UploadFile = File(...)):
    try:
        directory = os.environ["UPLOAD_FOLDER"] + f"/jobs/{job_id}/cv"
        await save_upload_file(file, directory, id)
    except Exception as e:
        return {"file_name": file.filename, "success": False, "message": e}

    return {"file_name": file.filename, "success": True}


@router.post("/job/rankcvs", tags=["job"])
async def job_rank_cv(job_id: int):
    directory = os.environ["UPLOAD_FOLDER"] + f"/jobs/{job_id}"
    job_listing, cv_score = get_cvranks_for_job(directory)

    return {"job_listing": job_listing, "cv_score": cv_score, "success": True}


@router.get("/job/{job_id}", tags=["job"])
async def job_and_cvs(job_id: int):
    directory = os.environ["UPLOAD_FOLDER"] + f"/jobs/{job_id}"
    job_listing, cv_list = get_job_and_cvs(directory)

    return {"job_listing": job_listing, "cvs": cv_list, "success": True}