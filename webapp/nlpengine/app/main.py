from dotenv import load_dotenv

load_dotenv(verbose=True)

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .routers import nlp, files, jobs

app = FastAPI(title="Hitech ML Lab", version="0.1.0")
origins = [
    "http://localhost",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_token_header(x_token: str = Header(...)):
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")


app.include_router(nlp.router)
app.include_router(files.router)
app.include_router(jobs.router)
