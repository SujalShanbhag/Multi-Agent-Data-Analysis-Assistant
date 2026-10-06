from pathlib import Path
import os
import re
import sys
import uuid

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Ensure project root is always in Python's search path
BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

# Load environment configuration (.env file)
ENV_PATH = BASE / ".env"
load_dotenv(ENV_PATH)

from backend.tools.data_tools import load_dataframe
from backend.agents import profiling_agent
from backend.agents.query_agent import discover_gemini_models

# Determine writable storage directory (supports serverless /tmp on Vercel / AWS Lambda)
import tempfile

if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    RUNTIME_TMP = Path(tempfile.gettempdir()) / "data_analysis_assistant"
    UPLOAD_DIR = RUNTIME_TMP / "data"
    REPORT_DIR = RUNTIME_TMP / "reports"
else:
    UPLOAD_DIR = BASE / "data"
    REPORT_DIR = BASE / "reports"

FRONTEND_DIR = BASE / "frontend"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def get_workflow():
    from backend.workflows.analysis_graph import run_workflow
    return run_workflow

app = FastAPI(
    title="Multi-Agent Data Analysis Assistant",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="static",
)

STATE = {
    "path": None,
    "name": None,
    "df": None,
    "last_result": None,
}


class AnalyzeRequest(BaseModel):
    target: str | None = None


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico")
def favicon():
    return JSONResponse(status_code=204, content=None)


@app.get("/style.css")
def style_css():
    return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")


@app.get("/app.js")
def app_js():
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "service": "multi-agent-data-analysis-assistant",
        "version": "2.0.0",
    }


@app.get("/api/ai-status")
def ai_status():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    is_configured = bool(api_key)
    discovered = discover_gemini_models() if is_configured else []
    return {
        "provider": "Google Gemini Multi-Model AI" if is_configured else "Local Analytics Engine",
        "configured": is_configured,
        "models_discovered": discovered,
        "models_count": len(discovered),
        "mode": "dynamic_multi_model",
    }





@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_dataset.csv"
    suffix = Path(filename).suffix.lower()

    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(400, "Only CSV (.csv) and Excel (.xlsx, .xls) files are supported.")

    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    dest_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{safe_name}"

    try:
        content = await file.read()
        dest_path.write_bytes(content)
        df = load_dataframe(dest_path)
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read uploaded dataset: {exc}")

    if df.empty:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(400, "The uploaded dataset is empty.")

    STATE.update({
        "path": str(dest_path),
        "name": safe_name,
        "df": df,
        "last_result": None,
    })

    return {
        "name": safe_name,
        "profile": profiling_agent.run(df),
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    if STATE["df"] is None:
        raise HTTPException(400, "Please upload a dataset first.")

    target = req.target.strip() if req.target else None
    if target and target not in STATE["df"].columns:
        raise HTTPException(400, f"Target column '{target}' was not found in dataset columns.")

    try:
        workflow = get_workflow()
        result = workflow(STATE["df"], STATE["name"], target)
        STATE["last_result"] = result
        return result
    except Exception as exc:
        raise HTTPException(500, f"Multi-agent analysis failed: {exc}")


@app.post("/api/query")
def query(req: QueryRequest):
    if STATE["df"] is None:
        raise HTTPException(400, "Please upload a dataset first.")

    try:
        from backend.agents.query_agent import answer_question
        return answer_question(STATE["df"], req.question)
    except Exception as exc:
        raise HTTPException(500, f"Query processing failed: {exc}")


@app.get("/api/report")
def report():
    path = None
    if STATE.get("last_result") and STATE["last_result"].get("report"):
        candidate = Path(STATE["last_result"]["report"])
        if candidate.exists():
            path = candidate

    if not path or not path.exists():
        path = REPORT_DIR / "data_analysis_report.pdf"

    if not path.exists():
        raise HTTPException(404, "No PDF report has been generated yet. Click 'Run Full Analysis' first.")

    return FileResponse(
        path,
        filename="data_analysis_report.pdf",
        media_type="application/pdf",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
