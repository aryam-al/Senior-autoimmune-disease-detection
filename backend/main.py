"""ImmunoAI backend - autoimmune disease classification with XGBoost and Qwen2.5-7B."""

import asyncio
import json
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from typing import List
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from sqlalchemy.orm import Session

import chat
import config
import qwen_client
from auth import create_access_token, get_current_user, hash_password, verify_password
from database import ReportHistory, User, get_db, init_db
from features import CONTINUOUS, DISPLAY_NAMES, FEATURES
from model import XGBoostPredictor
from pdf_parser import extract_lab_values, extract_text_from_pdf, merge_reports
from schemas import (
    ChangePasswordRequest, ChatRequest, GoogleSignIn, LoginRequest, PatientData, PredictRequest,
    RegisterRequest, password_problems,
)
from social_auth import SocialAuthError, verify_id_token

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    state["xgb"] = XGBoostPredictor()
    yield


app = FastAPI(
    title="ImmunoAI Backend",
    description="Autoimmune disease classification API (XGBoost + fine-tuned Qwen2.5-7B)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_origin_regex=config.CORS_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# --------------------------------------------------------------------------- helpers

def _has_password(user: User) -> bool:
    # Accounts made with Google sign-in store "!google:..." instead of a password hash.
    return not (user.password_hash or "").startswith("!")


def _user_out(user: User) -> dict:
    return {"name": user.name, "email": user.email, "has_password": _has_password(user)}


def _auth_response(user: User) -> dict:
    return {"token": create_access_token(user.email), **_user_out(user)}


def _selected(r: ReportHistory) -> list[str]:
    if r.result_json:
        try:
            return json.loads(r.result_json).get("selected") or ["xgboost", "qwen"]
        except ValueError:
            pass
    return ["xgboost", "qwen"]


def _missing_count(r: ReportHistory) -> int:
    try:
        return sum(1 for v in json.loads(r.input_json or "{}").values() if v is None)
    except ValueError:
        return 0


def _history_item(r: ReportHistory) -> dict:
    return {
        "selected": _selected(r),
        "missing_count": _missing_count(r),
        "id": str(r.id),
        "disease": r.disease,
        "confidence": r.confidence,
        "qwen_disease": r.qwen_disease,
        "qwen_confidence": r.qwen_confidence,
        "source_filename": r.source_filename,
        "date": r.date,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


MODEL_INFO = {
    "xgboost": {"name": "XGBoost", "type": "Machine learning (gradient-boosted trees)"},
    "qwen": {"name": "Qwen2.5-7B (fine-tuned)", "type": "Deep learning (large language model, QLoRA)"},
}


def _skipped(key: str) -> dict:
    return {"available": False, "skipped": True, "model": MODEL_INFO[key]["name"], "error": "Not selected for this analysis."}


async def _timed(coro) -> dict:
    start = time.perf_counter()
    result = await coro
    return {**result, "time_ms": round((time.perf_counter() - start) * 1000)}


async def _run_models(patient: dict, selected: list[str] | None = None, extra: list | None = None) -> dict:
    """Run the chosen models on the same values (both run at the same time)."""
    selected = [k for k in ("xgboost", "qwen") if k in (selected or ["xgboost", "qwen"])]

    async def nothing(key):
        return _skipped(key)

    xgb_coro = _timed(asyncio.to_thread(state["xgb"].predict, patient)) if "xgboost" in selected else nothing("xgboost")
    qwen_coro = _timed(qwen_client.predict(patient, extra)) if "qwen" in selected else nothing("qwen")
    xgb_result, qwen_result = await asyncio.gather(xgb_coro, qwen_coro)
    for key, res in (("xgboost", xgb_result), ("qwen", qwen_result)):
        res["type"] = MODEL_INFO[key]["type"]
    both = xgb_result.get("available") and qwen_result.get("available")
    agreement = qwen_result["disease"] == xgb_result["disease"] if both else None
    return {
        "selected": selected,
        "xgboost": xgb_result,
        "qwen": qwen_result,
        "agreement": agreement,
        "missing": [f for f in FEATURES if patient.get(f) is None],  # left out, never guessed
        "extra": extra or [],  # other lab results: used by Qwen and the chat, not by XGBoost
    }


def _save(db: Session, user: User, patient: dict, models: dict, filename: str | None) -> ReportHistory:
    xgb, qwen = models["xgboost"], models["qwen"]
    main_model = xgb if xgb.get("available") else qwen  # the summary shown in lists
    entry = ReportHistory(
        user_email=user.email,
        disease=main_model.get("disease"),
        confidence=main_model.get("confidence"),
        qwen_disease=qwen.get("disease"),
        qwen_confidence=qwen.get("confidence"),
        source_filename=filename,
        input_json=json.dumps(patient),
        result_json=json.dumps(models),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    entry.date = entry.created_at.strftime("%b %d, %Y")
    db.commit()
    return entry


def _result(entry: ReportHistory, patient: dict, models: dict) -> dict:
    return {
        "id": str(entry.id),
        "created_at": entry.created_at.isoformat(),
        "source_filename": entry.source_filename,
        "input": patient,
        **models,
        "disclaimer": config.DISCLAIMER,
    }


async def _read_pdf(file: UploadFile, allow_empty: bool = False) -> dict:
    name = file.filename or "report.pdf"
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=f"{name}: only PDF files are allowed.")
    pdf_bytes = await file.read(config.MAX_PDF_BYTES + 1)
    if len(pdf_bytes) > config.MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail=f"{name} is too large.")
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail=f"{name} is not a valid PDF.")
    try:
        text = extract_text_from_pdf(pdf_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail=f"{name} could not be read.")
    if not text.strip() and allow_empty:
        return {"filename": name, "values": {}, "missing": list(FEATURES), "extra": [],
                "warnings": ["No text found (scanned image?) - nothing was read from this file."]}
    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No text found in the PDF. Scanned images are not supported - please enter the values manually.",
        )
    return {"filename": name, **extract_lab_values(text)}


# --------------------------------------------------------------------------- general

@app.get("/")
def home():
    return {"status": "ImmunoAI backend is running"}


@app.get("/health")
async def health():
    qwen = await qwen_client.health()
    return {"api": "ok", "xgboost": "xgb" in state, "qwen": qwen, "chat": chat.available() and qwen.get("online", False)}


@app.get("/models")
def models_info():
    """Model details and test results, for the about / info screen."""
    return {
        "xgboost": {"name": "XGBoost", "type": MODEL_INFO["xgboost"]["type"], **state["xgb"].test_metrics},
        "qwen": {
            "name": "Qwen2.5-7B-Instruct (QLoRA fine-tuned)",
            "type": MODEL_INFO["qwen"]["type"],
            "accuracy": 0.9325, "macro_precision": 0.9333, "macro_recall": 0.9325, "macro_f1": 0.9311,
            "train_samples": 2000, "test_samples": 400,
        },
        "classes": state["xgb"].classes,
        "disclaimer": config.DISCLAIMER,
    }


@app.get("/features")
def features_info():
    """Input fields for the manual-entry form."""
    return [
        {
            "name": f,
            "label": DISPLAY_NAMES[f],
            "type": "number" if f in CONTINUOUS else ("gender" if f == "Gender" else "binary"),
            **({"unit": CONTINUOUS[f][0], "min": CONTINUOUS[f][1], "max": CONTINUOUS[f][2]} if f in CONTINUOUS else {}),
        }
        for f in FEATURES
    ]


# --------------------------------------------------------------------------- auth

@app.post("/register", status_code=201)
def register(data: RegisterRequest, db: Session = Depends(get_db)):
    email = data.email.lower()
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="This email is already registered.")
    user = User(email=email, name=data.name.strip(), password_hash=hash_password(data.password))
    db.add(user)
    db.commit()
    return _auth_response(user)


@app.post("/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email.lower()).first()
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    return _auth_response(user)


@app.get("/auth/providers")
def auth_providers():
    """Whether the frontend should show the Google button (the client ID is public)."""
    return {"google": {"client_id": config.GOOGLE_CLIENT_ID} if config.GOOGLE_CLIENT_ID else None}


def _social_sign_in(db: Session, provider: str, token: str, name: str | None = None) -> dict:
    try:
        identity = verify_id_token(provider, token)
    except SocialAuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    user = db.query(User).filter(User.email == identity["email"]).first()
    if not user:
        # No password: this account signs in through the provider only.
        display = (name or identity.get("name") or identity["email"].split("@")[0]).strip()
        user = User(email=identity["email"], name=display, password_hash=f"!{provider}:{secrets.token_hex(16)}")
        db.add(user)
        db.commit()
    return _auth_response(user)


@app.post("/auth/google")
def google_sign_in(data: GoogleSignIn, db: Session = Depends(get_db)):
    return _social_sign_in(db, "google", data.credential)


@app.get("/me")
def me(user: User = Depends(get_current_user)):
    return _user_out(user)


@app.post("/me/password")
def change_password(data: ChangePasswordRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Change the password (current password required). A Google-only account can set one here,
    so it can also sign in with email and password."""
    if _has_password(user):
        if not data.current_password or not verify_password(data.current_password, user.password_hash):
            raise HTTPException(status_code=400, detail="Your current password is not correct.")
        if verify_password(data.new_password, user.password_hash):
            raise HTTPException(status_code=400, detail="The new password must be different from the current one.")
    problems = password_problems(data.new_password, user.email, user.name)
    if problems:
        raise HTTPException(status_code=400, detail="Password is too weak. It needs " + ", ".join(problems) + ".")
    user.password_hash = hash_password(data.new_password)
    db.commit()
    return {"message": "Your password was changed.", **_user_out(user)}


# --------------------------------------------------------------------------- prediction

@app.post("/predict")
async def predict(data: PredictRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Run the chosen models (XGBoost, Qwen, or both to compare) on values entered or confirmed by the user."""
    patient = data.model_dict()
    extra = [e.model_dump(exclude_none=True) for e in data.extra]
    models = await _run_models(patient, data.models, extra)
    entry = _save(db, user, patient, models, data.source_filename)
    return _result(entry, patient, models)


@app.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...), user: User = Depends(get_current_user)):
    """Read a lab report and return the values found, without predicting.
    The frontend shows them for the user to check and complete, then calls /predict."""
    return await _read_pdf(file)


@app.post("/extract-pdfs")
async def extract_pdfs(files: List[UploadFile] = File(...), user: User = Depends(get_current_user)):
    """Read several lab reports of the same patient and combine what was found.
    Values in more than one report keep the first value and add a warning to check it."""
    if len(files) > config.MAX_PDF_FILES:
        raise HTTPException(status_code=400, detail=f"Please upload at most {config.MAX_PDF_FILES} PDFs at a time.")
    reports = [await _read_pdf(f, allow_empty=True) for f in files]
    if not any(r["values"] or r["extra"] for r in reports):
        raise HTTPException(
            status_code=422,
            detail="No lab results were found in these PDFs. Scanned images are not supported - please type the values instead.",
        )
    return merge_reports(reports)


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Read a lab report and predict straight away - only when every value was found."""
    extracted = await _read_pdf(file)
    try:
        patient = PatientData(**extracted["values"]).model_dict()
    except ValidationError as e:
        raise HTTPException(status_code=422, detail={"message": "Some values look wrong.", "errors": e.errors(), **extracted})
    models = await _run_models(patient, None, extracted["extra"])
    entry = _save(db, user, patient, models, extracted["filename"])
    return {**_result(entry, patient, models), "warnings": extracted["warnings"]}


# --------------------------------------------------------------------------- history

@app.get("/history")
def get_history(limit: int = 50, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    reports = (
        db.query(ReportHistory)
        .filter(ReportHistory.user_email == user.email)
        .order_by(ReportHistory.created_at.desc())
        .limit(max(1, min(limit, 200)))
        .all()
    )
    return [_history_item(r) for r in reports]


@app.get("/history/{report_id}")
def get_report(report_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    r = db.query(ReportHistory).filter(ReportHistory.id == report_id, ReportHistory.user_email == user.email).first()
    if not r:
        raise HTTPException(status_code=404, detail="Report not found.")
    if not r.result_json:  # saved by the old version: only the summary exists
        return {**_history_item(r), "legacy": True, "disclaimer": config.DISCLAIMER}
    return {
        "id": str(r.id),
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "source_filename": r.source_filename,
        "input": json.loads(r.input_json),
        **json.loads(r.result_json),
        "disclaimer": config.DISCLAIMER,
    }


@app.delete("/history/{report_id}")
def delete_report(report_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    r = db.query(ReportHistory).filter(ReportHistory.id == report_id, ReportHistory.user_email == user.email).first()
    if not r:
        raise HTTPException(status_code=404, detail="Report not found.")
    db.delete(r)
    db.commit()
    return {"message": "Report deleted."}


@app.delete("/history")
def clear_history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.query(ReportHistory).filter(ReportHistory.user_email == user.email).delete()
    db.commit()
    return {"message": "History cleared."}


# --------------------------------------------------------------------------- chat

@app.post("/chat")
async def chat_about_report(data: ChatRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Ask Qwen about one of your analyses. The answer is streamed as plain text."""
    if not chat.available():
        raise HTTPException(status_code=503, detail="The chat needs Qwen running on this computer (QWEN_MODE=local).")
    if data.messages[-1].role != "user":
        raise HTTPException(status_code=422, detail="The last message must be from the user.")
    r = db.query(ReportHistory).filter(ReportHistory.id == data.report_id, ReportHistory.user_email == user.email).first()
    if not r:
        raise HTTPException(status_code=404, detail="Report not found.")
    if not r.result_json or not r.input_json:
        raise HTTPException(status_code=422, detail="This analysis was saved by an older version and has no details to discuss.")
    report = {"input": json.loads(r.input_json), **json.loads(r.result_json)}
    messages = [m.model_dump() for m in data.messages]

    gen = chat.stream_answer(report, messages)
    try:
        first = await gen.__anext__()  # fail now (with a proper error) if Qwen isn't running
    except StopAsyncIteration:
        first = ""
    except Exception:
        raise HTTPException(status_code=503, detail="Qwen is not running. Start it with START.bat and try again.")

    async def body():
        yield first
        try:
            async for piece in gen:
                yield piece
        except Exception:
            yield "\n\n[The answer was cut off - Qwen stopped responding.]"

    return StreamingResponse(body(), media_type="text/plain; charset=utf-8", headers={"Cache-Control": "no-cache"})
