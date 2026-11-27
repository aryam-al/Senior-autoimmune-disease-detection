"""Settings read from environment variables (or a local .env file)."""

import os
import secrets
import warnings

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


# --- Security -----------------------------------------------------------------
SECRET_KEY = os.getenv("SECRET_KEY", "")
if not SECRET_KEY:
    # A random key works for local development but logs everyone out on restart.
    SECRET_KEY = secrets.token_urlsafe(48)
    warnings.warn("SECRET_KEY is not set - using a temporary random key. Set it in .env.")

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", str(60 * 24)))

# Frontend origins allowed to call the API.
CORS_ORIGINS = _list(os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"))
# Also allow phones/laptops on the same Wi-Fi (private network addresses). Set to empty to turn off.
CORS_ORIGIN_REGEX = os.getenv(
    "CORS_ORIGIN_REGEX",
    r"https?://(localhost|127\.0\.0\.1|10(\.\d{1,3}){3}|192\.168(\.\d{1,3}){2}|172\.(1[6-9]|2\d|3[01])(\.\d{1,3}){2})(:\d+)?",
) or None

# --- Sign in with Google (leave empty to hide the button) ----------------------
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()

# --- Storage ------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'immunoai.db')}")
LEGACY_USERS_FILE = os.path.join(BASE_DIR, "users.json")

# --- Models -------------------------------------------------------------------
XGB_MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.json")
XGB_META_PATH = os.path.join(BASE_DIR, "models", "xgboost_meta.json")
XGB_MISSING_MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_missing_model.json")
XGB_MISSING_META_PATH = os.path.join(BASE_DIR, "models", "xgboost_missing_meta.json")

# Qwen runs outside the backend. Leave QWEN_API_URL empty to disable.
#   QWEN_MODE=local   llama-server on this computer (CPU, see qwen_local/)
#   QWEN_MODE=remote  GPU service from qwen_service/ (Colab or Modal)
QWEN_MODE = os.getenv("QWEN_MODE", "remote").strip().lower()
QWEN_API_URL = os.getenv("QWEN_API_URL", "").rstrip("/")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_TIMEOUT_SECONDS = float(os.getenv("QWEN_TIMEOUT_SECONDS", "60"))
# true for Modal: the GPU starts on the first request, so /health must not wake it up
QWEN_ON_DEMAND = os.getenv("QWEN_ON_DEMAND", "false").strip().lower() in ("1", "true", "yes")

# --- Uploads ------------------------------------------------------------------
MAX_PDF_BYTES = int(os.getenv("MAX_PDF_MB", "10")) * 1024 * 1024
MAX_PDF_FILES = int(os.getenv("MAX_PDF_FILES", "10"))

DISCLAIMER = (
    "This result is produced by research models trained on a limited dataset. "
    "It is not a medical diagnosis. Please consult a qualified physician."
)
