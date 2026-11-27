"""Request / response models."""

import re
from typing import Literal, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

from features import BINARY, CONTINUOUS, FEATURES

_POSITIVE = {"1", "1.0", "positive", "pos", "+", "true", "yes", "reactive", "detected"}
_NEGATIVE = {"0", "0.0", "negative", "neg", "-", "false", "no", "non-reactive", "not detected"}


class ExtraResult(BaseModel):
    """A lab result that is not one of the 23 model inputs (used by Qwen and the chat only)."""

    name: str = Field(min_length=1, max_length=60)
    value: str = Field(min_length=1, max_length=40)
    unit: str = Field(default="", max_length=20)
    flag: str = Field(default="", max_length=10)
    source: Optional[str] = Field(default=None, max_length=255)

    @field_validator("name", "value", "unit", "flag", mode="before")
    @classmethod
    def _strip(cls, v):
        return str(v).strip() if v is not None else ""


class PatientData(BaseModel):
    """The 23 inputs the models were trained on. Any of them may be left out (None):
    a value that was not tested is never guessed - XGBoost treats it as missing and
    Qwen is simply not shown that line.

    Gender accepts "Male"/"Female" or 0/1 (0 = Male, 1 = Female).
    Markers accept 0/1, true/false or "positive"/"negative".
    """

    Age: Optional[float] = None
    Gender: Optional[Literal["Male", "Female"]] = None
    ESR: Optional[float] = None
    CRP: Optional[float] = None
    RF: Optional[int] = None
    Anti_CCP: Optional[int] = None
    HLA_B27: Optional[int] = None
    ANA: Optional[int] = None
    Anti_Ro: Optional[int] = None
    Anti_La: Optional[int] = None
    Anti_dsDNA: Optional[int] = None
    Anti_Sm: Optional[int] = None
    C3: Optional[float] = None
    C4: Optional[float] = None
    ASCA: Optional[int] = None
    Anti_CBir1: Optional[int] = None
    Anti_OmpC: Optional[int] = None
    pANCA: Optional[int] = None
    EMA: Optional[int] = None
    DGP: Optional[int] = None
    Anti_tTG: Optional[int] = None
    Anti_TPO: Optional[int] = None
    Anti_SMA: Optional[int] = None

    @field_validator(*FEATURES, mode="before")
    @classmethod
    def _empty_is_missing(cls, v):
        if v is None or (isinstance(v, str) and v.strip().lower() in ("", "not tested", "unknown", "n/a", "na", "-")):
            return None
        return v

    @field_validator("Gender", mode="before")
    @classmethod
    def _gender(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if v in (0, 1):
                return "Female" if v == 1 else "Male"
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("male", "m", "0"):
                return "Male"
            if s in ("female", "f", "1"):
                return "Female"
        raise ValueError("Gender must be 'Male' or 'Female' (or 0 = Male, 1 = Female)")

    @field_validator(*BINARY, mode="before")
    @classmethod
    def _binary(cls, v):
        if v is None:
            return None
        if isinstance(v, bool):
            return int(v)
        s = str(v).strip().lower()
        if s in _POSITIVE:
            return 1
        if s in _NEGATIVE:
            return 0
        raise ValueError("must be positive (1) or negative (0)")

    @model_validator(mode="after")
    def _ranges(self):
        for name, (unit, low, high) in CONTINUOUS.items():
            value = getattr(self, name)
            if value is not None and not (low <= value <= high):
                raise ValueError(f"{name} = {value} is outside the expected range {low}-{high} {unit}")
        return self

    def model_dict(self) -> dict:
        return {f: getattr(self, f) for f in FEATURES}

    def missing(self) -> list[str]:
        return [f for f in FEATURES if getattr(self, f) is None]


MODEL_KEYS = ("xgboost", "qwen")


class PredictRequest(PatientData):
    source_filename: Optional[str] = Field(default=None, max_length=1000)  # several PDFs: "a.pdf, b.pdf"
    # Which models to run: one of them, or both to compare them on the same values.
    models: list[Literal["xgboost", "qwen"]] = Field(default_factory=lambda: list(MODEL_KEYS), min_length=1, max_length=2)
    # Other lab results from the reports (not model inputs): given to Qwen and the chat.
    extra: list[ExtraResult] = Field(default_factory=list, max_length=40)

    @model_validator(mode="after")
    def _enough_values(self):
        tests = [f for f in FEATURES if f not in ("Age", "Gender") and getattr(self, f) is not None]
        if not tests:
            raise ValueError("Enter at least one lab result (a blood test or an antibody test).")
        return self


COMMON_PASSWORDS = {
    "password", "password1", "password123", "passw0rd", "12345678", "123456789", "1234567890", "qwerty123",
    "qwertyuiop", "iloveyou", "admin123", "welcome1", "welcome123", "letmein1", "abc12345", "11111111",
    "00000000", "football", "baseball", "sunshine", "princess", "immunoai", "immunoai1", "immunoai123",
}


def password_problems(password: str, email: str = "", name: str = "") -> list[str]:
    """What is wrong with a password (empty list = strong enough)."""
    problems = []
    if len(password) < 10:
        problems.append("at least 10 characters")
    if not re.search(r"[a-z]", password):
        problems.append("a lowercase letter")
    if not re.search(r"[A-Z]", password):
        problems.append("an uppercase letter")
    if not re.search(r"\d", password):
        problems.append("a number")
    if not re.search(r"[^A-Za-z0-9]", password):
        problems.append("a symbol (for example ! @ # ?)")
    low = password.lower()
    simple = re.sub(r"[^a-z0-9]", "", low)
    personal = [p for p in (email.split("@")[0].lower() if email else "", *(name.lower().split() if name else [])) if len(p) >= 3]
    if low in COMMON_PASSWORDS or simple in COMMON_PASSWORDS or re.fullmatch(r"(.)\1+", password):
        problems.append("not a common password")
    elif any(p in low for p in personal):
        problems.append("not your name or email")
    return problems


def check_password(password: str, email: str = "", name: str = "") -> str:
    problems = password_problems(password, email, name)
    if problems:
        raise ValueError("Password is too weak. It needs " + ", ".join(problems) + ".")
    return password


class RegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def _strong(self):
        check_password(self.password, str(self.email), self.name)
        return self


class ChangePasswordRequest(BaseModel):
    current_password: Optional[str] = Field(default=None, max_length=128)
    new_password: str = Field(min_length=1, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class GoogleSignIn(BaseModel):
    credential: str = Field(min_length=20, max_length=5000)



class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    report_id: int
    messages: list[ChatMessage] = Field(min_length=1, max_length=40)
