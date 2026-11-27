# ImmunoAI Backend

FastAPI backend for autoimmune disease classification. Every analysis runs two models:

| Model | Where it runs | Test accuracy (same 400 rows) |
|---|---|---|
| XGBoost | inside this backend (CPU) | 94.25% |
| Qwen2.5-7B-Instruct, QLoRA fine-tuned | separate GPU service (`qwen_service/`) | 93.25% |

If the Qwen service is off, the API still answers with XGBoost and marks Qwen as unavailable.

## Run locally

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env           # macOS/Linux: cp .env.example .env  - then set SECRET_KEY
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

## Turn on Qwen

Qwen runs on this computer's CPU with llama.cpp - free, no GPU needed. One-time setup in `qwen_local/README.md`
(make the GGUF file in Colab, download it, run `QWEN_LOCAL_SETUP.bat`). `START.bat` then starts it with the app.

GPU alternatives (Colab, Modal) are in `qwen_service/README.md`.

`GET /health` shows whether Qwen is online.

## Turn on "Continue with Google"

1. Go to https://console.cloud.google.com → create a project → **APIs & Services → OAuth consent screen** (External, add your email as a test user).
2. **Credentials → Create credentials → OAuth client ID → Web application**.
3. Under **Authorized JavaScript origins** add `http://localhost:3000` and `http://localhost`.
4. Copy the client ID into `.env` as `GOOGLE_CLIENT_ID=...` and restart the backend. The button appears on the sign-in page.

Accounts created with Google have no password; if the email already has an account, Google signs into that account.

## Endpoints

| Method | Path | Auth | What it does |
|---|---|---|---|
| GET | `/health` | – | API, XGBoost and Qwen status |
| GET | `/models` | – | Model names and test metrics |
| GET | `/features` | – | The 23 input fields (for the manual-entry form) |
| POST | `/register`, `/login` | – | Returns `token`, `name`, `email` |
| GET | `/auth/providers` | – | Google client ID if Google sign-in is enabled |
| POST | `/auth/google` | – | Body `{"credential": "<Google ID token>"}`; returns `token`, `name`, `email` |
| GET | `/me` | ✓ | Current user (`has_password` is false for Google-only accounts) |
| POST | `/me/password` | ✓ | `{"current_password", "new_password"}` - change (or, for a Google-only account, set) the password |
| POST | `/predict` | ✓ | Runs the chosen models on the values given (any may be missing) + other results; saves to history |
| POST | `/extract-pdfs` | ✓ | Several PDFs (`files`, up to 10) of one patient: values combined, conflicts reported, other results listed |
| POST | `/extract-pdf` | ✓ | One PDF; returns values found + missing list + other results (no prediction) |
| POST | `/upload-pdf` | ✓ | Reads one PDF and predicts with whatever was found |
| GET | `/history`, `/history/{id}` | ✓ | List / full result of past analyses |
| POST | `/chat` | ✓ | `{"report_id", "messages"}` - ask Qwen about one analysis; the answer is streamed as text (local Qwen only) |
| DELETE | `/history/{id}`, `/history` | ✓ | Delete one / all |

Frontend flow: `/extract-pdfs` → the user checks and completes the values → `/predict`.

**Missing values** are never filled in. Send `null` (or leave the field out). XGBoost then uses
`models/xgboost_missing_model.json`, trained on the same 2,000 rows with values randomly hidden
(see `training/train_xgboost_missing.py`); with all 23 values the main model is used. Qwen is simply
not shown the missing lines. At least one lab result is required.

**Passwords** need 10+ characters, upper and lowercase letters, a number and a symbol, and must not be
a common password or contain the user's name or email (`schemas.password_problems`).

### `/predict` body

```json
{"Age": 57, "Gender": "Female", "ESR": 8, "CRP": 0.4, "RF": 1, "Anti_CCP": 1, "HLA_B27": 1, "ANA": 1,
 "Anti_Ro": 1, "Anti_La": 1, "Anti_dsDNA": 1, "Anti_Sm": 1, "C3": 53, "C4": 5, "ASCA": 0, "Anti_CBir1": 0,
 "Anti_OmpC": 0, "pANCA": 0, "EMA": 0, "DGP": 0, "Anti_tTG": null, "Anti_TPO": null, "Anti_SMA": null,
 "source_filename": "optional.pdf", "models": ["xgboost", "qwen"],
 "extra": [{"name": "Hemoglobin", "value": "10.9", "unit": "g/dL", "flag": "low"}]}
```

Units: ESR mm/hr, CRP mg/L, C3 and C4 mg/dL. Markers: 1 = positive, 0 = negative, null = not tested.
Gender: "Male"/"Female" (or 0/1). `extra` = other lab results (not model inputs): added to Qwen's prompt
under "Other lab results" and to the chat; XGBoost cannot use them.

### `/predict` response (short)

```json
{
  "id": "12", "created_at": "...", "input": {...},
  "xgboost": {"available": true, "disease": "...", "confidence": 98.1,
              "probabilities": [{"disease": "...", "probability": 98.1}, ...],
              "top_features": [{"feature": "Anti_Sm", "label": "Anti-Sm", "value": 1, "impact": 2.31}, ...]},
  "qwen":    {"available": true, "disease": "...", "confidence": 97.4, "probabilities": [...], "raw_output": "..."},
  "agreement": true,
  "disclaimer": "..."
}
```

## Project layout

```
main.py            API routes
model.py           XGBoost predictor + per-patient feature impact (TreeSHAP)
qwen_client.py     Calls the Qwen GPU service
pdf_parser.py      Reads lab values from PDFs (never invents missing values)
schemas.py         Input validation
features.py        Feature list, units, ranges, display names
auth.py            bcrypt passwords + JWT
social_auth.py     Verifies Google ID tokens
database.py        SQLite: users, report_history (adds new columns to old databases)
config.py          Settings from .env
models/            xgboost_model.json, xgboost_meta.json
training/          train_xgboost.py - rebuilds the model with the Qwen 2000/400 split
qwen_llamacpp.py   Talks to Qwen running on the CPU (llama-server)
chat.py            "Ask about this result" chat: prompt with the report + safety rules, streamed answers
qwen_local/        CPU setup: Colab conversion notebook, setup + run scripts
qwen_service/      GPU options: shared prompt (qwen_core.py), Colab and Modal runners
tests/             pip install pytest, then: python -m pytest -q
```

## Retrain XGBoost

```bash
python training/train_xgboost.py           # main model (the reported results)
python training/train_xgboost_missing.py   # version used when values are missing
```

Accuracy on the same 400 test rows with a share of the values hidden at random:

| Values hidden | Main XGBoost | Missing-aware XGBoost |
|---|---|---|
| 0% | 94.25% | 96.00% |
| 10% | 73.75% | 93.75% |
| 20% | 56.25% | 92.00% |
| 30% | 42.75% | 87.50% |
| 45% | 29.25% | 81.50% |

Uses `data/Autoimmune_Dataset_Junior_Balanced_clean.csv` and the exact split from the Qwen fine-tuning notebook (seed 42).

## Notes

- Accounts from the old `users.json` are copied into the database on first start; the file itself is not changed.
- Old history rows (made by the previous Random Forest version) still show in the history list with their summary.
- Qwen was fine-tuned on prompts with all 23 values and no other results. A prompt with missing lines or an
  "Other lab results" section is new to it, so its answers on such prompts have not been measured.
- Results come from research models trained on a limited dataset and are not a medical diagnosis.
