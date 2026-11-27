"""Feature definitions shared by training, the API schema and the PDF parser."""

import pandas as pd

# Order must match training data exactly.
FEATURES = [
    "Age", "Gender", "ESR", "CRP", "RF", "Anti_CCP", "HLA_B27",
    "ANA", "Anti_Ro", "Anti_La", "Anti_dsDNA", "Anti_Sm",
    "C3", "C4", "ASCA", "Anti_CBir1", "Anti_OmpC", "pANCA",
    "EMA", "DGP", "Anti_tTG", "Anti_TPO", "Anti_SMA",
]

# Numeric lab values: (unit, plausible min, plausible max).
# Ranges are wide on purpose: they reject typos and wrong units, not unusual patients.
CONTINUOUS = {
    "Age": ("years", 1, 120),
    "ESR": ("mm/hr", 0, 200),
    "CRP": ("mg/L", 0, 500),
    "C3": ("mg/dL", 10, 400),
    "C4": ("mg/dL", 1, 150),
}

# Positive / negative antibody and genetic markers (1 = positive, 0 = negative).
BINARY = [f for f in FEATURES if f not in CONTINUOUS and f != "Gender"]

# Human-readable names for the frontend and SHAP output.
DISPLAY_NAMES = {
    "Age": "Age",
    "Gender": "Gender",
    "ESR": "ESR",
    "CRP": "CRP",
    "RF": "Rheumatoid Factor (RF)",
    "Anti_CCP": "Anti-CCP",
    "HLA_B27": "HLA-B27",
    "ANA": "ANA",
    "Anti_Ro": "Anti-Ro (SSA)",
    "Anti_La": "Anti-La (SSB)",
    "Anti_dsDNA": "Anti-dsDNA",
    "Anti_Sm": "Anti-Sm",
    "C3": "Complement C3",
    "C4": "Complement C4",
    "ASCA": "ASCA",
    "Anti_CBir1": "Anti-CBir1",
    "Anti_OmpC": "Anti-OmpC",
    "pANCA": "pANCA",
    "EMA": "EMA",
    "DGP": "DGP",
    "Anti_tTG": "Anti-tTG",
    "Anti_TPO": "Anti-TPO",
    "Anti_SMA": "Anti-SMA",
}


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a frame with the 23 features into the numeric matrix the model expects."""
    X = df[FEATURES].copy()
    X["Gender"] = X["Gender"].map({"Male": 0, "Female": 1}).astype(float)
    return X.astype(float)
