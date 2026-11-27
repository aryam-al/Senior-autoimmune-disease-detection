"""Verify "Sign in with Google" ID tokens.

Google gives the frontend a signed JWT (ID token). The backend checks its
signature against the provider's public keys, its audience (our client ID),
issuer and expiry, then trusts the email inside it.
"""

import time

import httpx
from jose import JWTError, jwt

import config

PROVIDERS = {
    "google": {
        "jwks_url": "https://www.googleapis.com/oauth2/v3/certs",
        "issuers": ["https://accounts.google.com", "accounts.google.com"],
        "client_id": lambda: config.GOOGLE_CLIENT_ID,
    },
}

_jwks_cache: dict[str, tuple[float, list]] = {}
_JWKS_TTL = 60 * 60  # providers rotate keys rarely; refresh hourly


class SocialAuthError(Exception):
    pass


def _keys(provider: str, force: bool = False) -> list:
    cached = _jwks_cache.get(provider)
    if cached and not force and time.time() - cached[0] < _JWKS_TTL:
        return cached[1]
    try:
        r = httpx.get(PROVIDERS[provider]["jwks_url"], timeout=10)
        r.raise_for_status()
        keys = r.json()["keys"]
    except (httpx.HTTPError, KeyError, ValueError):
        raise SocialAuthError(f"Could not reach {provider.title()} to check the sign-in. Try again.")
    _jwks_cache[provider] = (time.time(), keys)
    return keys


def verify_id_token(provider: str, token: str) -> dict:
    """Return {"email", "name"} from a valid ID token, or raise SocialAuthError."""
    p = PROVIDERS[provider]
    client_id = p["client_id"]()
    if not client_id:
        raise SocialAuthError(f"Sign in with {provider.title()} is not set up on this server.")

    try:
        kid = jwt.get_unverified_header(token).get("kid")
    except JWTError:
        raise SocialAuthError("The sign-in token is not valid.")

    key = next((k for k in _keys(provider) if k.get("kid") == kid), None)
    if key is None:  # keys may have rotated since we cached them
        key = next((k for k in _keys(provider, force=True) if k.get("kid") == kid), None)
    if key is None:
        raise SocialAuthError("The sign-in token is not valid.")

    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=client_id,
            options={"verify_at_hash": False},
        )
    except JWTError:
        raise SocialAuthError("The sign-in token is not valid or has expired.")

    if claims.get("iss") not in p["issuers"]:
        raise SocialAuthError("The sign-in token is not valid.")

    email = (claims.get("email") or "").strip().lower()
    verified = claims.get("email_verified") in (True, "true")
    if not email or not verified:
        raise SocialAuthError(f"Your {provider.title()} account did not share a verified email address.")

    return {"email": email, "name": claims.get("name")}
