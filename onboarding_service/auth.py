"""
SECURITY: authentication for the onboarding FastAPI service.

The service previously had no auth at all and trusted a `user_id` in the request body,
so anyone on the internet could overwrite any user's onboarding profile and burn LLM credits.

Now every request must carry the same JWT access token the Django API issues
(`Authorization: Bearer <access_token>`). It is verified with Django's SECRET_KEY
(SimpleJWT signs HS256 with it), and the user id is taken from the token.

=> The SECRET_KEY environment variable MUST be identical for Django and this service.
"""
import os

import jwt
from fastapi import Header, HTTPException


def current_user_id(authorization: str = Header(default="")) -> int:
    secret = os.environ.get("SECRET_KEY")
    if not secret:
        raise HTTPException(status_code=503, detail="Auth not configured")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if payload.get("token_type") != "access":
        raise HTTPException(status_code=401, detail="Access token required")
    try:
        return int(payload["user_id"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token")
