from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from app.config import Settings, get_settings


logger = logging.getLogger(__name__)

_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_BUCKETS: dict[str, deque[float]] = defaultdict(deque)


def _parse_api_keys(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    tokens = [token.strip() for token in raw_value.replace("\n", ",").split(",")]
    keys: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if not token or token in seen:
            continue
        seen.add(token)
        keys.append(token)
    return keys


def _api_key_id(api_key: str) -> str:
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return digest[:12]


def _authenticate_api_key(
    provided_key: str,
    *,
    allowed_keys: list[str],
) -> Optional[str]:
    for candidate in allowed_keys:
        if hmac.compare_digest(provided_key, candidate):
            return _api_key_id(candidate)
    return None


def _apply_rate_limit(*, key_id: str, settings: Settings) -> None:
    if not settings.api_rate_limit_enabled:
        return

    limit = max(1, int(settings.api_rate_limit_requests))
    window = max(1, int(settings.api_rate_limit_window_seconds))
    now = time.monotonic()
    cutoff = now - window

    with _RATE_LIMIT_LOCK:
        bucket = _RATE_LIMIT_BUCKETS[key_id]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            retry_after = max(1, int(round(bucket[0] + window - now)))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="limite_de_requisicoes_excedido",
                headers={"Retry-After": str(retry_after)},
            )
        bucket.append(now)


def enforce_api_key_access(
    *,
    request: Request,
    settings: Settings,
    apply_rate_limit: bool = True,
    force_require: bool = False,
) -> str:
    if not settings.api_key_required and not force_require:
        return "auth_disabled"

    allowed_keys = _parse_api_keys(settings.api_keys)
    if not allowed_keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="api_key_required_ativado_sem_api_keys_configuradas",
        )

    header_name = settings.api_key_header_name
    provided_key = (request.headers.get(header_name) or "").strip()
    if not provided_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"cabecalho_{header_name}_obrigatorio",
        )

    key_id = _authenticate_api_key(provided_key, allowed_keys=allowed_keys)
    if key_id is None:
        logger.warning(
            "Acesso negado por API key inválida. path=%s client=%s",
            request.url.path,
            request.client.host if request.client else "unknown",
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="api_key_invalida",
        )

    if apply_rate_limit:
        _apply_rate_limit(key_id=key_id, settings=settings)

    return key_id


def require_api_key_access(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> str:
    return enforce_api_key_access(request=request, settings=settings, apply_rate_limit=True)
