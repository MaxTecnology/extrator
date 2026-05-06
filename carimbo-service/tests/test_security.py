from fastapi import HTTPException
from starlette.requests import Request

from app.config import Settings
from app.security import (
    _RATE_LIMIT_BUCKETS,
    _api_key_id,
    _parse_api_keys,
    enforce_api_key_access,
)


def _build_request(*, path: str = "/teste", headers: dict[str, str] | None = None) -> Request:
    scope_headers = []
    for key, value in (headers or {}).items():
        scope_headers.append((key.lower().encode("latin-1"), value.encode("latin-1")))
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": path,
        "raw_path": path.encode("latin-1"),
        "query_string": b"",
        "headers": scope_headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


def test_parse_api_keys_splits_and_deduplicates() -> None:
    parsed = _parse_api_keys(" abc , def,abc,\nxyz ")
    assert parsed == ["abc", "def", "xyz"]


def test_enforce_api_key_access_works_when_disabled() -> None:
    _RATE_LIMIT_BUCKETS.clear()
    settings = Settings(api_key_required=False)
    request = _build_request()
    assert enforce_api_key_access(request=request, settings=settings) == "auth_disabled"


def test_enforce_api_key_access_can_force_requirement() -> None:
    _RATE_LIMIT_BUCKETS.clear()
    settings = Settings(api_key_required=False, api_keys="forcada")
    request = _build_request(headers={"X-API-Key": "forcada"})
    key_id = enforce_api_key_access(request=request, settings=settings, force_require=True)
    assert key_id == _api_key_id("forcada")


def test_enforce_api_key_access_fails_when_required_without_keys() -> None:
    _RATE_LIMIT_BUCKETS.clear()
    settings = Settings(api_key_required=True, api_keys="")
    request = _build_request()
    try:
        enforce_api_key_access(request=request, settings=settings)
        assert False, "Expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 503


def test_enforce_api_key_access_fails_when_header_missing() -> None:
    _RATE_LIMIT_BUCKETS.clear()
    settings = Settings(api_key_required=True, api_keys="chave-1")
    request = _build_request()
    try:
        enforce_api_key_access(request=request, settings=settings)
        assert False, "Expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 401


def test_enforce_api_key_access_fails_when_header_invalid() -> None:
    _RATE_LIMIT_BUCKETS.clear()
    settings = Settings(api_key_required=True, api_keys="chave-1")
    request = _build_request(headers={"X-API-Key": "chave-invalida"})
    try:
        enforce_api_key_access(request=request, settings=settings)
        assert False, "Expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 403


def test_enforce_api_key_access_accepts_valid_key() -> None:
    _RATE_LIMIT_BUCKETS.clear()
    key = "chave-valida"
    settings = Settings(api_key_required=True, api_keys=key)
    request = _build_request(headers={"X-API-Key": key})
    key_id = enforce_api_key_access(request=request, settings=settings)
    assert key_id == _api_key_id(key)


def test_enforce_api_key_access_applies_rate_limit() -> None:
    _RATE_LIMIT_BUCKETS.clear()
    key = "chave-rate-limit"
    settings = Settings(
        api_key_required=True,
        api_keys=key,
        api_rate_limit_enabled=True,
        api_rate_limit_requests=1,
        api_rate_limit_window_seconds=60,
    )
    request = _build_request(headers={"X-API-Key": key})
    enforce_api_key_access(request=request, settings=settings)
    try:
        enforce_api_key_access(request=request, settings=settings)
        assert False, "Expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 429
