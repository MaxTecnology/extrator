from __future__ import annotations

from io import BytesIO
from urllib import error

import pytest

from app.services.url_fetcher import (
    SourceUrlFetchError,
    download_file_from_source_url,
    sanitize_source_url_for_log,
    validate_source_url,
)


class _FakeResponse:
    def __init__(self, chunks: list[bytes], headers: dict[str, str], url: str):
        self._chunks = list(chunks)
        self.headers = headers
        self.url = url

    def read(self, _size: int = -1) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return False


def test_sanitize_source_url_hides_query_string() -> None:
    safe = sanitize_source_url_for_log(
        "https://tenant.sharepoint.com/path/file.pdf?tempauth=abc123&x=1"
    )
    assert safe == "https://tenant.sharepoint.com/path/file.pdf"


def test_validate_source_url_accepts_sharepoint_subdomain() -> None:
    validated = validate_source_url(
        source_url="https://digitalpointltda-my.sharepoint.com/path/file.pdf",
        allowed_domains_csv="sharepoint.com",
        require_https=True,
    )
    assert validated.startswith("https://digitalpointltda-my.sharepoint.com/")


def test_validate_source_url_rejects_non_https_when_required() -> None:
    with pytest.raises(SourceUrlFetchError) as exc:
        validate_source_url(
            source_url="http://digitalpointltda-my.sharepoint.com/path/file.pdf",
            allowed_domains_csv="sharepoint.com",
            require_https=True,
        )
    assert exc.value.status_code == 422


def test_validate_source_url_rejects_domain_outside_allowlist() -> None:
    with pytest.raises(SourceUrlFetchError) as exc:
        validate_source_url(
            source_url="https://example.com/file.pdf",
            allowed_domains_csv="sharepoint.com",
            require_https=True,
        )
    assert exc.value.status_code == 422


def test_download_file_from_source_url_success(monkeypatch) -> None:
    def fake_urlopen(req, timeout):  # noqa: ANN001
        assert timeout == 15
        assert req.full_url.startswith("https://digitalpointltda-my.sharepoint.com/")
        return _FakeResponse(
            chunks=[b"%PDF-", b"123", b""],
            headers={"Content-Type": "application/pdf", "Content-Length": "8"},
            url=req.full_url,
        )

    monkeypatch.setattr("app.services.url_fetcher.request.urlopen", fake_urlopen)
    result = download_file_from_source_url(
        source_url="https://digitalpointltda-my.sharepoint.com/path/file.pdf",
        timeout_seconds=15,
        max_file_size_mb=20,
        user_agent="carimbo-test/1.0",
        allowed_domains_csv="sharepoint.com",
        require_https=True,
    )
    assert result.content_type == "application/pdf"
    assert result.file_bytes.startswith(b"%PDF-123")


def test_download_file_from_source_url_rejects_content_length_too_large(monkeypatch) -> None:
    def fake_urlopen(req, timeout):  # noqa: ANN001
        return _FakeResponse(
            chunks=[b"%PDF-123", b""],
            headers={"Content-Type": "application/pdf", "Content-Length": str(25 * 1024 * 1024)},
            url=req.full_url,
        )

    monkeypatch.setattr("app.services.url_fetcher.request.urlopen", fake_urlopen)
    with pytest.raises(SourceUrlFetchError) as exc:
        download_file_from_source_url(
            source_url="https://digitalpointltda-my.sharepoint.com/path/file.pdf",
            timeout_seconds=15,
            max_file_size_mb=20,
            user_agent="carimbo-test/1.0",
            allowed_domains_csv="sharepoint.com",
            require_https=True,
        )
    assert exc.value.status_code == 413


def test_download_file_from_source_url_maps_404_to_422(monkeypatch) -> None:
    def fake_urlopen(req, timeout):  # noqa: ANN001
        raise error.HTTPError(
            url=req.full_url,
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=BytesIO(b"404"),
        )

    monkeypatch.setattr("app.services.url_fetcher.request.urlopen", fake_urlopen)
    with pytest.raises(SourceUrlFetchError) as exc:
        download_file_from_source_url(
            source_url="https://digitalpointltda-my.sharepoint.com/path/file.pdf",
            timeout_seconds=15,
            max_file_size_mb=20,
            user_agent="carimbo-test/1.0",
            allowed_domains_csv="sharepoint.com",
            require_https=True,
        )
    assert exc.value.status_code == 422
