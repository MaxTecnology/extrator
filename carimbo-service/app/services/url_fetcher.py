from __future__ import annotations

import ipaddress
import logging
import random
import socket
import time
from dataclasses import dataclass
from typing import Iterable
from urllib import error, parse, request


logger = logging.getLogger(__name__)

RETRIABLE_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}


class SourceUrlFetchError(RuntimeError):
    def __init__(
        self,
        *,
        status_code: int,
        detail: str,
        code: str = "source_url_error",
        retryable: bool = False,
        stage: str = "source_url_download",
        upstream_http_status: int | None = None,
        attempts_used: int | None = None,
        max_attempts: int | None = None,
    ):
        super().__init__(detail)
        self.status_code = int(status_code)
        self.detail = detail
        self.code = code
        self.retryable = bool(retryable)
        self.stage = stage
        self.upstream_http_status = (
            int(upstream_http_status) if upstream_http_status is not None else None
        )
        self.attempts_used = int(attempts_used) if attempts_used is not None else None
        self.max_attempts = int(max_attempts) if max_attempts is not None else None

    def to_detail_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "codigo": self.code,
            "detalhe": self.detail,
            "etapa": self.stage,
            "retryable": self.retryable,
        }
        if self.upstream_http_status is not None:
            payload["origem_http_status"] = self.upstream_http_status
        if self.attempts_used is not None:
            payload["tentativas_usadas"] = self.attempts_used
        if self.max_attempts is not None:
            payload["tentativas_maximas"] = self.max_attempts
        return payload


@dataclass(slots=True)
class SourceUrlFetchResult:
    file_bytes: bytes
    content_type: str
    final_url: str


def sanitize_source_url_for_log(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return "<vazio>"
    try:
        parsed = parse.urlsplit(raw)
    except Exception:
        return "<url_invalida>"
    safe_path = parsed.path or "/"
    return parse.urlunsplit((parsed.scheme, parsed.netloc, safe_path, "", ""))


def _normalize_allowed_domains(raw_domains: str) -> tuple[str, ...]:
    parts = []
    for item in (raw_domains or "").split(","):
        candidate = item.strip().lower().lstrip(".")
        if candidate:
            parts.append(candidate)
    # remove duplicidade preservando ordem
    unique: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part in seen:
            continue
        seen.add(part)
        unique.append(part)
    return tuple(unique)


def _host_allowed(host: str, allowed_domains: Iterable[str]) -> bool:
    host_l = (host or "").strip().lower().rstrip(".")
    if not host_l:
        return False
    for allowed in allowed_domains:
        if host_l == allowed:
            return True
        if host_l.endswith(f".{allowed}"):
            return True
    return False


def _validate_host_is_not_local_or_private(host: str) -> None:
    host_l = (host or "").strip().lower().rstrip(".")
    if host_l in {"localhost", "localhost.localdomain"}:
        raise SourceUrlFetchError(
            status_code=422,
            detail="arquivo_url inválida: host local não é permitido",
            code="source_url_host_local_not_allowed",
        )
    try:
        parsed_ip = ipaddress.ip_address(host_l)
    except ValueError:
        return

    if (
        parsed_ip.is_private
        or parsed_ip.is_loopback
        or parsed_ip.is_link_local
        or parsed_ip.is_multicast
        or parsed_ip.is_unspecified
        or parsed_ip.is_reserved
    ):
        raise SourceUrlFetchError(
            status_code=422,
            detail="arquivo_url inválida: IP local/privado não é permitido",
            code="source_url_ip_private_not_allowed",
        )


def validate_source_url(
    *,
    source_url: str,
    allowed_domains_csv: str,
    require_https: bool = True,
) -> str:
    url = (source_url or "").strip()
    if not url:
        raise SourceUrlFetchError(
            status_code=422,
            detail="arquivo_url não pode ser vazio",
            code="source_url_empty",
        )

    try:
        parsed = parse.urlsplit(url)
    except Exception as exc:
        raise SourceUrlFetchError(
            status_code=422,
            detail="arquivo_url inválida",
            code="source_url_invalid",
        ) from exc

    scheme = parsed.scheme.lower()
    if require_https and scheme != "https":
        raise SourceUrlFetchError(
            status_code=422,
            detail="arquivo_url deve usar HTTPS",
            code="source_url_scheme_https_required",
        )
    if not require_https and scheme not in {"http", "https"}:
        raise SourceUrlFetchError(
            status_code=422,
            detail="arquivo_url deve usar HTTP/HTTPS",
            code="source_url_scheme_not_supported",
        )

    hostname = parsed.hostname or ""
    if not hostname:
        raise SourceUrlFetchError(
            status_code=422,
            detail="arquivo_url sem host válido",
            code="source_url_host_invalid",
        )

    _validate_host_is_not_local_or_private(hostname)

    allowed_domains = _normalize_allowed_domains(allowed_domains_csv)
    if allowed_domains and not _host_allowed(hostname, allowed_domains):
        raise SourceUrlFetchError(
            status_code=422,
            detail=(
                "arquivo_url fora dos domínios permitidos. "
                f"Domínios aceitos: {', '.join(allowed_domains)}"
            ),
            code="source_url_domain_not_allowed",
        )

    return url


def download_file_from_source_url(
    *,
    source_url: str,
    timeout_seconds: int,
    max_file_size_mb: int,
    user_agent: str,
    allowed_domains_csv: str,
    require_https: bool = True,
    retry_attempts: int = 2,
    retry_backoff_seconds: float = 1.0,
    retry_jitter_seconds: float = 0.3,
) -> SourceUrlFetchResult:
    safe_url = validate_source_url(
        source_url=source_url,
        allowed_domains_csv=allowed_domains_csv,
        require_https=require_https,
    )

    timeout = max(3, int(timeout_seconds))
    max_bytes = max(1, int(max_file_size_mb)) * 1024 * 1024
    max_attempts = max(1, int(retry_attempts) + 1)
    backoff = max(0.1, float(retry_backoff_seconds))
    jitter = max(0.0, float(retry_jitter_seconds))
    headers = {
        "User-Agent": (user_agent or "carimbo-service/1.0"),
        "Accept": "application/pdf,image/*;q=0.9,*/*;q=0.8",
    }
    for attempt_index in range(max_attempts):
        attempt_number = attempt_index + 1
        req = request.Request(url=safe_url, method="GET", headers=headers)

        try:
            with request.urlopen(req, timeout=timeout) as response:
                content_type_header = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                content_length_raw = (response.headers.get("Content-Length") or "").strip()
                if content_length_raw.isdigit() and int(content_length_raw) > max_bytes:
                    raise SourceUrlFetchError(
                        status_code=413,
                        detail=f"Arquivo da URL excede o limite de {max_file_size_mb}MB",
                        code="source_url_file_too_large",
                        retryable=False,
                        attempts_used=attempt_number,
                        max_attempts=max_attempts,
                    )

                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise SourceUrlFetchError(
                            status_code=413,
                            detail=f"Arquivo da URL excede o limite de {max_file_size_mb}MB",
                            code="source_url_file_too_large",
                            retryable=False,
                            attempts_used=attempt_number,
                            max_attempts=max_attempts,
                        )
                    chunks.append(chunk)

                if total <= 0:
                    raise SourceUrlFetchError(
                        status_code=422,
                        detail="arquivo_url não retornou conteúdo",
                        code="source_url_empty_content",
                        retryable=False,
                        attempts_used=attempt_number,
                        max_attempts=max_attempts,
                    )

                final_url = str(getattr(response, "url", safe_url) or safe_url)
                return SourceUrlFetchResult(
                    file_bytes=b"".join(chunks),
                    content_type=content_type_header,
                    final_url=final_url,
                )

        except SourceUrlFetchError:
            raise
        except error.HTTPError as exc:
            upstream_status = int(exc.code)
            _ = exc.read().decode("utf-8", errors="ignore")

            if upstream_status in {401, 403, 404}:
                raise SourceUrlFetchError(
                    status_code=422,
                    detail=(
                        f"arquivo_url não pôde ser baixada (HTTP {upstream_status}). "
                        "Verifique se o link ainda está válido."
                    ),
                    code="source_url_auth_or_not_found",
                    retryable=False,
                    upstream_http_status=upstream_status,
                    attempts_used=attempt_number,
                    max_attempts=max_attempts,
                ) from exc

            is_retryable = upstream_status in RETRIABLE_HTTP_CODES
            if is_retryable and attempt_number < max_attempts:
                wait_seconds = (backoff * (2**attempt_index)) + (random.random() * jitter)
                logger.warning(
                    "Falha HTTP ao baixar arquivo da URL (status=%s, tentativa %s/%s). Retry em %.2fs.",
                    upstream_status,
                    attempt_number,
                    max_attempts,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                continue

            raise SourceUrlFetchError(
                status_code=502,
                detail=f"Falha ao baixar arquivo da URL (HTTP {upstream_status}).",
                code="source_url_upstream_http_error",
                retryable=is_retryable,
                upstream_http_status=upstream_status,
                attempts_used=attempt_number,
                max_attempts=max_attempts,
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            if attempt_number < max_attempts:
                wait_seconds = (backoff * (2**attempt_index)) + (random.random() * jitter)
                logger.warning(
                    "Timeout ao baixar arquivo da URL (tentativa %s/%s). Retry em %.2fs.",
                    attempt_number,
                    max_attempts,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                continue
            raise SourceUrlFetchError(
                status_code=502,
                detail="Falha de conexão/timeout ao baixar arquivo da URL",
                code="source_url_timeout",
                retryable=True,
                attempts_used=attempt_number,
                max_attempts=max_attempts,
            ) from exc
        except error.URLError as exc:
            if attempt_number < max_attempts:
                wait_seconds = (backoff * (2**attempt_index)) + (random.random() * jitter)
                logger.warning(
                    "Falha de rede ao baixar arquivo da URL (tentativa %s/%s): %s. Retry em %.2fs.",
                    attempt_number,
                    max_attempts,
                    exc,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                continue
            raise SourceUrlFetchError(
                status_code=502,
                detail="Falha de rede ao baixar arquivo da URL",
                code="source_url_network_error",
                retryable=True,
                attempts_used=attempt_number,
                max_attempts=max_attempts,
            ) from exc

    raise SourceUrlFetchError(
        status_code=502,
        detail="Falha ao baixar arquivo da URL",
        code="source_url_unknown_error",
        retryable=True,
        attempts_used=max_attempts,
        max_attempts=max_attempts,
    )
