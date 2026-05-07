from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Iterable
from urllib import error, parse, request


class SourceUrlFetchError(RuntimeError):
    def __init__(self, *, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = int(status_code)
        self.detail = detail


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
        )


def validate_source_url(
    *,
    source_url: str,
    allowed_domains_csv: str,
    require_https: bool = True,
) -> str:
    url = (source_url or "").strip()
    if not url:
        raise SourceUrlFetchError(status_code=422, detail="arquivo_url não pode ser vazio")

    try:
        parsed = parse.urlsplit(url)
    except Exception as exc:
        raise SourceUrlFetchError(status_code=422, detail="arquivo_url inválida") from exc

    scheme = parsed.scheme.lower()
    if require_https and scheme != "https":
        raise SourceUrlFetchError(
            status_code=422,
            detail="arquivo_url deve usar HTTPS",
        )
    if not require_https and scheme not in {"http", "https"}:
        raise SourceUrlFetchError(
            status_code=422,
            detail="arquivo_url deve usar HTTP/HTTPS",
        )

    hostname = parsed.hostname or ""
    if not hostname:
        raise SourceUrlFetchError(status_code=422, detail="arquivo_url sem host válido")

    _validate_host_is_not_local_or_private(hostname)

    allowed_domains = _normalize_allowed_domains(allowed_domains_csv)
    if allowed_domains and not _host_allowed(hostname, allowed_domains):
        raise SourceUrlFetchError(
            status_code=422,
            detail=(
                "arquivo_url fora dos domínios permitidos. "
                f"Domínios aceitos: {', '.join(allowed_domains)}"
            ),
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
) -> SourceUrlFetchResult:
    safe_url = validate_source_url(
        source_url=source_url,
        allowed_domains_csv=allowed_domains_csv,
        require_https=require_https,
    )

    timeout = max(3, int(timeout_seconds))
    max_bytes = max(1, int(max_file_size_mb)) * 1024 * 1024
    headers = {
        "User-Agent": (user_agent or "carimbo-service/1.0"),
        "Accept": "application/pdf,image/*;q=0.9,*/*;q=0.8",
    }
    req = request.Request(url=safe_url, method="GET", headers=headers)

    try:
        with request.urlopen(req, timeout=timeout) as response:
            content_type_header = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            content_length_raw = (response.headers.get("Content-Length") or "").strip()
            if content_length_raw.isdigit() and int(content_length_raw) > max_bytes:
                raise SourceUrlFetchError(
                    status_code=413,
                    detail=f"Arquivo da URL excede o limite de {max_file_size_mb}MB",
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
                    )
                chunks.append(chunk)

            if total <= 0:
                raise SourceUrlFetchError(
                    status_code=422,
                    detail="arquivo_url não retornou conteúdo",
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
        body = exc.read().decode("utf-8", errors="ignore")
        if exc.code in {401, 403, 404}:
            raise SourceUrlFetchError(
                status_code=422,
                detail=(
                    f"arquivo_url não pôde ser baixada (HTTP {exc.code}). "
                    "Verifique se o link ainda está válido."
                ),
            ) from exc
        raise SourceUrlFetchError(
            status_code=502,
            detail=f"Falha ao baixar arquivo da URL (HTTP {exc.code}).",
        ) from exc
    except (TimeoutError, socket.timeout, error.URLError) as exc:
        raise SourceUrlFetchError(
            status_code=502,
            detail="Falha de conexão/timeout ao baixar arquivo da URL",
        ) from exc
