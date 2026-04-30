from __future__ import annotations

import base64
import io
import json
import logging
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional
from urllib import error, request

from PIL import Image

from app.services.detector import BBoxTuple


logger = logging.getLogger(__name__)

UF_BRASIL = {
    "AC",
    "AL",
    "AP",
    "AM",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MT",
    "MS",
    "MG",
    "PA",
    "PB",
    "PR",
    "PE",
    "PI",
    "RJ",
    "RN",
    "RS",
    "RO",
    "RR",
    "SC",
    "SP",
    "SE",
    "TO",
}

CRM_NUM_UF = re.compile(r"(?P<num>\d{4,8})\s*[/-]?\s*(?P<uf>[A-Za-z]{2})")
CRM_UF_NUM = re.compile(r"(?P<uf>[A-Za-z]{2})\s*[/-]?\s*(?P<num>\d{4,8})")
CRM_NUMBER_WITH_SEPARATORS = re.compile(r"\d(?:[\d\.\- ]{2,14})\d")
HEADER_OBS_HINTS = (
    "CABECALHO",
    "PCMSO",
    "MEDICO RESPONSAVEL",
)


class GeminiServiceError(RuntimeError):
    pass


@dataclass(slots=True)
class BBoxCandidate:
    bbox: BBoxTuple
    score: float
    reason: str


@dataclass(slots=True)
class MedicoExtraction:
    nome: Optional[str]
    crm_numero: Optional[str]
    crm_uf: Optional[str]
    crm: Optional[str]
    confidence: float
    valido: bool
    observacoes: str


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return float(max(minimum, min(maximum, value)))


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _pil_to_png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _extract_first_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].lstrip()

    start = text.find("{")
    if start < 0:
        raise GeminiServiceError("Gemini não retornou JSON válido.")

    depth = 0
    in_string = False
    escape = False
    end = -1
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break

    if end < 0:
        raise GeminiServiceError("Gemini retornou JSON incompleto.")

    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError as exc:
        raise GeminiServiceError("Falha ao decodificar JSON do Gemini.") from exc


def _normalize_text_for_rule_checks(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_only = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", ascii_only.upper()).strip()


def probe_gemini_model(
    *,
    api_key: str,
    model: str,
    timeout_seconds: int,
    retry_attempts: int = 0,
) -> tuple[bool, str]:
    probe_image = Image.new("RGB", (8, 8), color=(255, 255, 255))
    prompt = """
Responda SOMENTE JSON válido:
{"ok": true}
""".strip()
    try:
        _call_gemini(
            api_key=api_key,
            model=model,
            prompt=prompt,
            image=probe_image,
            timeout_seconds=max(2, int(timeout_seconds)),
            retry_attempts=max(0, int(retry_attempts)),
            retry_backoff_seconds=0.7,
            retry_jitter_seconds=0.2,
        )
        return True, ""
    except GeminiServiceError as exc:
        return False, str(exc)[:260]


def _call_gemini(
    api_key: str,
    model: str,
    prompt: str,
    image: Image.Image,
    timeout_seconds: int,
    retry_attempts: int,
    retry_backoff_seconds: float,
    retry_jitter_seconds: float,
    usage_sink: Optional[list[dict[str, Any]]] = None,
    usage_stage: str = "",
) -> dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": _pil_to_png_base64(image),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    req = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    retriable_http_codes = {429, 500, 502, 503, 504}
    attempts = max(0, int(retry_attempts))
    backoff = max(0.05, float(retry_backoff_seconds))
    jitter = max(0.0, float(retry_jitter_seconds))
    last_error: Optional[Exception] = None
    started_at = time.monotonic()
    attempts_used = 0

    body = ""
    for attempt in range(attempts + 1):
        attempts_used = attempt + 1
        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
            break
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            last_error = exc
            should_retry = exc.code in retriable_http_codes and attempt < attempts
            if should_retry:
                sleep_seconds = (backoff * (2**attempt)) + (random.random() * jitter)
                logger.warning(
                    "Gemini HTTP %s (tentativa %s/%s). Retry em %.2fs.",
                    exc.code,
                    attempt + 1,
                    attempts + 1,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
                continue
            raise GeminiServiceError(
                f"Gemini HTTP {exc.code}: {body[:300]}"
            ) from exc
        except error.URLError as exc:
            last_error = exc
            should_retry = attempt < attempts
            if should_retry:
                sleep_seconds = (backoff * (2**attempt)) + (random.random() * jitter)
                logger.warning(
                    "Falha de conexão com Gemini (tentativa %s/%s): %s. Retry em %.2fs.",
                    attempt + 1,
                    attempts + 1,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
                continue
            raise GeminiServiceError(f"Falha de conexão com Gemini: {exc}") from exc
    else:
        raise GeminiServiceError(f"Falha inesperada no Gemini após retries: {last_error}")

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise GeminiServiceError("Resposta inválida do Gemini.") from exc

    usage = data.get("usageMetadata", {}) if isinstance(data, dict) else {}
    if usage_sink is not None:
        usage_sink.append(
            {
                "stage": usage_stage or "gemini",
                "model": model,
                "prompt_tokens": int(usage.get("promptTokenCount", 0) or 0),
                "output_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
                "total_tokens": int(usage.get("totalTokenCount", 0) or 0),
                "latency_ms": int(max(0, round((time.monotonic() - started_at) * 1000))),
                "attempts_used": int(max(1, attempts_used)),
            }
        )

    texts: list[str] = []
    for candidate in data.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text_part = part.get("text")
            if isinstance(text_part, str) and text_part.strip():
                texts.append(text_part.strip())

    if not texts:
        if data.get("promptFeedback"):
            raise GeminiServiceError(f"Gemini bloqueou a resposta: {data['promptFeedback']}")
        raise GeminiServiceError("Gemini não retornou texto.")

    return _extract_first_json_object(texts[0])


def _sanitize_bbox_candidates(
    raw_payload: dict[str, Any],
    width: int,
    height: int,
    max_candidates: int,
) -> list[BBoxCandidate]:
    raw_candidates: Any = raw_payload.get("candidatos") or raw_payload.get("candidates") or []
    if not isinstance(raw_candidates, list):
        return []

    candidates: list[BBoxCandidate] = []
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            continue

        bbox_data = candidate.get("bbox", candidate)
        if not isinstance(bbox_data, dict):
            continue

        try:
            x = int(round(float(bbox_data.get("x", 0))))
            y = int(round(float(bbox_data.get("y", 0))))
            w = int(round(float(bbox_data.get("w", 0))))
            h = int(round(float(bbox_data.get("h", 0))))
        except (TypeError, ValueError):
            continue

        score_value = candidate.get("score", candidate.get("confianca", 0.0))
        try:
            score = _clamp(float(score_value))
        except (TypeError, ValueError):
            score = 0.0

        reason = str(candidate.get("motivo", candidate.get("reason", ""))).strip()

        if w <= 0 or h <= 0:
            continue

        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        w = min(w, width - x)
        h = min(h, height - y)
        if w <= 0 or h <= 0:
            continue

        candidates.append(BBoxCandidate(bbox=(x, y, w, h), score=score, reason=reason))

    candidates.sort(key=lambda item: item.score, reverse=True)
    unique: list[BBoxCandidate] = []
    seen: set[tuple[int, int, int, int]] = set()
    for candidate in candidates:
        if candidate.bbox in seen:
            continue
        seen.add(candidate.bbox)
        unique.append(candidate)
        if len(unique) >= max_candidates:
            break
    return unique


def _extract_crm_fields(raw_payload: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    def _sanitize_crm_number(raw_value: Any) -> Optional[str]:
        if raw_value is None:
            return None
        digits = re.sub(r"\D", "", str(raw_value))
        if not digits:
            return None
        if 4 <= len(digits) <= 8:
            return digits
        return None

    def _extract_number_from_free_text(text: str) -> Optional[str]:
        if not text:
            return None
        # Prioriza blocos que já parecem um número CRM com separadores OCR (ex: 60.750).
        for match in CRM_NUMBER_WITH_SEPARATORS.finditer(text):
            digits = _sanitize_crm_number(match.group(0))
            if digits:
                return digits
        # Fallback: qualquer bloco contínuo de 4-8 dígitos.
        simple_match = re.search(r"\d{4,8}", text)
        if simple_match:
            return simple_match.group(0)
        return None

    crm_numero_raw = raw_payload.get("crm_numero")
    crm_uf_raw = raw_payload.get("crm_uf")
    crm_raw = raw_payload.get("crm")

    crm_numero = _sanitize_crm_number(crm_numero_raw)
    crm_uf = None if crm_uf_raw is None else str(crm_uf_raw).strip().upper()
    crm_text = None if crm_raw is None else str(crm_raw).strip().upper()

    if crm_numero and crm_uf:
        return crm_numero, crm_uf

    if crm_text:
        compact_crm_text = re.sub(r"[^A-Z0-9]+", " ", crm_text).strip()
        direct_match = CRM_NUM_UF.search(compact_crm_text)
        if direct_match:
            return _sanitize_crm_number(direct_match.group("num")), direct_match.group("uf").upper()
        reverse_match = CRM_UF_NUM.search(compact_crm_text)
        if reverse_match:
            return _sanitize_crm_number(reverse_match.group("num")), reverse_match.group("uf").upper()
        crm_num_from_text = _extract_number_from_free_text(crm_text)
        crm_uf_from_text = None
        uf_match = re.search(r"\b([A-Z]{2})\b", compact_crm_text)
        if uf_match:
            crm_uf_from_text = uf_match.group(1).upper()
        if crm_num_from_text and crm_uf_from_text:
            return crm_num_from_text, crm_uf_from_text

    combined = f"{crm_numero or ''} {crm_uf or ''}".strip()
    direct_match = CRM_NUM_UF.search(combined)
    if direct_match:
        return _sanitize_crm_number(direct_match.group("num")), direct_match.group("uf").upper()
    reverse_match = CRM_UF_NUM.search(combined)
    if reverse_match:
        return _sanitize_crm_number(reverse_match.group("num")), reverse_match.group("uf").upper()
    return crm_numero, crm_uf


def _normalize_medico_payload(raw_payload: dict[str, Any]) -> MedicoExtraction:
    nome_raw = raw_payload.get("nome")
    nome = None if nome_raw is None else " ".join(str(nome_raw).split())
    if nome == "":
        nome = None

    crm_numero, crm_uf = _extract_crm_fields(raw_payload)
    crm_numero = crm_numero if crm_numero else None
    crm_uf = crm_uf if crm_uf else None
    if crm_uf and crm_uf not in UF_BRASIL:
        crm_uf = None

    confidence_raw = raw_payload.get("confianca", raw_payload.get("score", 0.0))
    try:
        confidence = _clamp(float(confidence_raw))
    except (TypeError, ValueError):
        confidence = 0.0

    observacoes_raw = raw_payload.get("observacoes")
    observacoes = "" if observacoes_raw is None else str(observacoes_raw).strip()
    observacoes_norm = _normalize_text_for_rule_checks(observacoes)
    has_header_hint = any(hint in observacoes_norm for hint in HEADER_OBS_HINTS)

    nome_valido = bool(nome and len(nome.split()) >= 2)
    crm_valido = bool(crm_numero and crm_uf and crm_uf in UF_BRASIL)
    valido = nome_valido and crm_valido
    if has_header_hint:
        valido = False
        confidence = min(confidence, 0.35)
        tag = "suspeita_origem_cabecalho_pcms0"
        observacoes = f"{observacoes} | {tag}" if observacoes else tag
    crm = f"{crm_numero}/{crm_uf}" if crm_valido else None

    return MedicoExtraction(
        nome=nome,
        crm_numero=crm_numero,
        crm_uf=crm_uf,
        crm=crm,
        confidence=confidence,
        valido=valido,
        observacoes=observacoes,
    )


def detect_stamp_candidates_with_gemini(
    image: Image.Image,
    api_key: str,
    model: str,
    timeout_seconds: int,
    max_candidates: int,
    retry_attempts: int,
    retry_backoff_seconds: float,
    retry_jitter_seconds: float,
    usage_sink: Optional[list[dict[str, Any]]] = None,
) -> list[BBoxCandidate]:
    width, height = image.size
    prompt = f"""
Você detecta BLOCO DE ASSINATURA E CARIMBO MÉDICO em ASO.
Imagem em pixels: largura={width}, altura={height}.

Retorne SOMENTE JSON no formato:
{{
  "candidatos": [
    {{
      "bbox": {{"x": 0, "y": 0, "w": 0, "h": 0}},
      "score": 0.0,
      "motivo": "curto"
    }}
  ]
}}

Regras:
- x, y, w, h inteiros em pixels da imagem original.
- Não retorne markdown.
- Foque no BLOCO COMPLETO (não apenas uma palavra), incluindo:
  1) texto "Assinatura do Médico" (quando existir),
  2) assinatura manuscrita,
  3) palavra "Carimbo" e/ou linha com "CRM".
- Evite caixas apertadas em texto isolado. A caixa deve cobrir contexto suficiente
  para leitura de nome/CRM.
- Em geral está na metade inferior da página, mas pode variar.
- Retorne no máximo {max_candidates} candidatos.
- score de 0.0 a 1.0.
""".strip()
    raw_payload = _call_gemini(
        api_key=api_key,
        model=model,
        prompt=prompt,
        image=image,
        timeout_seconds=timeout_seconds,
        retry_attempts=retry_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        retry_jitter_seconds=retry_jitter_seconds,
        usage_sink=usage_sink,
        usage_stage="detection",
    )
    return _sanitize_bbox_candidates(
        raw_payload=raw_payload,
        width=width,
        height=height,
        max_candidates=max_candidates,
    )


def extract_medico_with_gemini(
    crop_image: Image.Image,
    api_key: str,
    model: str,
    timeout_seconds: int,
    retry_attempts: int,
    retry_backoff_seconds: float,
    retry_jitter_seconds: float,
    usage_sink: Optional[list[dict[str, Any]]] = None,
) -> MedicoExtraction:
    prompt = """
Extraia dados do médico a partir da imagem do carimbo/assinatura.
Retorne SOMENTE JSON:
{
  "nome": "string ou null",
  "crm_numero": "somente dígitos ou null",
  "crm_uf": "UF com 2 letras ou null",
  "confianca": 0.0,
  "observacoes": "string curta"
}

Regras:
- Não invente valores.
- Se não tiver certeza, use null.
- confianca entre 0.0 e 1.0.
- Não retornar markdown.
- Se CRM vier com separadores (ex: "60.750", "26-807-2"), normalize para somente dígitos
  em crm_numero (ex: "60750", "268072").
- PRIORIZE EXCLUSIVAMENTE o médico assinante do RODAPÉ (bloco "MÉDICO ENCARREGADO DO EXAME",
  "Assinatura do Médico", "Carimbo", "CRM" próximo da assinatura).
- IGNORE médico do cabeçalho/PCMSO. Se só houver médico do cabeçalho visível no recorte,
  retorne nome=null e crm_numero=null e crm_uf=null.
- Se perceber que os dados vieram do cabeçalho/PCMSO, escreva em observacoes:
  "suspeita_origem_cabecalho_pcms0".
""".strip()
    raw_payload = _call_gemini(
        api_key=api_key,
        model=model,
        prompt=prompt,
        image=crop_image,
        timeout_seconds=timeout_seconds,
        retry_attempts=retry_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        retry_jitter_seconds=retry_jitter_seconds,
        usage_sink=usage_sink,
        usage_stage="extraction",
    )
    return _normalize_medico_payload(raw_payload)


def combined_candidate_score(
    detection_score: float,
    extraction: MedicoExtraction,
    geometry_adjustment: float = 0.0,
) -> float:
    valid_bonus = 0.2 if extraction.valido else -0.1
    value = (
        (0.55 * _clamp(detection_score))
        + (0.45 * extraction.confidence)
        + valid_bonus
        + geometry_adjustment
    )
    return _clamp(value)


def bbox_area_ratio(bbox: BBoxTuple, width: int, height: int) -> float:
    _, _, w, h = bbox
    image_area = float(max(1, width * height))
    return float((w * h) / image_area)


def bbox_geometry_adjustment(bbox: BBoxTuple, width: int, height: int) -> float:
    x, y, w, h = bbox
    if width <= 0 or height <= 0:
        return 0.0

    area_ratio = bbox_area_ratio(bbox, width=width, height=height)
    w_ratio = float(w / width)
    h_ratio = float(h / height)
    y_ratio = float(y / height)

    adjustment = 0.0
    if area_ratio < 0.015:
        adjustment -= 0.30
    elif area_ratio < 0.03:
        adjustment -= 0.18
    elif area_ratio < 0.05:
        adjustment -= 0.08
    elif area_ratio > 0.65:
        adjustment -= 0.08

    if w_ratio < 0.15:
        adjustment -= 0.10
    if h_ratio < 0.08:
        adjustment -= 0.10
    if y_ratio >= 0.40:
        adjustment += 0.04

    return float(max(-0.45, min(0.10, adjustment)))


def expand_stamp_bbox(
    bbox: BBoxTuple,
    width: int,
    height: int,
    *,
    left_ratio: float = 0.22,
    right_ratio: float = 0.28,
    top_ratio: float = 0.20,
    bottom_ratio: float = 1.10,
    min_w_ratio: float = 0.24,
    min_h_ratio: float = 0.16,
) -> BBoxTuple:
    x, y, w, h = bbox

    x0 = int(round(x - (w * left_ratio)))
    x1 = int(round(x + w + (w * right_ratio)))
    y0 = int(round(y - (h * top_ratio)))
    y1 = int(round(y + h + (h * bottom_ratio)))

    min_w = max(1, int(round(width * min_w_ratio)))
    min_h = max(1, int(round(height * min_h_ratio)))

    cur_w = x1 - x0
    if cur_w < min_w:
        deficit = min_w - cur_w
        left_extra = deficit // 2
        right_extra = deficit - left_extra
        x0 -= left_extra
        x1 += right_extra

    cur_h = y1 - y0
    if cur_h < min_h:
        deficit = min_h - cur_h
        # Para carimbo, o conteúdo útil costuma estar mais abaixo que o título.
        up_extra = int(round(deficit * 0.30))
        down_extra = deficit - up_extra
        y0 -= up_extra
        y1 += down_extra

    x0 = _clamp_int(x0, 0, max(0, width - 1))
    y0 = _clamp_int(y0, 0, max(0, height - 1))
    x1 = _clamp_int(x1, x0 + 1, max(1, width))
    y1 = _clamp_int(y1, y0 + 1, max(1, height))
    return x0, y0, x1 - x0, y1 - y0


def build_stamp_crop_variants(
    bbox: BBoxTuple,
    width: int,
    height: int,
) -> list[BBoxTuple]:
    variants = [
        expand_stamp_bbox(
            bbox=bbox,
            width=width,
            height=height,
            left_ratio=0.22,
            right_ratio=0.28,
            top_ratio=0.20,
            bottom_ratio=1.10,
            min_w_ratio=0.24,
            min_h_ratio=0.16,
        ),
        expand_stamp_bbox(
            bbox=bbox,
            width=width,
            height=height,
            left_ratio=0.15,
            right_ratio=0.22,
            top_ratio=0.12,
            bottom_ratio=1.55,
            min_w_ratio=0.28,
            min_h_ratio=0.19,
        ),
        expand_stamp_bbox(
            bbox=bbox,
            width=width,
            height=height,
            left_ratio=0.35,
            right_ratio=0.35,
            top_ratio=0.30,
            bottom_ratio=1.00,
            min_w_ratio=0.34,
            min_h_ratio=0.20,
        ),
    ]
    unique: list[BBoxTuple] = []
    seen: set[BBoxTuple] = set()
    for variant in variants:
        if variant in seen:
            continue
        seen.add(variant)
        unique.append(variant)
    return unique
