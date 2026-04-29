from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import Any, Optional
from urllib import error, parse, request


class SocServiceError(RuntimeError):
    pass


@dataclass(slots=True)
class SocRecord:
    cd_pessoa: str
    nm_pessoa: str
    cd_cpf: str
    cd_conselho: str
    nm_conselho: str
    sg_ufconselho: str
    cd_usuario: str


@dataclass(slots=True)
class SocValidationResult:
    enabled: bool
    consulted: bool
    crm_consultado: Optional[str]
    uf_detectada: Optional[str]
    total_registros: int
    nome_detectado: Optional[str]
    melhor_nome_soc: Optional[str]
    melhor_crm_soc: Optional[str]
    melhor_uf_soc: Optional[str]
    similaridade_nome: float
    limiar_similaridade: float
    nome_parecido: bool
    revisao_humana_recomendada: bool
    motivo: str
    erro: str
    amostra: list[SocRecord]
    correcao_sugerida: bool = False
    crm_numero_sugerido: Optional[str] = None
    crm_uf_sugerida: Optional[str] = None
    crm_sugerido: Optional[str] = None
    nome_sugerido_soc: Optional[str] = None
    similaridade_nome_sugerida: float = 0.0
    variacoes_crm_consultadas: int = 0


_NON_ALNUM_RE = re.compile(r"[^A-Z0-9 ]+")
_SPACE_RE = re.compile(r"\s+")
_NAME_STOPWORDS = {
    "DR",
    "DRA",
    "DOUTOR",
    "DOUTORA",
    "MEDICO",
    "MEDICA",
}
_CRM_ONLY_DIGITS_RE = re.compile(r"^\d{4,8}$")


def _compose_crm(numero: Optional[str], uf: Optional[str]) -> Optional[str]:
    if not numero:
        return None
    if uf:
        return f"{numero}/{uf}"
    return numero


def _compose_error_details(*messages: str) -> str:
    parts = [part.strip() for part in messages if part and part.strip()]
    return " | ".join(parts[:3])


def _build_crm_suffix_variants(crm_numero: str) -> list[str]:
    if not _CRM_ONLY_DIGITS_RE.match(crm_numero):
        return []
    if len(crm_numero) >= 8:
        return []
    # OCR em carimbo costuma perder o último dígito do CRM.
    return [f"{crm_numero}{digit}" for digit in "0123456789"]


def _build_crm_prefix_variants(crm_numero: str) -> list[str]:
    if not _CRM_ONLY_DIGITS_RE.match(crm_numero):
        return []
    if len(crm_numero) >= 8:
        return []
    # Em alguns casos o OCR perde o primeiro dígito do CRM.
    return [f"{digit}{crm_numero}" for digit in "0123456789"]


def _should_try_suffix_recovery(
    *,
    validation: SocValidationResult,
    crm_numero: str,
    nome_detectado: Optional[str],
) -> bool:
    if not validation.consulted:
        return False
    if not validation.revisao_humana_recomendada:
        return False
    if not _CRM_ONLY_DIGITS_RE.match(crm_numero):
        return False
    if len(crm_numero) < 4 or len(crm_numero) >= 8:
        return False
    if not normalize_person_name(nome_detectado):
        return False
    return True


def _attempt_crm_variants_recovery(
    *,
    base_url: str,
    empresa: str,
    codigo: str,
    chave: str,
    tipo_saida: str,
    timeout_seconds: int,
    crm_numero: str,
    crm_uf: Optional[str],
    nome_detectado: Optional[str],
    threshold: float,
    variants: list[str],
) -> tuple[Optional[SocValidationResult], int, str]:
    if not variants:
        return None, 0, ""

    attempts = 0
    errors: list[str] = []
    best_variant_result: Optional[SocValidationResult] = None

    for variant in variants:
        attempts += 1
        try:
            variant_records = query_soc_by_crm(
                base_url=base_url,
                empresa=empresa,
                codigo=codigo,
                chave=chave,
                tipo_saida=tipo_saida,
                crm_numero=variant,
                timeout_seconds=timeout_seconds,
            )
        except SocServiceError as exc:
            errors.append(f"{variant}: {str(exc)[:180]}")
            continue

        variant_result = evaluate_soc_records(
            records=variant_records,
            crm_numero=variant,
            crm_uf=crm_uf,
            nome_detectado=nome_detectado,
            threshold=threshold,
        )
        if best_variant_result is None:
            best_variant_result = variant_result
        else:
            current_rank = (
                best_variant_result.similaridade_nome,
                1
                if (
                    crm_uf
                    and best_variant_result.melhor_uf_soc
                    and best_variant_result.melhor_uf_soc == crm_uf
                )
                else 0,
            )
            new_rank = (
                variant_result.similaridade_nome,
                1
                if (crm_uf and variant_result.melhor_uf_soc and variant_result.melhor_uf_soc == crm_uf)
                else 0,
            )
            if new_rank > current_rank:
                best_variant_result = variant_result

        # Atalho para reduzir chamadas quando já há match forte.
        if (
            variant_result.nome_parecido
            and variant_result.similaridade_nome >= max(0.92, min(0.97, threshold + 0.12))
            and ((not crm_uf) or (variant_result.melhor_uf_soc == crm_uf))
        ):
            best_variant_result = variant_result
            break

    return best_variant_result, attempts, "; ".join(errors[:2])


def _is_variant_result_meaningful(
    *,
    initial_result: SocValidationResult,
    variant_result: SocValidationResult,
    crm_uf: Optional[str],
    nome_detectado: Optional[str],
    threshold: float,
) -> bool:
    common_tokens = _count_common_name_tokens(
        nome_detectado,
        variant_result.melhor_nome_soc,
    )
    variant_name_acceptable = (
        variant_result.nome_parecido
        or (
            common_tokens >= 2
            and variant_result.similaridade_nome >= max(0.60, threshold - 0.18)
        )
    )
    score_gain = variant_result.similaridade_nome - initial_result.similaridade_nome
    fixes_uf = bool(
        crm_uf
        and initial_result.melhor_uf_soc != crm_uf
        and variant_result.melhor_uf_soc == crm_uf
    )
    score_gain_ok = (
        score_gain >= 0.20 if not initial_result.nome_parecido else score_gain >= 0.07
    )
    return bool(variant_name_acceptable and (score_gain_ok or fixes_uf))


def normalize_person_name(value: Optional[str]) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = "".join(char for char in normalized if not unicodedata.combining(char))
    upper = ascii_only.upper()
    cleaned = _NON_ALNUM_RE.sub(" ", upper)
    return _SPACE_RE.sub(" ", cleaned).strip()


def _tokenize_name(value: Optional[str]) -> list[str]:
    normalized = normalize_person_name(value)
    if not normalized:
        return []
    return [token for token in normalized.split() if token not in _NAME_STOPWORDS]


def _count_common_name_tokens(left: Optional[str], right: Optional[str]) -> int:
    left_tokens = set(_tokenize_name(left))
    right_tokens = set(_tokenize_name(right))
    return len(left_tokens.intersection(right_tokens))


def compute_name_similarity(detected_name: Optional[str], soc_name: Optional[str]) -> float:
    left_tokens = _tokenize_name(detected_name)
    right_tokens = _tokenize_name(soc_name)
    if not left_tokens or not right_tokens:
        return 0.0

    left = " ".join(left_tokens)
    right = " ".join(right_tokens)
    seq_ratio = SequenceMatcher(None, left, right).ratio()
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    common = left_set.intersection(right_set)
    union = left_set.union(right_set)

    jaccard = (len(common) / len(union)) if union else 0.0
    containment_den = max(len(left_set), len(right_set), 1)
    containment = len(common) / containment_den
    score = (0.60 * seq_ratio) + (0.25 * jaccard) + (0.15 * containment)
    return round(float(max(0.0, min(1.0, score))), 4)


def build_soc_request_url(
    *,
    base_url: str,
    empresa: str,
    codigo: str,
    chave: str,
    tipo_saida: str,
    conselho_classe: str,
) -> str:
    payload = {
        "empresa": empresa,
        "codigo": codigo,
        "chave": chave,
        "tipoSaida": tipo_saida,
        "conselhoClasse": conselho_classe,
    }
    encoded_param = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    parsed = parse.urlsplit(base_url)
    current_query = parse.parse_qs(parsed.query, keep_blank_values=True)
    current_query["parametro"] = [encoded_param]
    flattened_query = parse.urlencode(
        {key: value[-1] if value else "" for key, value in current_query.items()}
    )
    return parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            flattened_query,
            parsed.fragment,
        )
    )


def _to_soc_record(entry: dict[str, Any]) -> SocRecord:
    return SocRecord(
        cd_pessoa=str(entry.get("CD_PESSOA", "")).strip(),
        nm_pessoa=str(entry.get("NM_PESSOA", "")).strip(),
        cd_cpf=str(entry.get("CD_CPF", "")).strip(),
        cd_conselho=str(entry.get("CD_CONSELHO", "")).strip(),
        nm_conselho=str(entry.get("NM_CONSELHO", "")).strip(),
        sg_ufconselho=str(entry.get("SG_UFCONSELHO", "")).strip().upper(),
        cd_usuario=str(entry.get("CD_USUARIO", "")).strip(),
    )


def query_soc_by_crm(
    *,
    base_url: str,
    empresa: str,
    codigo: str,
    chave: str,
    tipo_saida: str,
    crm_numero: str,
    timeout_seconds: int,
) -> list[SocRecord]:
    if not crm_numero:
        return []

    url = build_soc_request_url(
        base_url=base_url,
        empresa=empresa,
        codigo=codigo,
        chave=chave,
        tipo_saida=tipo_saida,
        conselho_classe=crm_numero,
    )
    req = request.Request(
        url=url,
        method="GET",
        headers={"Accept": "application/json"},
    )

    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            raw_body = response.read()
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise SocServiceError(f"SOC HTTP {exc.code}: {details[:300]}") from exc
    except error.URLError as exc:
        raise SocServiceError(f"Falha de conexão no SOC: {exc}") from exc

    try:
        body = raw_body.decode("utf-8")
    except UnicodeDecodeError:
        try:
            body = raw_body.decode("latin-1")
        except UnicodeDecodeError:
            body = raw_body.decode("utf-8", errors="ignore")

    try:
        raw_payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SocServiceError("Resposta inválida do SOC (não é JSON).") from exc

    if not isinstance(raw_payload, list):
        raise SocServiceError("Resposta inválida do SOC (esperado array de médicos).")

    records: list[SocRecord] = []
    for raw_entry in raw_payload:
        if isinstance(raw_entry, dict):
            records.append(_to_soc_record(raw_entry))
    return records


def evaluate_soc_records(
    *,
    records: list[SocRecord],
    crm_numero: str,
    crm_uf: Optional[str],
    nome_detectado: Optional[str],
    threshold: float,
) -> SocValidationResult:
    if not records:
        return SocValidationResult(
            enabled=True,
            consulted=True,
            crm_consultado=crm_numero,
            uf_detectada=crm_uf,
            total_registros=0,
            nome_detectado=nome_detectado,
            melhor_nome_soc=None,
            melhor_crm_soc=None,
            melhor_uf_soc=None,
            similaridade_nome=0.0,
            limiar_similaridade=threshold,
            nome_parecido=False,
            revisao_humana_recomendada=True,
            motivo="crm_nao_encontrado_no_soc",
            erro="",
            amostra=[],
        )

    scored_records: list[tuple[float, SocRecord]] = []
    for record in records:
        score = compute_name_similarity(nome_detectado, record.nm_pessoa)
        scored_records.append((score, record))
    scored_records.sort(key=lambda item: item[0], reverse=True)
    best_score, best_record = scored_records[0]

    name_detected_ok = bool(normalize_person_name(nome_detectado))
    name_is_similar = bool(name_detected_ok and best_score >= threshold)
    crm_matches = best_record.nm_conselho == crm_numero
    uf_matches = (not crm_uf) or (best_record.sg_ufconselho == crm_uf)

    if not name_detected_ok:
        motivo = "nome_nao_detectado_para_comparacao_soc"
        needs_review = True
    elif name_is_similar and crm_matches and uf_matches:
        motivo = "soc_ok_nome_e_crm_compativeis"
        needs_review = False
    elif name_is_similar and crm_matches and not uf_matches:
        motivo = "soc_nome_compativel_mas_uf_divergente"
        needs_review = True
    elif crm_matches and not name_is_similar:
        motivo = "soc_crm_encontrado_nome_divergente"
        needs_review = True
    else:
        motivo = "soc_divergencia_em_nome_ou_crm"
        needs_review = True

    return SocValidationResult(
        enabled=True,
        consulted=True,
        crm_consultado=crm_numero,
        uf_detectada=crm_uf,
        total_registros=len(records),
        nome_detectado=nome_detectado,
        melhor_nome_soc=best_record.nm_pessoa or None,
        melhor_crm_soc=best_record.nm_conselho or None,
        melhor_uf_soc=best_record.sg_ufconselho or None,
        similaridade_nome=best_score,
        limiar_similaridade=threshold,
        nome_parecido=name_is_similar,
        revisao_humana_recomendada=needs_review,
        motivo=motivo,
        erro="",
        amostra=[record for _, record in scored_records[:3]],
    )


def validate_with_soc(
    *,
    enabled: bool,
    crm_numero: Optional[str],
    crm_uf: Optional[str],
    nome_detectado: Optional[str],
    threshold: float,
    base_url: str,
    empresa: str,
    codigo: str,
    chave: str,
    tipo_saida: str,
    timeout_seconds: int,
) -> SocValidationResult:
    if not enabled:
        return SocValidationResult(
            enabled=False,
            consulted=False,
            crm_consultado=crm_numero,
            uf_detectada=crm_uf,
            total_registros=0,
            nome_detectado=nome_detectado,
            melhor_nome_soc=None,
            melhor_crm_soc=None,
            melhor_uf_soc=None,
            similaridade_nome=0.0,
            limiar_similaridade=threshold,
            nome_parecido=False,
            revisao_humana_recomendada=False,
            motivo="soc_desabilitado",
            erro="",
            amostra=[],
        )

    if not crm_numero:
        return SocValidationResult(
            enabled=True,
            consulted=False,
            crm_consultado=crm_numero,
            uf_detectada=crm_uf,
            total_registros=0,
            nome_detectado=nome_detectado,
            melhor_nome_soc=None,
            melhor_crm_soc=None,
            melhor_uf_soc=None,
            similaridade_nome=0.0,
            limiar_similaridade=threshold,
            nome_parecido=False,
            revisao_humana_recomendada=True,
            motivo="crm_ausente_para_consulta_soc",
            erro="",
            amostra=[],
        )

    try:
        records = query_soc_by_crm(
            base_url=base_url,
            empresa=empresa,
            codigo=codigo,
            chave=chave,
            tipo_saida=tipo_saida,
            crm_numero=crm_numero,
            timeout_seconds=timeout_seconds,
        )
    except SocServiceError as exc:
        return SocValidationResult(
            enabled=True,
            consulted=False,
            crm_consultado=crm_numero,
            uf_detectada=crm_uf,
            total_registros=0,
            nome_detectado=nome_detectado,
            melhor_nome_soc=None,
            melhor_crm_soc=None,
            melhor_uf_soc=None,
            similaridade_nome=0.0,
            limiar_similaridade=threshold,
            nome_parecido=False,
            revisao_humana_recomendada=True,
            motivo="falha_consulta_soc",
            erro=str(exc),
            amostra=[],
        )

    initial_result = evaluate_soc_records(
        records=records,
        crm_numero=crm_numero,
        crm_uf=crm_uf,
        nome_detectado=nome_detectado,
        threshold=threshold,
    )
    if not _should_try_suffix_recovery(
        validation=initial_result,
        crm_numero=crm_numero,
        nome_detectado=nome_detectado,
    ):
        return initial_result

    suffix_variants = _build_crm_suffix_variants(crm_numero)
    variant_result, variant_attempts, variant_errors = _attempt_crm_variants_recovery(
        base_url=base_url,
        empresa=empresa,
        codigo=codigo,
        chave=chave,
        tipo_saida=tipo_saida,
        timeout_seconds=timeout_seconds,
        crm_numero=crm_numero,
        crm_uf=crm_uf,
        nome_detectado=nome_detectado,
        threshold=threshold,
        variants=suffix_variants,
    )
    total_variant_attempts = variant_attempts
    variant_error_details = [variant_errors]
    selected_variant_result: Optional[SocValidationResult] = None

    if (
        variant_result is not None
        and _is_variant_result_meaningful(
            initial_result=initial_result,
            variant_result=variant_result,
            crm_uf=crm_uf,
            nome_detectado=nome_detectado,
            threshold=threshold,
        )
    ):
        selected_variant_result = variant_result
    else:
        # Se não houver sugestão pela perda de dígito final, tenta perda de dígito inicial.
        prefix_variants = _build_crm_prefix_variants(crm_numero)
        prefix_result, prefix_attempts, prefix_errors = _attempt_crm_variants_recovery(
            base_url=base_url,
            empresa=empresa,
            codigo=codigo,
            chave=chave,
            tipo_saida=tipo_saida,
            timeout_seconds=timeout_seconds,
            crm_numero=crm_numero,
            crm_uf=crm_uf,
            nome_detectado=nome_detectado,
            threshold=threshold,
            variants=prefix_variants,
        )
        total_variant_attempts += prefix_attempts
        variant_error_details.append(prefix_errors)
        if (
            prefix_result is not None
            and _is_variant_result_meaningful(
                initial_result=initial_result,
                variant_result=prefix_result,
                crm_uf=crm_uf,
                nome_detectado=nome_detectado,
                threshold=threshold,
            )
        ):
            selected_variant_result = prefix_result

    if total_variant_attempts <= 0:
        return initial_result

    result_with_attempts = replace(
        initial_result,
        variacoes_crm_consultadas=total_variant_attempts,
        erro=_compose_error_details(initial_result.erro, *variant_error_details),
    )

    if selected_variant_result is None:
        return result_with_attempts

    crm_numero_sugerido = selected_variant_result.crm_consultado
    crm_uf_sugerida = selected_variant_result.melhor_uf_soc
    crm_sugerido = _compose_crm(crm_numero_sugerido, crm_uf_sugerida)
    return replace(
        result_with_attempts,
        motivo="soc_sugere_crm_truncado_por_nome_compativel",
        revisao_humana_recomendada=True,
        correcao_sugerida=True,
        crm_numero_sugerido=crm_numero_sugerido,
        crm_uf_sugerida=crm_uf_sugerida,
        crm_sugerido=crm_sugerido,
        nome_sugerido_soc=selected_variant_result.melhor_nome_soc,
        similaridade_nome_sugerida=selected_variant_result.similaridade_nome,
    )
