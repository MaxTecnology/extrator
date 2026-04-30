import base64
import io
import logging
import mimetypes
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from app.config import Settings, get_settings
from app.schemas.carimbo import (
    AsoGeralExtractRequest,
    AsoGeralExtractResponse,
    AsoEmpresaInfo,
    AsoExameInfo,
    AsoFuncionarioInfo,
    AsoParecerInfo,
    AsoRiscosInfo,
    BBox,
    DebugResponse,
    ErrorResponse,
    ExtractRequest,
    ExtractResponse,
    GeminiHealthResponse,
    GeminiModelHealthInfo,
    GeminiTelemetryInfo,
    GeminiExtractRequest,
    GeminiExtractResponse,
    MedicoInfo,
    SocRecordInfo,
    SocValidationInfo,
    SUPPORTED_MIME_TYPES,
)
from app.services.detector import BBoxTuple, detect_stamp_region
from app.services.gemini_pipeline import (
    BBoxCandidate,
    GeminiServiceError,
    MedicoExtraction,
    bbox_geometry_adjustment,
    build_stamp_crop_variants,
    combined_candidate_score,
    detect_stamp_candidates_with_gemini,
    extract_aso_general_with_gemini,
    extract_medico_with_gemini,
    probe_gemini_model,
)
from app.services.pdf_renderer import render_page
from app.services.preprocessor import preprocess_stamp
from app.services.soc_validator import SocValidationResult, validate_with_soc


logger = logging.getLogger(__name__)
router = APIRouter(tags=["carimbo"])


def _to_bbox_model(bbox: Optional[BBoxTuple]) -> Optional[BBox]:
    if bbox is None:
        return None
    x, y, w, h = bbox
    return BBox(x=x, y=y, w=w, h=h)


@router.get("/health/gemini", response_model=GeminiHealthResponse)
def gemini_health(settings: Settings = Depends(get_settings)) -> GeminiHealthResponse:
    timeout = max(3, min(12, int(settings.gemini_timeout_seconds)))
    if not settings.gemini_api_key:
        return GeminiHealthResponse(
            status="not_configured",
            api_key_configurada=False,
            timeout_segundos=timeout,
            detection=GeminiModelHealthInfo(
                modelo=settings.gemini_detection_model,
                disponivel=False,
                erro="GEMINI_API_KEY não configurada",
            ),
            extraction=GeminiModelHealthInfo(
                modelo=settings.gemini_extraction_model,
                disponivel=False,
                erro="GEMINI_API_KEY não configurada",
            ),
        )

    detection_ok, detection_error = probe_gemini_model(
        api_key=settings.gemini_api_key,
        model=settings.gemini_detection_model,
        timeout_seconds=timeout,
        retry_attempts=0,
    )
    extraction_ok, extraction_error = probe_gemini_model(
        api_key=settings.gemini_api_key,
        model=settings.gemini_extraction_model,
        timeout_seconds=timeout,
        retry_attempts=0,
    )
    if detection_ok and extraction_ok:
        status = "ok"
    elif detection_ok or extraction_ok:
        status = "degraded"
    else:
        status = "unavailable"

    return GeminiHealthResponse(
        status=status,
        api_key_configurada=True,
        timeout_segundos=timeout,
        detection=GeminiModelHealthInfo(
            modelo=settings.gemini_detection_model,
            disponivel=detection_ok,
            erro=detection_error,
        ),
        extraction=GeminiModelHealthInfo(
            modelo=settings.gemini_extraction_model,
            disponivel=extraction_ok,
            erro=extraction_error,
        ),
    )


def _decode_file(file_base64: str) -> bytes:
    return base64.b64decode(file_base64, validate=True)


def _enforce_file_size(file_bytes: bytes, limit_mb: int) -> None:
    max_bytes = limit_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Arquivo excede o limite de {limit_mb}MB",
        )


def _load_input_image(
    file_bytes: bytes,
    mime_type: str,
    pagina: int,
    settings: Settings,
) -> Image.Image:
    if mime_type == "application/pdf":
        try:
            return render_page(file_bytes, pagina=pagina, dpi=settings.dpi_render)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    try:
        with Image.open(io.BytesIO(file_bytes)) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Imagem inválida ou corrompida",
        ) from exc


def _encode_png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _save_image_artifact(
    image: Image.Image,
    settings: Settings,
    prefix: str,
) -> str:
    base_dir = Path(settings.image_artifacts_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{uuid4().hex}.png"
    target_path = base_dir / filename
    image.save(target_path, format="PNG")
    url_prefix = "/" + settings.image_artifacts_url_prefix.strip("/")
    return f"{url_prefix}/{filename}"


def _build_image_outputs(
    image: Image.Image,
    *,
    settings: Settings,
    include_base64: bool,
    include_url: bool,
    artifact_prefix: str,
) -> tuple[Optional[str], Optional[str]]:
    img_base64: Optional[str] = _encode_png_base64(image) if include_base64 else None
    img_url: Optional[str] = (
        _save_image_artifact(image, settings=settings, prefix=artifact_prefix)
        if include_url
        else None
    )
    return img_base64, img_url


def _resolve_upload_mime_type(
    *,
    upload: UploadFile,
    mime_type_override: Optional[str],
    file_bytes: bytes,
) -> str:
    def _detect_mime_by_signature(raw: bytes) -> Optional[str]:
        if raw.startswith(b"%PDF-"):
            return "application/pdf"
        if raw.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if raw.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if raw.startswith((b"II*\x00", b"MM\x00*")):
            return "image/tiff"
        return None

    candidate = (mime_type_override or upload.content_type or "").strip().lower()
    if candidate in SUPPORTED_MIME_TYPES:
        return candidate

    filename = (upload.filename or "").strip().lower()
    guessed, _ = mimetypes.guess_type(filename)
    if guessed and guessed in SUPPORTED_MIME_TYPES:
        return guessed

    if candidate in {"application/octet-stream", "binary/octet-stream"} and filename.endswith(".pdf"):
        return "application/pdf"

    by_signature = _detect_mime_by_signature(file_bytes[:32])
    if by_signature and by_signature in SUPPORTED_MIME_TYPES:
        return by_signature

    allowed = ", ".join(sorted(SUPPORTED_MIME_TYPES))
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=(
            "mime_type inválido para upload. "
            f"Use um dos suportados: {allowed}, ou envie um arquivo PDF/PNG/JPEG/TIFF válido."
        ),
    )


def _draw_debug_overlay(
    image: Image.Image,
    bbox: Optional[BBoxTuple],
    confidence: float,
    detected: bool,
) -> Image.Image:
    debug_image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(debug_image)
    font = ImageFont.load_default()
    label_status = "detected" if detected else "fallback"
    label_text = f"{label_status} {confidence:.2f}"

    if bbox is not None:
        x, y, w, h = bbox
        draw.rectangle((x, y, x + w, y + h), outline=(255, 0, 0), width=3)
        label_x = x
        label_y = max(5, y - 16)
    else:
        label_x = 12
        label_y = 12

    text_bbox = draw.textbbox((label_x, label_y), label_text, font=font)
    draw.rectangle(
        (text_bbox[0] - 3, text_bbox[1] - 2, text_bbox[2] + 3, text_bbox[3] + 2),
        fill=(255, 255, 255),
    )
    draw.text((label_x, label_y), label_text, fill=(255, 0, 0), font=font)
    return debug_image


def _medico_info_model(extraction: MedicoExtraction, origem: str) -> MedicoInfo:
    return MedicoInfo(
        nome=extraction.nome,
        crm=extraction.crm,
        crm_numero=extraction.crm_numero,
        crm_uf=extraction.crm_uf,
        confianca=float(extraction.confidence),
        valido=bool(extraction.valido),
        origem=origem,
        observacoes=extraction.observacoes,
    )


def _soc_validation_model(result: SocValidationResult) -> SocValidationInfo:
    return SocValidationInfo(
        habilitada=result.enabled,
        consultada=result.consulted,
        crm_consultado=result.crm_consultado,
        uf_detectada=result.uf_detectada,
        total_registros=result.total_registros,
        nome_detectado=result.nome_detectado,
        melhor_nome_soc=result.melhor_nome_soc,
        melhor_crm_soc=result.melhor_crm_soc,
        melhor_uf_soc=result.melhor_uf_soc,
        similaridade_nome=float(result.similaridade_nome),
        limiar_similaridade=float(result.limiar_similaridade),
        nome_parecido=result.nome_parecido,
        revisao_humana_recomendada=result.revisao_humana_recomendada,
        motivo=result.motivo,
        erro=result.erro,
        correcao_sugerida=result.correcao_sugerida,
        crm_numero_sugerido=result.crm_numero_sugerido,
        crm_uf_sugerida=result.crm_uf_sugerida,
        crm_sugerido=result.crm_sugerido,
        nome_sugerido_soc=result.nome_sugerido_soc,
        similaridade_nome_sugerida=float(result.similaridade_nome_sugerida),
        variacoes_crm_consultadas=int(result.variacoes_crm_consultadas),
        amostra=[
            SocRecordInfo(
                cd_pessoa=record.cd_pessoa,
                nm_pessoa=record.nm_pessoa,
                nm_conselho=record.nm_conselho,
                sg_ufconselho=record.sg_ufconselho,
                cd_usuario=record.cd_usuario,
            )
            for record in result.amostra
        ],
    )


def _build_gemini_telemetry(events: list[dict[str, object]]) -> GeminiTelemetryInfo:
    if not events:
        return GeminiTelemetryInfo()
    prompt_tokens = sum(int(item.get("prompt_tokens", 0) or 0) for item in events)
    output_tokens = sum(int(item.get("output_tokens", 0) or 0) for item in events)
    total_tokens = sum(int(item.get("total_tokens", 0) or 0) for item in events)
    latency_total_ms = sum(int(item.get("latency_ms", 0) or 0) for item in events)
    attempts_total = sum(int(item.get("attempts_used", 0) or 0) for item in events)
    detection_calls = sum(
        1 for item in events if str(item.get("stage", "")).lower() == "detection"
    )
    extraction_calls = sum(
        1
        for item in events
        if str(item.get("stage", "")).lower() in {"extraction", "aso_general"}
    )
    return GeminiTelemetryInfo(
        chamadas_total=len(events),
        chamadas_deteccao=detection_calls,
        chamadas_extracao=extraction_calls,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        latencia_total_ms=latency_total_ms,
        tentativas_total=attempts_total,
    )


def _build_aso_review_flags(aso_payload: dict[str, dict[str, str]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    empresa = aso_payload.get("empresa", {})
    funcionario = aso_payload.get("funcionario", {})
    exame = aso_payload.get("exame", {})
    parecer = aso_payload.get("parecer", {})

    critical_fields: list[tuple[str, str]] = [
        ("empresa.razao_social", empresa.get("razao_social", "")),
        ("empresa.cnpj", empresa.get("cnpj", "")),
        ("funcionario.nome", funcionario.get("nome", "")),
        ("funcionario.cpf", funcionario.get("cpf", "")),
        ("exame.tipo", exame.get("tipo", "")),
        ("exame.data_aso", exame.get("data_aso", "")),
        ("parecer.geral", parecer.get("geral", "")),
    ]
    for field_name, field_value in critical_fields:
        value = str(field_value or "").strip()
        if value in {"", "**", "Ausente"}:
            reasons.append(f"campo_critico_incompleto:{field_name}")

    cpf = str(funcionario.get("cpf", "") or "").strip()
    if cpf not in {"", "**", "Ausente"}:
        digits = "".join(char for char in cpf if char.isdigit())
        if len(digits) != 11:
            reasons.append("cpf_formato_duvidoso")

    return bool(reasons), reasons


def _clamp_ratio(value: float, minimum: float, maximum: float) -> float:
    return float(max(minimum, min(maximum, value)))


def _compute_bottom_focus_bbox(
    *,
    width: int,
    height: int,
    y_start_ratio: float,
    x_end_ratio: float,
) -> tuple[int, int, int, int]:
    x0 = 0
    x1 = max(1, min(width, int(round(width * x_end_ratio))))
    y0 = max(0, min(height - 1, int(round(height * y_start_ratio))))
    y1 = height
    return x0, y0, x1, y1


def _bottom_priority_adjustment(
    *,
    bbox: BBoxTuple,
    image_height: int,
    bottom_y_start_ratio: float,
) -> float:
    if image_height <= 0:
        return 0.0
    y_ratio = float(bbox[1] / image_height)
    if y_ratio >= bottom_y_start_ratio:
        return 0.08
    if y_ratio >= (bottom_y_start_ratio - 0.08):
        return 0.02
    if y_ratio >= (bottom_y_start_ratio - 0.18):
        return -0.10
    return -0.28


def _build_bottom_left_fallback_proposals(
    *,
    width: int,
    height: int,
    fallback_y_start_ratio: float,
    fallback_x_end_ratio: float,
    bottom_priority_y_start: float,
) -> list[BBoxTuple]:
    x_end_primary = _clamp_ratio(max(0.52, min(0.72, fallback_x_end_ratio + 0.06)), 0.45, 0.80)
    y_start_primary = _clamp_ratio(
        max(0.62, fallback_y_start_ratio, bottom_priority_y_start),
        0.50,
        0.92,
    )
    x0, y0, x1, y1 = _compute_bottom_focus_bbox(
        width=width,
        height=height,
        y_start_ratio=y_start_primary,
        x_end_ratio=x_end_primary,
    )
    primary = (x0, y0, x1 - x0, y1 - y0)

    x_end_tight = _clamp_ratio(min(0.78, x_end_primary + 0.05), 0.50, 0.85)
    y_start_tight = _clamp_ratio(max(0.70, y_start_primary + 0.08), 0.58, 0.95)
    tx0, ty0, tx1, ty1 = _compute_bottom_focus_bbox(
        width=width,
        height=height,
        y_start_ratio=y_start_tight,
        x_end_ratio=x_end_tight,
    )
    tight = (tx0, ty0, tx1 - tx0, ty1 - ty0)

    unique: list[BBoxTuple] = []
    seen: set[BBoxTuple] = set()
    for item in (primary, tight):
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _signature_shape_adjustment(
    *,
    bbox: BBoxTuple,
    width: int,
    height: int,
) -> float:
    if width <= 0 or height <= 0:
        return 0.0
    _, _, w, h = bbox
    w_ratio = float(w / width)
    h_ratio = float(h / height)
    if w_ratio >= 0.88 and h_ratio <= 0.34:
        # Caixa em faixa horizontal larga costuma capturar formulário/tabela.
        return -0.26
    if w_ratio >= 0.78 and h_ratio <= 0.30:
        return -0.18
    return 0.0


def _build_bottom_up_extraction_windows(crop: Image.Image) -> list[tuple[Image.Image, float, str]]:
    width, height = crop.size
    if width <= 4 or height <= 4:
        return [(crop, 0.0, "janela_completa")]

    windows_spec = [
        # Janela focada no bloco de assinatura/carimbo do médico (lado esquerdo-superior do recorte).
        (0.00, 0.08, 0.84, 0.74, 0.20, "janela_assinatura_superior_esquerda"),
        # Janela mais ampla para cobrir variações de layout no rodapé.
        (0.00, 0.18, 0.90, 0.88, 0.14, "janela_assinatura_central_esquerda"),
        # Mantém janela inferior para documentos onde o carimbo cai mais abaixo.
        (0.00, 0.48, 0.88, 1.00, 0.10, "janela_rodape_48_88"),
        (0.00, 0.00, 1.00, 1.00, 0.00, "janela_completa"),
    ]
    windows: list[tuple[Image.Image, float, str]] = []
    for x0_r, y0_r, x1_r, y1_r, bonus, label in windows_spec:
        x0 = max(0, min(width - 1, int(round(width * x0_r))))
        y0 = max(0, min(height - 1, int(round(height * y0_r))))
        x1 = max(x0 + 1, min(width, int(round(width * x1_r))))
        y1 = max(y0 + 1, min(height, int(round(height * y1_r))))
        if (x1 - x0) < 50 or (y1 - y0) < 40:
            continue
        windows.append((crop.crop((x0, y0, x1, y1)), float(bonus), label))
    if not windows:
        windows.append((crop, 0.0, "janela_completa"))
    return windows


@router.post(
    "/extrair-carimbo",
    response_model=ExtractResponse,
    responses={
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def extract_stamp(
    payload: ExtractRequest,
    settings: Settings = Depends(get_settings),
) -> ExtractResponse:
    file_bytes = _decode_file(payload.arquivo_base64)
    _enforce_file_size(file_bytes, settings.max_file_size_mb)
    image = _load_input_image(file_bytes, payload.mime_type, payload.pagina, settings)

    try:
        detection = detect_stamp_region(
            image=image,
            min_contour_area=settings.min_contour_area,
            padding_px=settings.padding_px,
            fallback_roi_y_start=settings.fallback_roi_y_start,
            fallback_roi_x_end=settings.fallback_roi_x_end,
        )
        crop_bbox = detection.bbox if detection.bbox is not None else detection.fallback_bbox
        x, y, w, h = crop_bbox
        crop = image.crop((x, y, x + w, y + h))
        processed_crop = preprocess_stamp(crop)
    except cv2.error as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro no processamento da imagem: {exc}",
        ) from exc

    carimbo_base64, carimbo_url = _build_image_outputs(
        processed_crop,
        settings=settings,
        include_base64=payload.retornar_imagem_base64,
        include_url=payload.retornar_imagem_url,
        artifact_prefix="carimbo",
    )

    return ExtractResponse(
        carimbo_base64=carimbo_base64,
        carimbo_url=carimbo_url,
        carimbo_encontrado=detection.found,
        confianca=float(detection.confidence),
        bbox=_to_bbox_model(detection.bbox),
        regiao_fallback=not detection.found,
        motivo=detection.reason,
        mensagem=detection.message,
    )


@router.post(
    "/extrair-carimbo/upload",
    response_model=ExtractResponse,
    responses={
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def extract_stamp_upload(
    file: UploadFile = File(...),
    pagina: int = Form(0),
    mime_type: Optional[str] = Form(None),
    retornar_imagem_base64: bool = Form(False),
    retornar_imagem_url: bool = Form(False),
    settings: Settings = Depends(get_settings),
) -> ExtractResponse:
    file_bytes = await file.read()
    _enforce_file_size(file_bytes, settings.max_file_size_mb)
    resolved_mime_type = _resolve_upload_mime_type(
        upload=file,
        mime_type_override=mime_type,
        file_bytes=file_bytes,
    )
    image = _load_input_image(
        file_bytes=file_bytes,
        mime_type=resolved_mime_type,
        pagina=max(0, int(pagina)),
        settings=settings,
    )

    try:
        detection = detect_stamp_region(
            image=image,
            min_contour_area=settings.min_contour_area,
            padding_px=settings.padding_px,
            fallback_roi_y_start=settings.fallback_roi_y_start,
            fallback_roi_x_end=settings.fallback_roi_x_end,
        )
        crop_bbox = detection.bbox if detection.bbox is not None else detection.fallback_bbox
        x, y, w, h = crop_bbox
        crop = image.crop((x, y, x + w, y + h))
        processed_crop = preprocess_stamp(crop)
    except cv2.error as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro no processamento da imagem: {exc}",
        ) from exc

    carimbo_base64, carimbo_url = _build_image_outputs(
        processed_crop,
        settings=settings,
        include_base64=retornar_imagem_base64,
        include_url=retornar_imagem_url,
        artifact_prefix="carimbo",
    )
    return ExtractResponse(
        carimbo_base64=carimbo_base64,
        carimbo_url=carimbo_url,
        carimbo_encontrado=detection.found,
        confianca=float(detection.confidence),
        bbox=_to_bbox_model(detection.bbox),
        regiao_fallback=not detection.found,
        motivo=detection.reason,
        mensagem=detection.message,
    )


@router.post(
    "/debug/visualizar",
    response_model=DebugResponse,
    responses={
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def debug_visualize(
    payload: ExtractRequest,
    settings: Settings = Depends(get_settings),
) -> DebugResponse:
    file_bytes = _decode_file(payload.arquivo_base64)
    _enforce_file_size(file_bytes, settings.max_file_size_mb)
    image = _load_input_image(file_bytes, payload.mime_type, payload.pagina, settings)

    try:
        detection = detect_stamp_region(
            image=image,
            min_contour_area=settings.min_contour_area,
            padding_px=settings.padding_px,
            fallback_roi_y_start=settings.fallback_roi_y_start,
            fallback_roi_x_end=settings.fallback_roi_x_end,
        )
    except cv2.error as exc:
        logger.exception("Falha OpenCV durante geração de debug")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro no processamento da imagem: {exc}",
        ) from exc

    debug_image = _draw_debug_overlay(
        image=image,
        bbox=detection.bbox,
        confidence=detection.confidence,
        detected=detection.found,
    )
    debug_base64, debug_url = _build_image_outputs(
        debug_image,
        settings=settings,
        include_base64=payload.retornar_imagem_base64,
        include_url=payload.retornar_imagem_url,
        artifact_prefix="debug",
    )
    return DebugResponse(
        imagem_debug_base64=debug_base64,
        imagem_debug_url=debug_url,
        bbox=_to_bbox_model(detection.bbox),
        carimbo_encontrado=detection.found,
        regiao_fallback=not detection.found,
        motivo=detection.reason,
    )


@router.post(
    "/debug/visualizar/upload",
    response_model=DebugResponse,
    responses={
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def debug_visualize_upload(
    file: UploadFile = File(...),
    pagina: int = Form(0),
    mime_type: Optional[str] = Form(None),
    retornar_imagem_base64: bool = Form(False),
    retornar_imagem_url: bool = Form(False),
    settings: Settings = Depends(get_settings),
) -> DebugResponse:
    file_bytes = await file.read()
    _enforce_file_size(file_bytes, settings.max_file_size_mb)
    resolved_mime_type = _resolve_upload_mime_type(
        upload=file,
        mime_type_override=mime_type,
        file_bytes=file_bytes,
    )
    image = _load_input_image(
        file_bytes=file_bytes,
        mime_type=resolved_mime_type,
        pagina=max(0, int(pagina)),
        settings=settings,
    )

    try:
        detection = detect_stamp_region(
            image=image,
            min_contour_area=settings.min_contour_area,
            padding_px=settings.padding_px,
            fallback_roi_y_start=settings.fallback_roi_y_start,
            fallback_roi_x_end=settings.fallback_roi_x_end,
        )
    except cv2.error as exc:
        logger.exception("Falha OpenCV durante geração de debug")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro no processamento da imagem: {exc}",
        ) from exc

    debug_image = _draw_debug_overlay(
        image=image,
        bbox=detection.bbox,
        confidence=detection.confidence,
        detected=detection.found,
    )
    debug_base64, debug_url = _build_image_outputs(
        debug_image,
        settings=settings,
        include_base64=retornar_imagem_base64,
        include_url=retornar_imagem_url,
        artifact_prefix="debug",
    )
    return DebugResponse(
        imagem_debug_base64=debug_base64,
        imagem_debug_url=debug_url,
        bbox=_to_bbox_model(detection.bbox),
        carimbo_encontrado=detection.found,
        regiao_fallback=not detection.found,
        motivo=detection.reason,
    )


@router.post(
    "/extrair-aso-geral",
    response_model=AsoGeralExtractResponse,
    responses={
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def extract_aso_geral_pipeline(
    payload: AsoGeralExtractRequest,
    settings: Settings = Depends(get_settings),
) -> AsoGeralExtractResponse:
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GEMINI_API_KEY não configurada",
        )

    file_bytes = _decode_file(payload.arquivo_base64)
    _enforce_file_size(file_bytes, settings.max_file_size_mb)
    image = _load_input_image(file_bytes, payload.mime_type, payload.pagina, settings)

    pipeline_started_at = time.monotonic()
    gemini_usage_events: list[dict[str, object]] = []

    timeout = settings.gemini_timeout_seconds
    retry_attempts = settings.gemini_retry_attempts
    retry_backoff_seconds = settings.gemini_retry_backoff_seconds
    retry_jitter_seconds = settings.gemini_retry_jitter_seconds
    extraction_retry_attempts = min(
        int(retry_attempts),
        max(0, int(settings.gemini_extraction_retry_attempts_cap)),
    )

    try:
        aso_payload = extract_aso_general_with_gemini(
            image=image,
            api_key=settings.gemini_api_key,
            model=settings.gemini_extraction_model,
            timeout_seconds=timeout,
            retry_attempts=extraction_retry_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            retry_jitter_seconds=retry_jitter_seconds,
            usage_sink=gemini_usage_events,
        )
    except GeminiServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Erro no Gemini (extração ASO geral): {exc}",
        ) from exc

    revisao_humana_recomendada, motivos_revisao = _build_aso_review_flags(aso_payload)
    gemini_telemetria = _build_gemini_telemetry(gemini_usage_events)
    elapsed_ms = int(max(0, round((time.monotonic() - pipeline_started_at) * 1000)))
    gemini_telemetria.latencia_total_ms = max(
        int(gemini_telemetria.latencia_total_ms),
        elapsed_ms,
    )

    return AsoGeralExtractResponse(
        empresa=AsoEmpresaInfo(**aso_payload["empresa"]),
        funcionario=AsoFuncionarioInfo(**aso_payload["funcionario"]),
        exame=AsoExameInfo(**aso_payload["exame"]),
        riscos=AsoRiscosInfo(**aso_payload["riscos"]),
        parecer=AsoParecerInfo(**aso_payload["parecer"]),
        origem=payload.origem,
        drive_item_id=payload.drive_item_id,
        folder_drive_id=payload.folder_drive_id,
        folder_name=payload.folder_name,
        user_code=payload.user_code,
        folder_path=payload.folder_path,
        folder_url=payload.folder_url,
        file_name=payload.file_name,
        file_web_url=payload.file_web_url,
        meta_queued_at=payload.meta_queued_at,
        revisao_humana_recomendada=revisao_humana_recomendada,
        motivos_revisao=motivos_revisao,
        gemini_telemetria=gemini_telemetria,
    )


@router.post(
    "/extrair-aso-geral/upload",
    response_model=AsoGeralExtractResponse,
    responses={
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def extract_aso_geral_upload(
    file: UploadFile = File(...),
    pagina: int = Form(0),
    mime_type: Optional[str] = Form(None),
    origem: Optional[str] = Form(None),
    drive_item_id: Optional[str] = Form(None),
    folder_drive_id: Optional[str] = Form(None),
    folder_name: Optional[str] = Form(None),
    user_code: Optional[str] = Form(None),
    folder_path: Optional[str] = Form(None),
    folder_url: Optional[str] = Form(None),
    file_name: Optional[str] = Form(None),
    file_web_url: Optional[str] = Form(None),
    meta_queued_at: Optional[str] = Form(None),
    settings: Settings = Depends(get_settings),
) -> AsoGeralExtractResponse:
    file_bytes = await file.read()
    _enforce_file_size(file_bytes, settings.max_file_size_mb)
    resolved_mime_type = _resolve_upload_mime_type(
        upload=file,
        mime_type_override=mime_type,
        file_bytes=file_bytes,
    )
    payload = AsoGeralExtractRequest(
        arquivo_base64=base64.b64encode(file_bytes).decode("utf-8"),
        mime_type=resolved_mime_type,
        pagina=max(0, int(pagina)),
        retornar_imagem_base64=False,
        retornar_imagem_url=False,
        origem=origem,
        drive_item_id=drive_item_id,
        folder_drive_id=folder_drive_id,
        folder_name=folder_name,
        user_code=user_code,
        folder_path=folder_path,
        folder_url=folder_url,
        file_name=file_name,
        file_web_url=file_web_url,
        meta_queued_at=meta_queued_at,
    )
    return extract_aso_geral_pipeline(payload=payload, settings=settings)


@router.post(
    "/extrair-medico-gemini",
    response_model=GeminiExtractResponse,
    responses={
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def extract_medico_with_gemini_pipeline(
    payload: GeminiExtractRequest,
    settings: Settings = Depends(get_settings),
) -> GeminiExtractResponse:
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GEMINI_API_KEY não configurada",
        )

    file_bytes = _decode_file(payload.arquivo_base64)
    _enforce_file_size(file_bytes, settings.max_file_size_mb)
    image = _load_input_image(file_bytes, payload.mime_type, payload.pagina, settings)

    pipeline_started_at = time.monotonic()
    gemini_usage_events: list[dict[str, object]] = []

    timeout = settings.gemini_timeout_seconds
    retry_attempts = settings.gemini_retry_attempts
    retry_backoff_seconds = settings.gemini_retry_backoff_seconds
    retry_jitter_seconds = settings.gemini_retry_jitter_seconds
    detection_retry_attempts = min(
        int(retry_attempts),
        max(0, int(settings.gemini_detection_retry_attempts_cap)),
    )
    extraction_retry_attempts = min(
        int(retry_attempts),
        max(0, int(settings.gemini_extraction_retry_attempts_cap)),
    )
    max_candidates = max(1, min(payload.max_candidatos, settings.gemini_max_candidates, 5))
    max_evaluations_cap = max(3, min(12, int(settings.gemini_max_evaluations)))
    max_evaluations = max(3, min(max_evaluations_cap, (max_candidates * 2) + 1))
    image_width, image_height = image.size

    selected_bbox: Optional[BBoxTuple] = None
    selected_crop: Optional[Image.Image] = None
    selected_medico: Optional[MedicoExtraction] = None
    selected_origin = "gemini_crop"
    selected_score = -1.0
    selected_reason = "gemini_detect_bbox"
    selected_detected = True
    selected_fallback_region = False

    candidates_evaluated = 0
    strategy = "gemini_duplo_estagio"
    reason = "gemini_detect_bbox"
    message = "Carimbo detectado e médico extraído via Gemini"

    gemini_candidates = []
    detection_error_detail = ""
    extraction_errors = 0
    bottom_y_start = _clamp_ratio(settings.stamp_bottom_priority_y_start, 0.35, 0.90)
    bottom_x_end = _clamp_ratio(settings.stamp_bottom_priority_x_end, 0.35, 1.00)
    focus_x0, focus_y0, focus_x1, focus_y1 = _compute_bottom_focus_bbox(
        width=image_width,
        height=image_height,
        y_start_ratio=bottom_y_start,
        x_end_ratio=bottom_x_end,
    )

    if settings.gemini_detection_enabled:
        # 1) Prioriza detecção Gemini na faixa inferior (onde o carimbo costuma estar).
        focus_image = image.crop((focus_x0, focus_y0, focus_x1, focus_y1))
        try:
            focused_candidates = detect_stamp_candidates_with_gemini(
                image=focus_image,
                api_key=settings.gemini_api_key,
                model=settings.gemini_detection_model,
                timeout_seconds=timeout,
                max_candidates=max_candidates,
                retry_attempts=detection_retry_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
                retry_jitter_seconds=retry_jitter_seconds,
                usage_sink=gemini_usage_events,
            )
            for candidate in focused_candidates:
                local_x, local_y, local_w, local_h = candidate.bbox
                global_bbox = (
                    focus_x0 + local_x,
                    focus_y0 + local_y,
                    local_w,
                    local_h,
                )
                gemini_candidates.append(
                    BBoxCandidate(
                        bbox=global_bbox,
                        score=min(1.0, candidate.score + 0.04),
                        reason="gemini_detect_bbox_foco_inferior",
                    )
                )
        except GeminiServiceError as exc:
            logger.warning("Falha no estágio Gemini de detecção (foco inferior): %s", exc)
            detection_error_detail = str(exc)

        # 2) Se o foco inferior não devolver candidatos, tenta página inteira.
        if not gemini_candidates:
            try:
                gemini_candidates = detect_stamp_candidates_with_gemini(
                    image=image,
                    api_key=settings.gemini_api_key,
                    model=settings.gemini_detection_model,
                    timeout_seconds=timeout,
                    max_candidates=max_candidates,
                    retry_attempts=detection_retry_attempts,
                    retry_backoff_seconds=retry_backoff_seconds,
                    retry_jitter_seconds=retry_jitter_seconds,
                    usage_sink=gemini_usage_events,
                )
            except GeminiServiceError as exc:
                logger.warning("Falha no estágio Gemini de detecção: %s", exc)
                detection_error_detail = str(exc)

    try:
        detection = detect_stamp_region(
            image=image,
            min_contour_area=settings.min_contour_area,
            padding_px=settings.padding_px,
            fallback_roi_y_start=settings.fallback_roi_y_start,
            fallback_roi_x_end=settings.fallback_roi_x_end,
        )
    except cv2.error as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro no processamento da imagem: {exc}",
        ) from exc

    ranked_proposals: list[tuple[BBoxTuple, float, str, bool, bool, str]] = []
    for candidate in gemini_candidates:
        variant_bboxes = build_stamp_crop_variants(
            bbox=candidate.bbox,
            width=image_width,
            height=image_height,
        )
        for variant_index, variant_bbox in enumerate(variant_bboxes):
            proposal_score = max(0.0, candidate.score - (0.03 * variant_index))
            ranked_proposals.append(
                (
                    variant_bbox,
                    proposal_score,
                    "gemini_crop",
                    True,
                    False,
                    candidate.reason or "gemini_detect_bbox",
                )
            )
            if len(ranked_proposals) >= max_evaluations:
                break
        if len(ranked_proposals) >= max_evaluations:
            break

    # Mesmo sem bbox do Gemini/OpenCV, força avaliação de ROIs no rodapé esquerdo.
    fallback_bottom_left = _build_bottom_left_fallback_proposals(
        width=image_width,
        height=image_height,
        fallback_y_start_ratio=_clamp_ratio(settings.fallback_roi_y_start, 0.40, 0.90),
        fallback_x_end_ratio=_clamp_ratio(settings.fallback_roi_x_end, 0.35, 0.85),
        bottom_priority_y_start=bottom_y_start,
    )
    for region_index, region_bbox in enumerate(fallback_bottom_left):
        ranked_proposals.append(
            (
                region_bbox,
                max(0.0, 0.74 - (0.05 * region_index)),
                "gemini_crop_fallback_opencv",
                True,
                True,
                "fallback_rodape_esquerdo_prioritario",
            )
        )
        if len(ranked_proposals) >= max_evaluations:
            break

    opencv_bbox = detection.bbox if detection.bbox is not None else detection.fallback_bbox
    opencv_variants = (
        build_stamp_crop_variants(opencv_bbox, image_width, image_height)[:2]
        if detection.found
        else [opencv_bbox]
    )
    for variant_index, variant_bbox in enumerate(opencv_variants):
        opencv_score = max(0.0, detection.confidence - (0.02 * variant_index))
        ranked_proposals.append(
            (
                variant_bbox,
                opencv_score,
                "gemini_crop_fallback_opencv",
                detection.found,
                not detection.found,
                detection.reason or "opencv_fallback",
            )
        )
        if len(ranked_proposals) >= max_evaluations:
            break

    if not gemini_candidates:
        strategy = "opencv_fallback_apos_gemini"
        reason = "gemini_sem_candidatos"
        message = "Gemini não retornou candidatos válidos. Usando fallback OpenCV."

    for bbox, proposal_score, origin, detected_flag, fallback_flag, proposal_reason in ranked_proposals:
        try:
            x, y, w, h = bbox
            crop = image.crop((x, y, x + w, y + h))
            processed_crop = preprocess_stamp(crop)
        except cv2.error as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Erro no processamento da imagem: {exc}",
            ) from exc

        medico = MedicoExtraction(
            nome=None,
            crm_numero=None,
            crm_uf=None,
            crm=None,
            confidence=0.0,
            valido=False,
            observacoes="sem_extracao",
        )
        local_best_score = -999.0
        local_errors: list[str] = []
        for extract_crop, window_bonus, window_label in _build_bottom_up_extraction_windows(processed_crop):
            try:
                extracted = extract_medico_with_gemini(
                    crop_image=extract_crop,
                    api_key=settings.gemini_api_key,
                    model=settings.gemini_extraction_model,
                    timeout_seconds=timeout,
                    retry_attempts=extraction_retry_attempts,
                    retry_backoff_seconds=retry_backoff_seconds,
                    retry_jitter_seconds=retry_jitter_seconds,
                    usage_sink=gemini_usage_events,
                )
            except GeminiServiceError as exc:
                extraction_errors += 1
                local_errors.append(f"{window_label}:{str(exc)[:120]}")
                continue

            obs_upper = extracted.observacoes.upper() if extracted.observacoes else ""
            header_penalty = (
                -0.40
                if ("CABECALHO" in obs_upper or "CABEÇALHO" in obs_upper or "PCMSO" in obs_upper)
                else 0.0
            )
            extracted_score = (
                float(extracted.confidence)
                + (0.25 if extracted.valido else -0.10)
                + float(window_bonus)
                + header_penalty
            )
            if extracted_score > local_best_score:
                local_best_score = extracted_score
                obs = extracted.observacoes or ""
                extracted.observacoes = f"{obs} | origem_janela={window_label}".strip(" |")
                medico = extracted

            if (
                extracted.valido
                and window_label != "janela_completa"
                and (extracted_score >= 0.92 or float(extracted.confidence) >= 0.72)
            ):
                break

        if local_best_score < -500.0 and local_errors:
            logger.warning("Falha no estágio Gemini de extração para um candidato: %s", local_errors[0])
            medico = MedicoExtraction(
                nome=None,
                crm_numero=None,
                crm_uf=None,
                crm=None,
                confidence=0.0,
                valido=False,
                observacoes=f"falha_extracao_gemini: {'; '.join(local_errors[:2])}",
            )

        candidates_evaluated += 1
        geometry_adjustment = bbox_geometry_adjustment(
            bbox=bbox,
            width=image_width,
            height=image_height,
        )
        bottom_adjustment = _bottom_priority_adjustment(
            bbox=bbox,
            image_height=image_height,
            bottom_y_start_ratio=bottom_y_start,
        )
        shape_adjustment = _signature_shape_adjustment(
            bbox=bbox,
            width=image_width,
            height=image_height,
        )
        combined_score = combined_candidate_score(
            proposal_score,
            medico,
            geometry_adjustment=geometry_adjustment + bottom_adjustment + shape_adjustment,
        )

        if combined_score > selected_score:
            selected_score = combined_score
            selected_bbox = bbox
            selected_crop = processed_crop
            selected_medico = medico
            selected_origin = origin
            selected_detected = detected_flag
            selected_fallback_region = fallback_flag
            selected_reason = proposal_reason

        y_ratio = float(bbox[1] / max(1, image_height))
        if medico.valido and combined_score >= 0.92 and y_ratio >= max(0.58, bottom_y_start - 0.04):
            break

    if selected_bbox is None or selected_crop is None or selected_medico is None:
        if extraction_errors > 0:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    "Erro no Gemini (extração): falha em todas as tentativas "
                    f"({extraction_errors})."
                ),
            )
        detail = "Nenhum candidato válido foi processado."
        if detection_error_detail:
            detail = f"{detail} Detalhe da detecção Gemini: {detection_error_detail}"
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail,
        )

    if selected_origin == "gemini_crop_fallback_opencv":
        strategy = (
            "opencv_fallback_apos_gemini"
            if not gemini_candidates
            else "gemini_duplo_estagio_com_fallback_opencv"
        )
        reason = selected_reason
        if not gemini_candidates:
            message = "Gemini não retornou candidatos válidos. Usando fallback OpenCV."
        else:
            message = "Seleção final obtida via fallback OpenCV + extração Gemini."
    else:
        reason = selected_reason
        if detection_error_detail:
            message = (
                "Detecção Gemini oscilou, mas a extração foi concluída com os candidatos disponíveis."
            )

    if not selected_medico.valido:
        message = (
            "Carimbo localizado, mas os dados do médico não passaram na validação "
            "de nome/CRM."
        )

    soc_threshold = max(0.0, min(1.0, float(settings.soc_name_similarity_threshold)))
    soc_config_ok = bool(
        settings.soc_base_url.strip()
        and settings.soc_empresa.strip()
        and settings.soc_codigo.strip()
        and settings.soc_chave.strip()
    )
    if settings.soc_enabled and not soc_config_ok:
        soc_validation = SocValidationResult(
            enabled=True,
            consulted=False,
            crm_consultado=selected_medico.crm_numero,
            uf_detectada=selected_medico.crm_uf,
            total_registros=0,
            nome_detectado=selected_medico.nome,
            melhor_nome_soc=None,
            melhor_crm_soc=None,
            melhor_uf_soc=None,
            similaridade_nome=0.0,
            limiar_similaridade=soc_threshold,
            nome_parecido=False,
            revisao_humana_recomendada=True,
            motivo="soc_configuracao_incompleta",
            erro="Preencha SOC_BASE_URL, SOC_EMPRESA, SOC_CODIGO e SOC_CHAVE.",
            amostra=[],
        )
    else:
        soc_validation = validate_with_soc(
            enabled=bool(settings.soc_enabled),
            crm_numero=selected_medico.crm_numero,
            crm_uf=selected_medico.crm_uf,
            nome_detectado=selected_medico.nome,
            threshold=soc_threshold,
            base_url=settings.soc_base_url,
            empresa=settings.soc_empresa,
            codigo=settings.soc_codigo,
            chave=settings.soc_chave,
            tipo_saida=settings.soc_tipo_saida,
            timeout_seconds=settings.soc_timeout_seconds,
        )

    if soc_validation.correcao_sugerida:
        crm_sugerido = soc_validation.crm_sugerido or soc_validation.crm_numero_sugerido or "indefinido"
        nome_sugerido = soc_validation.nome_sugerido_soc or "não informado"
        message = (
            "Carimbo detectado, mas a validação SOC sugere possível CRM truncado. "
            f"Sugestão: {crm_sugerido} | nome SOC: {nome_sugerido} "
            f"(sim={soc_validation.similaridade_nome_sugerida:.2f})."
        )
        suggestion_obs = (
            f"soc_sugestao_crm={crm_sugerido};"
            f"soc_nome={nome_sugerido};"
            f"sim={soc_validation.similaridade_nome_sugerida:.2f}"
        )
        if selected_medico.observacoes:
            selected_medico.observacoes = f"{selected_medico.observacoes} | {suggestion_obs}"
        else:
            selected_medico.observacoes = suggestion_obs

    revisao_humana_recomendada = bool(
        (not selected_medico.valido)
        or (not selected_detected)
        or selected_fallback_region
        or soc_validation.revisao_humana_recomendada
    )
    gemini_telemetria = _build_gemini_telemetry(gemini_usage_events)
    elapsed_ms = int(max(0, round((time.monotonic() - pipeline_started_at) * 1000)))
    gemini_telemetria.latencia_total_ms = max(
        int(gemini_telemetria.latencia_total_ms),
        elapsed_ms,
    )
    carimbo_base64, carimbo_url = _build_image_outputs(
        selected_crop,
        settings=settings,
        include_base64=payload.retornar_imagem_base64,
        include_url=payload.retornar_imagem_url,
        artifact_prefix="carimbo_gemini",
    )

    return GeminiExtractResponse(
        carimbo_base64=carimbo_base64,
        carimbo_url=carimbo_url,
        carimbo_encontrado=selected_detected,
        confianca=float(selected_score),
        bbox=_to_bbox_model(selected_bbox),
        regiao_fallback=selected_fallback_region,
        motivo=reason,
        mensagem=message,
        estrategia=strategy,
        candidatos_avaliados=candidates_evaluated,
        medico=_medico_info_model(
            extraction=selected_medico,
            origem=selected_origin,
        ),
        revisao_humana_recomendada=revisao_humana_recomendada,
        soc_validacao=_soc_validation_model(soc_validation),
        gemini_telemetria=gemini_telemetria,
    )


@router.post(
    "/extrair-medico-gemini/upload",
    response_model=GeminiExtractResponse,
    responses={
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def extract_medico_with_gemini_upload(
    file: UploadFile = File(...),
    pagina: int = Form(0),
    max_candidatos: int = Form(3),
    mime_type: Optional[str] = Form(None),
    retornar_imagem_base64: bool = Form(False),
    retornar_imagem_url: bool = Form(False),
    settings: Settings = Depends(get_settings),
) -> GeminiExtractResponse:
    file_bytes = await file.read()
    _enforce_file_size(file_bytes, settings.max_file_size_mb)
    resolved_mime_type = _resolve_upload_mime_type(
        upload=file,
        mime_type_override=mime_type,
        file_bytes=file_bytes,
    )
    payload = GeminiExtractRequest(
        arquivo_base64=base64.b64encode(file_bytes).decode("utf-8"),
        mime_type=resolved_mime_type,
        pagina=max(0, int(pagina)),
        max_candidatos=max(1, min(int(max_candidatos), 5)),
        retornar_imagem_base64=bool(retornar_imagem_base64),
        retornar_imagem_url=bool(retornar_imagem_url),
    )
    return extract_medico_with_gemini_pipeline(payload=payload, settings=settings)
