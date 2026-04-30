from app.services.gemini_pipeline import (
    bbox_geometry_adjustment,
    build_stamp_crop_variants,
    _extract_first_json_object,
    expand_stamp_bbox,
    _normalize_medico_payload,
    _sanitize_bbox_candidates,
)


def test_extract_first_json_object_from_markdown_code_block() -> None:
    raw = """
```json
{"candidatos":[{"bbox":{"x":10,"y":20,"w":100,"h":80},"score":0.9}]}
```
"""
    data = _extract_first_json_object(raw)
    assert "candidatos" in data
    assert data["candidatos"][0]["bbox"]["x"] == 10


def test_sanitize_bbox_candidates_clamps_bounds_and_limits_results() -> None:
    payload = {
        "candidatos": [
            {"bbox": {"x": -10, "y": 5, "w": 80, "h": 40}, "score": 0.8},
            {"bbox": {"x": 90, "y": 95, "w": 30, "h": 40}, "score": 0.7},
            {"bbox": {"x": 20, "y": 20, "w": 10, "h": 10}, "score": 0.6},
        ]
    }
    candidates = _sanitize_bbox_candidates(payload, width=100, height=100, max_candidates=2)
    assert len(candidates) == 2
    assert candidates[0].bbox == (0, 5, 80, 40)
    assert candidates[1].bbox == (90, 95, 10, 5)


def test_normalize_medico_payload_parses_crm_and_validates_uf() -> None:
    payload = {
        "nome": "Dr Lucas Bocatão de Paula",
        "crm": "60750/PR",
        "confianca": 0.88,
    }
    medico = _normalize_medico_payload(payload)
    assert medico.nome == "Dr Lucas Bocatão de Paula"
    assert medico.crm_numero == "60750"
    assert medico.crm_uf == "PR"
    assert medico.crm == "60750/PR"
    assert medico.valido is True
    assert medico.confidence == 0.88


def test_normalize_medico_payload_parses_crm_with_dot_separator() -> None:
    payload = {
        "nome": "Lucas Bocalao de Paula",
        "crm": "CRM-PR: 60.750",
        "confianca": 0.82,
    }
    medico = _normalize_medico_payload(payload)
    assert medico.crm_numero == "60750"
    assert medico.crm_uf == "PR"
    assert medico.crm == "60750/PR"
    assert medico.valido is True


def test_normalize_medico_payload_sanitizes_crm_numero_field() -> None:
    payload = {
        "nome": "Lucas Bocalao de Paula",
        "crm_numero": "60.750",
        "crm_uf": "PR",
        "confianca": 0.81,
    }
    medico = _normalize_medico_payload(payload)
    assert medico.crm_numero == "60750"
    assert medico.crm_uf == "PR"
    assert medico.crm == "60750/PR"
    assert medico.valido is True


def test_normalize_medico_payload_marks_header_source_as_invalid() -> None:
    payload = {
        "nome": "FLAVIO HISSAO SALVION UETA",
        "crm_numero": "107700",
        "crm_uf": "SP",
        "confianca": 0.99,
        "observacoes": "Dados extraídos do cabeçalho do documento (PCMSO).",
    }
    medico = _normalize_medico_payload(payload)
    assert medico.crm == "107700/SP"
    assert medico.valido is False
    assert medico.confidence <= 0.35
    assert "suspeita_origem_cabecalho_pcms0" in medico.observacoes


def test_expand_stamp_bbox_increases_context_for_small_label_like_bbox() -> None:
    expanded = expand_stamp_bbox(
        bbox=(220, 1080, 120, 28),
        width=1200,
        height=1600,
    )
    x, y, w, h = expanded
    assert x >= 0
    assert y >= 0
    assert w >= int(1200 * 0.24)
    assert h >= int(1600 * 0.16)
    assert y <= 1080


def test_bbox_geometry_adjustment_penalizes_tiny_boxes() -> None:
    tiny = bbox_geometry_adjustment((100, 900, 60, 30), width=1200, height=1600)
    regular = bbox_geometry_adjustment((120, 900, 360, 280), width=1200, height=1600)
    assert tiny < regular


def test_build_stamp_crop_variants_returns_unique_bboxes() -> None:
    variants = build_stamp_crop_variants((200, 950, 180, 80), width=1200, height=1600)
    assert len(variants) >= 2
    assert len(variants) == len(set(variants))
