from app.services.soc_validator import (
    SocRecord,
    build_soc_request_url,
    compute_name_similarity,
    evaluate_soc_records,
    normalize_person_name,
    validate_with_soc,
)


def test_normalize_person_name_strips_accents_and_symbols() -> None:
    value = normalize_person_name("Dra. Rafaéla-Cruz  Jandaia")
    assert value == "DRA RAFAELA CRUZ JANDAIA"


def test_compute_name_similarity_high_for_same_person_with_small_noise() -> None:
    score = compute_name_similarity(
        "Rafael Cruz Jandaia",
        "Rafaela Cruz Jandaia",
    )
    assert score >= 0.80


def test_build_soc_request_url_contains_parametro_query() -> None:
    url = build_soc_request_url(
        base_url="https://ws1.soc.com.br/WebSoc/exportadados",
        empresa="710",
        codigo="3001",
        chave="abc",
        tipo_saida="json",
        conselho_classe="26807",
    )
    assert "parametro=" in url
    assert "WebSoc/exportadados?" in url


def test_validate_with_soc_disabled() -> None:
    result = validate_with_soc(
        enabled=False,
        crm_numero="26807",
        crm_uf="PE",
        nome_detectado="Rafael Cruz Jandaia",
        threshold=0.78,
        base_url="https://ws1.soc.com.br/WebSoc/exportadados",
        empresa="710",
        codigo="3001",
        chave="abc",
        tipo_saida="json",
        timeout_seconds=5,
    )
    assert result.enabled is False
    assert result.consulted is False
    assert result.revisao_humana_recomendada is False
    assert result.motivo == "soc_desabilitado"


def test_name_similarity_flags_review_for_different_person() -> None:
    result = evaluate_soc_records(
        records=[
            SocRecord(
                cd_pessoa="127519",
                nm_pessoa="Henrique Andrade Furtado",
                cd_cpf="",
                cd_conselho="CRM",
                nm_conselho="26807",
                sg_ufconselho="PE",
                cd_usuario="323794",
            )
        ],
        crm_numero="26807",
        crm_uf="PE",
        nome_detectado="Rafael Cruz Jandaia",
        threshold=0.78,
    )
    assert result.consulted is True
    assert result.nome_parecido is False
    assert result.revisao_humana_recomendada is True
    assert result.motivo == "soc_crm_encontrado_nome_divergente"


def test_name_similarity_accepts_close_name() -> None:
    result = evaluate_soc_records(
        records=[
            SocRecord(
                cd_pessoa="127519",
                nm_pessoa="Rafael Cruz Jandaia",
                cd_cpf="",
                cd_conselho="CRM",
                nm_conselho="268072",
                sg_ufconselho="PE",
                cd_usuario="323794",
            )
        ],
        crm_numero="268072",
        crm_uf="PE",
        nome_detectado="Dra Rafaela Cruz Jandaia",
        threshold=0.78,
    )
    assert result.nome_parecido is True
    assert result.revisao_humana_recomendada is False
    assert result.motivo == "soc_ok_nome_e_crm_compativeis"


def test_validate_with_soc_suggests_suffix_when_crm_looks_truncated(monkeypatch) -> None:
    def fake_query_soc_by_crm(
        *,
        base_url: str,
        empresa: str,
        codigo: str,
        chave: str,
        tipo_saida: str,
        crm_numero: str,
        timeout_seconds: int,
    ) -> list[SocRecord]:
        if crm_numero == "6075":
            return [
                SocRecord(
                    cd_pessoa="114488",
                    nm_pessoa="Wanderson Reis Sales Vilela",
                    cd_cpf="",
                    cd_conselho="CRM",
                    nm_conselho="6075",
                    sg_ufconselho="MT",
                    cd_usuario="127622",
                ),
                SocRecord(
                    cd_pessoa="110266",
                    nm_pessoa="Martha Batista Guimaraes",
                    cd_cpf="",
                    cd_conselho="CRM",
                    nm_conselho="6075",
                    sg_ufconselho="AM",
                    cd_usuario="106035",
                ),
            ]
        if crm_numero == "60750":
            return [
                SocRecord(
                    cd_pessoa="200999",
                    nm_pessoa="Lucas Bocalao de Paula",
                    cd_cpf="",
                    cd_conselho="CRM",
                    nm_conselho="60750",
                    sg_ufconselho="PR",
                    cd_usuario="999001",
                )
            ]
        return []

    monkeypatch.setattr(
        "app.services.soc_validator.query_soc_by_crm",
        fake_query_soc_by_crm,
    )

    result = validate_with_soc(
        enabled=True,
        crm_numero="6075",
        crm_uf="RR",
        nome_detectado="Lucas Bocalão",
        threshold=0.78,
        base_url="https://ws1.soc.com.br/WebSoc/exportadados",
        empresa="710",
        codigo="3001",
        chave="abc",
        tipo_saida="json",
        timeout_seconds=5,
    )
    assert result.consulted is True
    assert result.correcao_sugerida is True
    assert result.crm_numero_sugerido == "60750"
    assert result.crm_uf_sugerida == "PR"
    assert result.crm_sugerido == "60750/PR"
    assert result.nome_sugerido_soc == "Lucas Bocalao de Paula"
    assert result.similaridade_nome_sugerida >= 0.60
    assert result.revisao_humana_recomendada is True
    assert result.motivo == "soc_sugere_crm_truncado_por_nome_compativel"
    assert result.variacoes_crm_consultadas > 0
