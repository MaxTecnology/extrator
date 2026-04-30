import base64
import binascii
from typing import Optional

from pydantic import BaseModel, Field, field_validator


SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/tiff",
}


class ExtractRequest(BaseModel):
    arquivo_base64: str
    mime_type: str
    pagina: int = Field(default=0, ge=0)
    retornar_imagem_base64: bool = False
    retornar_imagem_url: bool = False

    @field_validator("arquivo_base64")
    @classmethod
    def validate_base64(cls, value: str) -> str:
        cleaned_value = value.strip()
        if cleaned_value.startswith("data:") and "," in cleaned_value:
            cleaned_value = cleaned_value.split(",", 1)[1]
        try:
            base64.b64decode(cleaned_value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("arquivo_base64 não é um base64 válido") from exc
        return cleaned_value

    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, value: str) -> str:
        if value not in SUPPORTED_MIME_TYPES:
            raise ValueError(
                "mime_type deve ser application/pdf, image/png, image/jpeg ou image/tiff"
            )
        return value


class GeminiExtractRequest(ExtractRequest):
    max_candidatos: int = Field(default=3, ge=1, le=5)


class AsoGeralExtractRequest(ExtractRequest):
    origem: Optional[str] = None
    drive_item_id: Optional[str] = None
    folder_drive_id: Optional[str] = None
    folder_name: Optional[str] = None
    user_code: Optional[str] = None
    folder_path: Optional[str] = None
    folder_url: Optional[str] = None
    file_name: Optional[str] = None
    file_web_url: Optional[str] = None
    meta_queued_at: Optional[str] = None


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class ExtractResponse(BaseModel):
    carimbo_base64: Optional[str] = None
    carimbo_url: Optional[str] = None
    carimbo_encontrado: bool
    confianca: float
    bbox: Optional[BBox] = None
    regiao_fallback: bool
    motivo: str
    mensagem: str


class DebugResponse(BaseModel):
    imagem_debug_base64: Optional[str] = None
    imagem_debug_url: Optional[str] = None
    bbox: Optional[BBox] = None
    carimbo_encontrado: bool
    regiao_fallback: bool
    motivo: str


class ErrorResponse(BaseModel):
    erro: str
    detalhe: str


class HealthResponse(BaseModel):
    status: str
    version: str


class GeminiModelHealthInfo(BaseModel):
    modelo: str
    disponivel: bool
    erro: str = ""


class GeminiHealthResponse(BaseModel):
    status: str
    api_key_configurada: bool
    timeout_segundos: int
    detection: GeminiModelHealthInfo
    extraction: GeminiModelHealthInfo


class GeminiTelemetryInfo(BaseModel):
    chamadas_total: int = 0
    chamadas_deteccao: int = 0
    chamadas_extracao: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latencia_total_ms: int = 0
    tentativas_total: int = 0


class MedicoInfo(BaseModel):
    nome: Optional[str] = None
    crm: Optional[str] = None
    crm_numero: Optional[str] = None
    crm_uf: Optional[str] = None
    confianca: float = 0.0
    valido: bool = False
    origem: str
    observacoes: str = ""


class SocRecordInfo(BaseModel):
    cd_pessoa: str = ""
    nm_pessoa: str = ""
    nm_conselho: str = ""
    sg_ufconselho: str = ""
    cd_usuario: str = ""


class SocValidationInfo(BaseModel):
    habilitada: bool
    consultada: bool
    crm_consultado: Optional[str] = None
    uf_detectada: Optional[str] = None
    total_registros: int = 0
    nome_detectado: Optional[str] = None
    melhor_nome_soc: Optional[str] = None
    melhor_crm_soc: Optional[str] = None
    melhor_uf_soc: Optional[str] = None
    similaridade_nome: float = 0.0
    limiar_similaridade: float = 0.0
    nome_parecido: bool = False
    revisao_humana_recomendada: bool = False
    motivo: str
    erro: str = ""
    amostra: list[SocRecordInfo] = Field(default_factory=list)
    correcao_sugerida: bool = False
    crm_numero_sugerido: Optional[str] = None
    crm_uf_sugerida: Optional[str] = None
    crm_sugerido: Optional[str] = None
    nome_sugerido_soc: Optional[str] = None
    similaridade_nome_sugerida: float = 0.0
    variacoes_crm_consultadas: int = 0


class GeminiExtractResponse(BaseModel):
    carimbo_base64: Optional[str] = None
    carimbo_url: Optional[str] = None
    carimbo_encontrado: bool
    confianca: float
    bbox: Optional[BBox] = None
    regiao_fallback: bool
    motivo: str
    mensagem: str
    estrategia: str
    candidatos_avaliados: int
    medico: MedicoInfo
    revisao_humana_recomendada: bool = False
    soc_validacao: SocValidationInfo
    gemini_telemetria: GeminiTelemetryInfo = Field(default_factory=GeminiTelemetryInfo)


class AsoEmpresaInfo(BaseModel):
    razao_social: str
    cnpj: str


class AsoFuncionarioInfo(BaseModel):
    nome: str
    matricula: str
    cpf: str
    cargo: str
    setor: str


class AsoExameInfo(BaseModel):
    tipo: str
    data_aso: str


class AsoRiscosInfo(BaseModel):
    fisicos: str
    quimicos: str
    biologicos: str
    ergonomicos: str
    acidentes: str


class AsoParecerInfo(BaseModel):
    geral: str
    trabalho_altura: str
    espaco_confinado: str
    trabalho_eletricidade: str
    conducao_veiculos: str
    operacao_maquinas: str
    manipulacao_alimentos: str


class AsoGeralExtractResponse(BaseModel):
    empresa: AsoEmpresaInfo
    funcionario: AsoFuncionarioInfo
    exame: AsoExameInfo
    riscos: AsoRiscosInfo
    parecer: AsoParecerInfo
    origem: Optional[str] = None
    drive_item_id: Optional[str] = None
    folder_drive_id: Optional[str] = None
    folder_name: Optional[str] = None
    user_code: Optional[str] = None
    folder_path: Optional[str] = None
    folder_url: Optional[str] = None
    file_name: Optional[str] = None
    file_web_url: Optional[str] = None
    meta_queued_at: Optional[str] = None
    revisao_humana_recomendada: bool = False
    motivos_revisao: list[str] = Field(default_factory=list)
    gemini_telemetria: GeminiTelemetryInfo = Field(default_factory=GeminiTelemetryInfo)
