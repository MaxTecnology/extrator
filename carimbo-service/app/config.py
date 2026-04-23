from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "carimbo-service"
    version: str = "1.0.0"
    port: int = 8000
    dpi_render: int = 300
    min_contour_area: int = 4000
    padding_px: int = 20
    fallback_roi_y_start: float = 0.55
    fallback_roi_x_end: float = 0.55
    max_file_size_mb: int = 20
    gemini_api_key: str = ""
    gemini_detection_enabled: bool = False
    gemini_detection_model: str = "gemini-2.5-flash"
    gemini_extraction_model: str = "gemini-2.5-flash"
    gemini_timeout_seconds: int = 30
    gemini_max_candidates: int = 3
    gemini_max_evaluations: int = 6
    gemini_detection_retry_attempts_cap: int = 2
    gemini_extraction_retry_attempts_cap: int = 1
    gemini_retry_attempts: int = 3
    gemini_retry_backoff_seconds: float = 1.2
    gemini_retry_jitter_seconds: float = 0.35
    stamp_bottom_priority_y_start: float = 0.60
    stamp_bottom_priority_x_end: float = 0.80
    image_artifacts_dir: str = "/tmp/carimbo-artifacts"
    image_artifacts_url_prefix: str = "/artifacts"
    soc_enabled: bool = False
    soc_base_url: str = "https://ws1.soc.com.br/WebSoc/exportadados"
    soc_empresa: str = ""
    soc_codigo: str = ""
    soc_chave: str = ""
    soc_tipo_saida: str = "json"
    soc_timeout_seconds: int = 15
    soc_name_similarity_threshold: float = 0.78

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
