import logging
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.routers.carimbo import router as carimbo_router
from app.schemas.carimbo import ErrorResponse, HealthResponse


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


def _extract_validation_message(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "Requisição inválida"
    message = str(errors[0].get("msg", "Requisição inválida"))
    if message.startswith("Value error, "):
        return message.replace("Value error, ", "", 1)
    return message


settings_for_static = get_settings()
artifacts_dir = Path(settings_for_static.image_artifacts_dir)
artifacts_dir.mkdir(parents=True, exist_ok=True)
artifacts_prefix = "/" + settings_for_static.image_artifacts_url_prefix.strip("/")

app = FastAPI(title="carimbo-service")
app.mount(
    artifacts_prefix,
    StaticFiles(directory=str(artifacts_dir)),
    name="artifacts",
)
app.include_router(carimbo_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    error_tag = "erro_interno" if exc.status_code >= 500 else "erro_requisicao"
    body = ErrorResponse(erro=error_tag, detalhe=str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    body = ErrorResponse(erro="erro_requisicao", detalhe=_extract_validation_message(exc))
    return JSONResponse(status_code=422, content=body.model_dump())


@app.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(status="ok", version=settings.version)
