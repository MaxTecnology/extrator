from pdf2image import convert_from_bytes, pdfinfo_from_bytes
from PIL import Image


def render_page(pdf_bytes: bytes, pagina: int = 0, dpi: int = 300) -> Image.Image:
    """
    Renderiza uma página do PDF como imagem PIL RGB.
    Lança ValueError se pagina >= número de páginas.
    Usa pdf2image.convert_from_bytes com thread_count=2.
    """
    try:
        info = pdfinfo_from_bytes(pdf_bytes)
    except Exception as exc:  # pragma: no cover - depende do parser do poppler.
        raise ValueError("PDF não contém páginas") from exc

    total_pages = int(info.get("Pages", 0))
    if total_pages == 0:
        raise ValueError("PDF não contém páginas")
    if pagina >= total_pages:
        raise ValueError(f"PDF tem {total_pages} página(s). Índice {pagina} inválido.")

    pages = convert_from_bytes(
        pdf_bytes,
        dpi=dpi,
        first_page=pagina + 1,
        last_page=pagina + 1,
        thread_count=2,
    )
    if not pages:
        raise ValueError("PDF não contém páginas")

    return pages[0].convert("RGB")

