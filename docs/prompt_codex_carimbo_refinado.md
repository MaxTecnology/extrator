# Prompt para o Codex — POC de Extração de Carimbo Médico

## Objetivo

Crie uma **POC funcional e organizada** de um microserviço em **Python** para localizar, recortar e pré-processar a região do **carimbo médico** em documentos de ASO (Atestado de Saúde Ocupacional), para que essa imagem recortada seja enviada depois para um prompt especializado do Gemini.

O problema real é o seguinte:
- o pipeline principal no **n8n** já usa Gemini para extrair os dados do documento;
- os campos `medico.crm` e `medico.nome` costumam falhar porque o **carimbo médico está sobreposto à assinatura manuscrita**;
- a POC deve focar em **isolar melhor a área do carimbo** e devolver esse crop em base64;
- **não** deve integrar com Gemini agora;
- **não** deve usar ML/deep learning;
- a solução deve usar apenas **OpenCV clássico + heurísticas**.

A entrega deve ser pensada para uso real, com código limpo, estrutura boa de projeto, Docker, testes unitários e README útil.

---

## Escopo da POC

O microserviço deve:
1. Receber um **PDF** ou uma **imagem** do ASO em base64.
2. Renderizar a página escolhida se for PDF.
3. Detectar a **região provável do carimbo** com OpenCV clássico.
4. Recortar essa região com padding.
5. Pré-processar o recorte para melhorar a futura leitura.
6. Retornar o crop final em **PNG base64**.
7. Expor também um endpoint de debug que desenha o bbox detectado na página inteira.

A POC **não precisa ler CRM**, **não precisa OCR**, **não precisa Gemini**. Ela só precisa encontrar e preparar a melhor região possível do carimbo.

---

## Stack obrigatória

Use exatamente:
- **Python 3.11+**
- **FastAPI** com **Uvicorn**
- **OpenCV** (`opencv-python-headless`)
- **Pillow**
- **pdf2image**
- **NumPy**
- **Pydantic v2**
- **pydantic-settings**
- `poppler-utils` no Dockerfile
- **pytest** para testes unitários

---

## Estrutura de diretórios esperada

```text
carimbo-service/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── routers/
│   │   └── carimbo.py
│   ├── services/
│   │   ├── pdf_renderer.py
│   │   ├── detector.py
│   │   └── preprocessor.py
│   ├── schemas/
│   │   └── carimbo.py
│   └── config.py
├── tests/
│   ├── test_detector.py
│   ├── test_preprocessor.py
│   └── fixtures/
│       └── .gitkeep
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Requisitos obrigatórios de implementação

### 1. Endpoint principal

Implemente:

### `POST /extrair-carimbo`

**Content-Type:** `application/json`

### Request body

```json
{
  "arquivo_base64": "string (base64 do PDF ou imagem PNG/JPG/TIFF)",
  "mime_type": "application/pdf | image/png | image/jpeg | image/tiff",
  "pagina": 0
}
```

### Regras do request
- `pagina` deve ser **zero-based**.
- Se `pagina` não for enviada, assumir `0`.
- Se for PDF, processar **apenas a página solicitada**.
- Não processar todas as páginas automaticamente.

### Response de sucesso quando detectar carimbo

```json
{
  "carimbo_base64": "string (base64 PNG do carimbo recortado e pré-processado)",
  "carimbo_encontrado": true,
  "confianca": 0.92,
  "bbox": {
    "x": 120,
    "y": 850,
    "w": 380,
    "h": 180
  },
  "regiao_fallback": false,
  "motivo": "detectado_por_contorno",
  "mensagem": "Carimbo detectado por contorno"
}
```

### Response de sucesso com fallback

```json
{
  "carimbo_base64": "string (base64 PNG do fallback recortado e pré-processado)",
  "carimbo_encontrado": false,
  "confianca": 0.45,
  "bbox": null,
  "regiao_fallback": true,
  "motivo": "fallback_regiao_inferior",
  "mensagem": "Carimbo não detectado. Retornando região de fallback (quadrante inferior esquerdo)"
}
```

### Response de erro

```json
{
  "erro": "string",
  "detalhe": "string"
}
```

---

### 2. Endpoint de debug

Implemente:

### `POST /debug/visualizar`

Recebe o mesmo body do `/extrair-carimbo`.

Retorna a página inteira com o **bbox desenhado em vermelho**, espessura **3px**, e com um label textual contendo:
- `detected` ou `fallback`
- valor da `confianca`

### Response

```json
{
  "imagem_debug_base64": "string",
  "bbox": { "x": 120, "y": 850, "w": 380, "h": 180 },
  "carimbo_encontrado": true,
  "regiao_fallback": false,
  "motivo": "detectado_por_contorno"
}
```

Se não houver bbox, retornar a imagem original com o texto indicando fallback.

---

### 3. Endpoint de saúde

Implemente:

### `GET /health`

```json
{ "status": "ok", "version": "1.0.0" }
```

---

## Lógica obrigatória do detector (`detector.py`)

Implemente **exatamente esta sequência base**, mas organize o código de forma limpa e testável.

### Fluxo base obrigatório

```text
1. Recebe imagem PIL (RGB)
2. Converte para numpy array
3. Converte para escala de cinza
4. Aplica GaussianBlur com kernel 3x3
5. Aplica binarização adaptativa:
   - método: ADAPTIVE_THRESH_GAUSSIAN_C
   - blockSize: 11
   - C: 2
   - tipo: THRESH_BINARY_INV
6. Define ROI principal = quadrante inferior esquerdo da imagem:
   - y: 55% até 100% da altura
   - x: 0% até 55% da largura
7. Dentro do ROI, encontra contornos externos com:
   - RETR_EXTERNAL
   - CHAIN_APPROX_SIMPLE
8. Filtra contornos com:
   - área > min_contour_area
   - aspect ratio entre 0.5 e 5.0
9. Seleciona o melhor contorno entre os filtrados
10. Extrai bounding box
11. Aplica padding em todos os lados respeitando os limites da imagem
12. Retorna bbox + confiança
13. Se nenhum contorno passar no filtro, retorna bbox=None
```

### Critério de score do candidato

Não escolha apenas o maior contorno cegamente. Calcule um **score de candidato** levando em conta:
- área do contorno
- posição dentro da ROI inferior
- densidade de pixels escuros dentro do bbox
- retangularidade simples do contorno

Use esse score para escolher o melhor candidato.

### Cálculo da confiança

O campo `confianca` deve ser um `float` entre `0.0` e `1.0`.

A confiança deve ser baseada em uma combinação objetiva de:
- área do contorno normalizada
- densidade de pixels escuros
- posição esperada dentro da região inferior
- retangularidade

Regras obrigatórias:
- limitar entre `0.0` e `1.0`
- se cair no fallback, a confiança deve ser **no máximo `0.45`**
- se nenhum candidato existir, usar fallback com confiança baixa

### Regra do fallback

Se nenhum contorno passar no filtro:
- usar ROI fallback com:
  - `y` de `55%` até `100%`
  - `x` de `0%` até `55%`
- `regiao_fallback = true`
- `motivo = "fallback_regiao_inferior"`
- `carimbo_encontrado = false`
- `bbox = null`
- `confianca <= 0.45`

### Regra de bbox
- `bbox` nunca pode ultrapassar os limites da imagem
- o padding deve ser aplicado antes do crop final
- o bbox retornado deve refletir o valor **já com padding aplicado**

---

## Lógica obrigatória do pré-processamento (`preprocessor.py`)

Recebe a imagem PIL recortada e aplica na sequência:

```text
1. Redimensiona para largura mínima de 600px, mantendo aspect ratio, apenas se menor
2. Converte para escala de cinza
3. Aplica CLAHE:
   - clipLimit = 2.0
   - tileGridSize = (8, 8)
4. Aplica deskew:
   - detectar linhas com HoughLinesP
   - calcular ângulo mediano das linhas detectadas
   - rotacionar apenas se ângulo > 0.5 grau e < 15 graus
   - usar INTER_CUBIC
   - preencher bordas com branco (255)
5. Aplicar denoising leve com fastNlMeansDenoising(h=10)
6. Aplicar binarização final com Otsu
7. Retornar imagem PIL modo 'L'
```

### Regras adicionais do pré-processador
- não lançar erro em imagem quase branca ou vazia
- se não houver linhas suficientes para deskew, manter imagem como está
- saída final deve ser sempre uma imagem compatível com exportação em **PNG grayscale**

---

## Lógica do renderizador de PDF (`pdf_renderer.py`)

Crie uma função assim:

```python
def render_page(pdf_bytes: bytes, pagina: int = 0, dpi: int = 300) -> PIL.Image:
    """
    Renderiza uma página do PDF como imagem PIL RGB.
    Lança ValueError se pagina >= número de páginas.
    Usa pdf2image.convert_from_bytes com thread_count=2.
    """
```

Regras:
- se PDF não tiver páginas, lançar erro tratado
- usar `dpi` vindo das configurações
- retornar imagem em RGB

---

## Configuração (`config.py`)

Use `pydantic-settings` com `.env`.

Crie uma classe `Settings` com pelo menos:

```python
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

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
```

---

## Tratamento de erros obrigatório

Implemente exatamente estes cenários:

| Situação | HTTP | Mensagem |
|---|---:|---|
| base64 inválido | 422 | `arquivo_base64 não é um base64 válido` |
| mime_type não suportado | 422 | `mime_type deve ser application/pdf, image/png, image/jpeg ou image/tiff` |
| PDF com 0 páginas | 422 | `PDF não contém páginas` |
| pagina >= total de páginas | 422 | `PDF tem {n} página(s). Índice {pagina} inválido.` |
| arquivo > max_file_size_mb | 413 | `Arquivo excede o limite de {n}MB` |
| erro interno no OpenCV | 500 | `Erro no processamento da imagem: {detalhe}` |

Padronize as respostas de erro em JSON.

---

## O que NÃO fazer

- Não usar modelos de ML/deep learning como YOLO, SAM, TensorFlow, PyTorch, etc.
- Não criar frontend
- Não integrar com Gemini ou qualquer outro LLM
- Não salvar arquivos temporários em disco durante o processamento
- Não usar `cv2.imshow`
- Não criar banco de dados
- Não criar cache persistente
- Não criar testes de integração com FastAPI (`TestClient`)

---

## Testes obrigatórios (`tests/`)

Escreva **testes unitários** com `pytest` cobrindo:

### `test_detector.py`
1. imagem sintética com retângulo preenchido no quadrante inferior esquerdo → deve detectar com `confianca > 0.5`
2. imagem em branco → deve retornar `bbox=None` e cair no fallback
3. imagem com retângulo no quadrante superior direito → não deve considerar como carimbo válido e deve cair no fallback
4. imagem sintética com bloco preto e linhas azuis por cima simulando assinatura → ainda deve retornar uma região candidata válida se o bloco estiver na ROI esperada

### `test_preprocessor.py`
1. imagem pequena com largura < 600px → deve ser upscalada
2. imagem com inclinação de 5 graus → deve corrigir para ângulo final aproximado < 1 grau
3. imagem em branco → não deve lançar exceção

### Regras dos testes
- apenas testes unitários de services
- não testar endpoint com cliente HTTP
- deixar a pasta `tests/fixtures/` criada com `.gitkeep`

---

## Requisitos de performance e implementação

- evitar cópias desnecessárias de imagem em memória
- não fazer processamento mais pesado do que o necessário
- projetar código para rodar bem em container simples
- processar um único arquivo por requisição
- manter boa legibilidade e separação de responsabilidades
- usar type hints
- usar logging básico
- escrever funções pequenas e fáceis de testar

---

## Dockerfile obrigatório

Monte um `Dockerfile` funcional com:
- base `python:3.11-slim`
- instalação de `poppler-utils`
- dependências mínimas do OpenCV headless
- `WORKDIR /app`
- copiar `requirements.txt`
- instalar requirements
- copiar código da aplicação
- expor porta 8000
- rodar com Uvicorn

---

## docker-compose.yml obrigatório

Monte um `docker-compose.yml` simples com:
- serviço `carimbo-service`
- build local
- porta `8001:8000`
- variáveis de ambiente mínimas
- restart policy
- healthcheck chamando `/health`

---

## README.md obrigatório

O README deve conter:
1. visão geral do projeto
2. como rodar com Docker
3. como rodar localmente
4. exemplo de chamada `curl`
5. descrição das variáveis do `.env`
6. explicação do endpoint `/debug/visualizar`
7. seção de integração com n8n
8. limitações conhecidas da POC

### Snippet obrigatório de integração com n8n

Inclua no README exatamente um exemplo em JavaScript parecido com este:

```javascript
// No Code node do n8n, antes do Gemini
const binaryData = await this.helpers.getBinaryDataBuffer(0, 'data');
const pdfBase64 = binaryData.toString('base64');

const carimboResponse = await this.helpers.httpRequest({
  method: 'POST',
  url: 'http://carimbo-service:8001/extrair-carimbo',
  headers: { 'Content-Type': 'application/json' },
  body: {
    arquivo_base64: pdfBase64,
    mime_type: 'application/pdf',
    pagina: 0,
  },
  json: true,
  timeout: 30000,
});

// carimboResponse.carimbo_base64 -> enviar depois ao Gemini
// carimboResponse.regiao_fallback -> usar como sinal de baixa confiança
```

---

## Qualidade do código esperada

- código limpo e profissional
- boa separação entre router, services, schemas e config
- tratamento de exceções consistente
- respostas tipadas com Pydantic
- comentários apenas quando agregarem valor
- README claro
- projeto pronto para subir e testar

---

## Entrega esperada do Codex

Quero que você gere **todos os arquivos do projeto**, com conteúdo completo, incluindo:
- código da aplicação
- testes unitários
- Dockerfile
- docker-compose.yml
- requirements.txt
- `.env.example`
- README.md

A implementação deve ser **executável**, coerente entre os arquivos e pronta para rodar.
