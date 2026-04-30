# carimbo-service

POC de microserviço em Python para localizar, recortar e pré-processar a região provável do carimbo médico em ASO. O serviço oferece:

- fluxo OpenCV clássico (heurística);
- fluxo Gemini em 2 estágios (detecção do carimbo + extração de nome/CRM).
- fluxo Gemini para extração dos dados gerais do ASO (empresa, funcionário, exame, riscos e parecer).

## Visão Geral

- Entrada: PDF ou imagem (`PNG`, `JPEG`, `TIFF`) em base64.
- Detecção: candidatos por Gemini + fallback OpenCV com score heurístico.
- Saída principal: crop do carimbo pré-processado em PNG base64.
- Saída Gemini: crop + `medico.nome` e `medico.crm` com validação no backend.
- Saída ASO geral: `empresa`, `funcionario`, `exame`, `riscos`, `parecer` (compatível com o fluxo atual do n8n).
- Debug: imagem completa com bbox desenhado e confiança.

## Estrutura

```text
carimbo-service/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── routers/
│   │   └── carimbo.py
│   ├── services/
│   │   ├── detector.py
│   │   ├── gemini_pipeline.py
│   │   ├── pdf_renderer.py
│   │   ├── preprocessor.py
│   │   └── soc_validator.py
│   └── schemas/
│       └── carimbo.py
├── tests/
│   ├── test_detector.py
│   ├── test_preprocessor.py
│   └── fixtures/
│       └── .gitkeep
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Como Rodar Com Docker

```bash
cd carimbo-service
docker compose up --build
```

Serviço disponível em `http://localhost:8001`.

## Como Rodar Localmente

```bash
cd carimbo-service
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Se aparecer erro na subida com `Form data requires "python-multipart" to be installed`, reinstale as dependências:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Exemplo de Chamada curl

```bash
curl -X POST "http://localhost:8000/extrair-carimbo" \
  -H "Content-Type: application/json" \
  -d '{
    "arquivo_base64": "SEU_BASE64_AQUI",
    "mime_type": "application/pdf",
    "pagina": 0
  }'
```

### Exemplo Gemini (2 estágios)

```bash
curl -X POST "http://localhost:8000/extrair-medico-gemini" \
  -H "Content-Type: application/json" \
  -d '{
    "arquivo_base64": "SEU_BASE64_AQUI",
    "mime_type": "application/pdf",
    "pagina": 0,
    "max_candidatos": 3
  }'
```

## Deploy no Dockploy

Arquivos preparados para deploy:

- `docker-compose.dockploy.yml`
- `.env.dockploy.example`

Passos sugeridos:

1. Suba este diretório (`carimbo-service`) no seu Git remoto.
2. No Dockploy, crie o serviço com Docker Compose apontando para `docker-compose.dockploy.yml`.
3. Cadastre as variáveis de ambiente com base no `.env.dockploy.example`.
4. Configure pelo menos: `GEMINI_API_KEY`, `SOC_EMPRESA`, `SOC_CODIGO`, `SOC_CHAVE` (se `SOC_ENABLED=true`).
5. Faça o deploy e valide:
   - `GET /health`
   - `GET /health/gemini`
6. No n8n, use os endpoints `/upload` com `multipart/form-data` e deixe:
   - `retornar_imagem_base64=false`
   - `retornar_imagem_url=false`

Notas operacionais:

- O serviço já possui `healthcheck` no compose.
- O volume `carimbo_artifacts` persiste imagens de artefato entre recriações do container.
- Se usar proxy/reverse proxy no Dockploy, mantenha a porta interna em `8000`.

## Variáveis de Ambiente (.env)

- `APP_NAME`: nome da aplicação.
- `VERSION`: versão exposta no `/health`.
- `PORT`: porta de execução.
- `DPI_RENDER`: DPI para renderizar PDF.
- `MIN_CONTOUR_AREA`: área mínima dos contornos.
- `PADDING_PX`: padding aplicado ao bbox detectado.
- `FALLBACK_ROI_Y_START`: início da ROI inferior (ratio em Y).
- `FALLBACK_ROI_X_END`: limite da ROI esquerda (ratio em X).
- `MAX_FILE_SIZE_MB`: limite máximo do arquivo decodificado.
- `GEMINI_API_KEY`: chave da API Gemini.
- `GEMINI_DETECTION_MODEL`: modelo do estágio 1 (detecção de bbox).
- `GEMINI_EXTRACTION_MODEL`: modelo do estágio 2 (extração de nome/CRM).
- `GEMINI_TIMEOUT_SECONDS`: timeout de chamada na API Gemini.
- `GEMINI_MAX_CANDIDATES`: limite de candidatos no estágio de detecção.
- `GEMINI_MAX_EVALUATIONS`: limite total de propostas avaliadas por documento.
- `GEMINI_DETECTION_RETRY_ATTEMPTS_CAP`: teto de retries para detecção Gemini.
- `GEMINI_EXTRACTION_RETRY_ATTEMPTS_CAP`: teto de retries para extração Gemini.
- `GEMINI_RETRY_ATTEMPTS`: quantidade de retries para falhas transitórias (429/5xx).
- `GEMINI_RETRY_BACKOFF_SECONDS`: base do backoff exponencial.
- `GEMINI_RETRY_JITTER_SECONDS`: jitter aleatório por tentativa.
- `STAMP_BOTTOM_PRIORITY_Y_START`: início (ratio Y) da faixa inferior priorizada.
- `STAMP_BOTTOM_PRIORITY_X_END`: limite (ratio X) horizontal da faixa inferior priorizada.
- `IMAGE_ARTIFACTS_DIR`: pasta local onde imagens PNG são salvas.
- `IMAGE_ARTIFACTS_URL_PREFIX`: prefixo HTTP para servir as imagens salvas.
- `SOC_ENABLED`: habilita validação cruzada com SOC no retorno Gemini.
- `SOC_BASE_URL`: endpoint base do SOC (`/WebSoc/exportadados`).
- `SOC_EMPRESA`: código da empresa no SOC.
- `SOC_CODIGO`: código do usuário/aplicação no SOC.
- `SOC_CHAVE`: chave de autenticação do SOC.
- `SOC_TIPO_SAIDA`: tipo de saída (normalmente `json`).
- `SOC_TIMEOUT_SECONDS`: timeout da chamada ao SOC.
- `SOC_NAME_SIMILARITY_THRESHOLD`: limiar (0 a 1) para considerar nome semelhante.

## Endpoint /extrair-medico-gemini

`POST /extrair-medico-gemini` executa pipeline em dois prompts:

1. detecta candidatos de bbox do carimbo;
2. extrai `nome` e `crm` do médico para cada candidato.

O backend valida os dados (nome e CRM/UF) e retorna o melhor candidato.

Quando o Gemini não devolve bbox válida, o endpoint usa fallback OpenCV para recorte e mantém a extração via Gemini.

Para facilitar visualização sem base64, os endpoints aceitam:

- `retornar_imagem_base64` (default `false`)
- `retornar_imagem_url` (default `false`)

Quando `retornar_imagem_url=true`, a resposta inclui `carimbo_url`/`imagem_debug_url` apontando para `/artifacts/...`.

Para integração com n8n (binary), use também:

- `POST /extrair-carimbo/upload` com `multipart/form-data` e campo `file`.
- `POST /debug/visualizar/upload` com `multipart/form-data` e campo `file`.
- `POST /extrair-medico-gemini/upload` com `multipart/form-data` e campo `file`.
- `POST /extrair-aso-geral/upload` com `multipart/form-data` e campo `file`.
- Campos opcionais via form-data: `pagina`, `max_candidatos`, `mime_type`,
  `retornar_imagem_base64`, `retornar_imagem_url`.

Se `SOC_ENABLED=true`, o endpoint também consulta o SOC por CRM e compara similaridade de nome para detectar casos de CRM truncado/incorreto (ex: `26807` vs `268072`). Quando há divergência, o serviço faz uma segunda tentativa por variação de sufixo (`0-9`) para sugerir correção. O retorno inclui:

- `soc_validacao`: análise detalhada da consulta SOC;
- `revisao_humana_recomendada`: decisão consolidada para fila humana.

## Endpoint /extrair-aso-geral

`POST /extrair-aso-geral` e `POST /extrair-aso-geral/upload` extraem dados gerais do ASO (sem médico/carimbo):

- `empresa`
- `funcionario`
- `exame`
- `riscos`
- `parecer`

Decisões de implementação:

1. O backend converte PDF para imagem antes de enviar ao Gemini (melhora robustez quando o PDF é lido como "página em branco").
2. O contrato dos blocos principais segue o mesmo formato usado hoje no node n8n para reduzir impacto no fluxo.
3. Metadados opcionais (`origem`, `drive_item_id`, `folder_*`, `file_*`, `meta_queued_at`) são aceitos e retornados no payload final.
4. O retorno inclui `gemini_telemetria`, `revisao_humana_recomendada` e `motivos_revisao` para auditoria operacional.

Exemplo de retorno (shape principal):

```json
{
  "empresa": { "razao_social": "...", "cnpj": "..." },
  "funcionario": { "nome": "...", "matricula": "...", "cpf": "...", "cargo": "...", "setor": "..." },
  "exame": { "tipo": "...", "data_aso": "DD/MM/AAAA" },
  "riscos": { "fisicos": "...", "quimicos": "...", "biologicos": "...", "ergonomicos": "...", "acidentes": "..." },
  "parecer": {
    "geral": "Apto|Inapto|Apto com Restrição|Não Aplicável|**",
    "trabalho_altura": "Apto|Inapto|Apto com Restrição|Não Aplicável|**",
    "espaco_confinado": "Apto|Inapto|Apto com Restrição|Não Aplicável|**",
    "trabalho_eletricidade": "Apto|Inapto|Apto com Restrição|Não Aplicável|**",
    "conducao_veiculos": "Apto|Inapto|Apto com Restrição|Não Aplicável|**",
    "operacao_maquinas": "Apto|Inapto|Apto com Restrição|Não Aplicável|**",
    "manipulacao_alimentos": "Apto|Inapto|Apto com Restrição|Não Aplicável|**"
  }
}
```

## Fluxo De Decisão Do Endpoint Gemini

`POST /extrair-medico-gemini` segue esta ordem:

1. tenta detectar `bbox` com Gemini;
2. prioriza a faixa inferior da página e só depois tenta página inteira;
3. gera variantes expandidas de recorte para evitar caixa apertada;
4. executa extração Gemini (`nome`, `crm_numero`, `crm_uf`) em cada variante;
5. calcula score combinado (detecção + extração + ajuste geométrico);
6. inclui propostas OpenCV no ranking final;
7. escolhe o melhor resultado entre Gemini/OpenCV.

Valores de `estrategia` atualmente possíveis:

- `gemini_duplo_estagio`
- `gemini_duplo_estagio_com_fallback_opencv`
- `opencv_fallback_apos_gemini`

## Contrato Mínimo Para Não Perder Dados

Persista sempre o JSON completo da resposta, mas trate estes campos como obrigatórios de trilha:

- `carimbo_encontrado`
- `confianca`
- `bbox.x`, `bbox.y`, `bbox.w`, `bbox.h` (quando `bbox != null`)
- `regiao_fallback`
- `motivo`
- `mensagem`
- `estrategia`
- `candidatos_avaliados`
- `gemini_telemetria.chamadas_total`
- `gemini_telemetria.prompt_tokens`
- `gemini_telemetria.output_tokens`
- `gemini_telemetria.total_tokens`
- `gemini_telemetria.latencia_total_ms`
- `revisao_humana_recomendada`
- `medico.nome`
- `medico.crm`
- `medico.crm_numero`
- `medico.crm_uf`
- `medico.confianca`
- `medico.valido`
- `medico.origem`
- `medico.observacoes`
- `soc_validacao.habilitada`
- `soc_validacao.consultada`
- `soc_validacao.nome_parecido`
- `soc_validacao.similaridade_nome`
- `soc_validacao.motivo`
- `soc_validacao.correcao_sugerida`
- `soc_validacao.crm_sugerido`
- `soc_validacao.nome_sugerido_soc`
- `soc_validacao.similaridade_nome_sugerida`
- `carimbo_base64` (ou o arquivo derivado salvo em storage)

Também persista metadados do documento para auditoria:

- `documento_id` do seu sistema
- `sha256` do arquivo original
- `pagina`
- `mime_type`
- `endpoint` chamado
- `version` do serviço (via `/health`)
- timestamp de processamento

## Política De Reprocessamento E Revisão

Sinalize para revisão manual quando qualquer condição abaixo ocorrer:

- `carimbo_encontrado == false`
- `medico.valido == false`
- `medico.crm == null`
- `regiao_fallback == true`
- `revisao_humana_recomendada == true`
- `soc_validacao.nome_parecido == false` (quando SOC habilitado)
- `soc_validacao.correcao_sugerida == true` (suspeita de CRM truncado)
- `confianca` muito baixa para sua régua interna

Recomendação prática:

1. não sobrescrever resultados antigos; grave nova tentativa com histórico;
2. manter versão do processamento (`version`) para comparabilidade;
3. ao reprocessar, armazenar vínculo com a tentativa anterior;
4. em auditoria visual, usar `/debug/visualizar`.

Guia operacional detalhado: `../docs/operacao_sem_perda.md`.

## Endpoint /debug/visualizar

`POST /debug/visualizar` recebe o mesmo payload do endpoint principal e retorna:

- imagem da página inteira em base64 com overlay de debug;
- bbox detectado (ou `null` em fallback);
- `carimbo_encontrado`, `regiao_fallback` e `motivo`.

Quando há detecção, o bbox é desenhado em vermelho com 3px. O label textual inclui `detected` ou `fallback` com a confiança.

## Integração com n8n

Para usar binário direto (sem base64) no n8n, configure o node `HTTP Request`:

- `Method`: `POST`
- `Send Body`: `Form-Data`
- campo `file`: tipo `n8n Binary File`, propriedade binária `data`
- campos de texto:
  - `pagina=0`
  - `retornar_imagem_base64=false`
  - `retornar_imagem_url=false`
  - `max_candidatos` apenas para `/extrair-medico-gemini/upload`

Observação: se o n8n enviar `application/octet-stream`, o backend tenta detectar automaticamente
se o binário é PDF/PNG/JPEG/TIFF pelo conteúdo do arquivo.

Endpoints recomendados para n8n:

- `POST /extrair-carimbo/upload`
- `POST /debug/visualizar/upload`
- `POST /extrair-medico-gemini/upload`

Para throughput alto, mantenha imagens desligadas por padrão:

- `retornar_imagem_base64=false`
- `retornar_imagem_url=false`

## Limitações Conhecidas Da POC

- Heurística focada em layout esperado (carimbo na região inferior esquerda).
- Pode degradar em documentos com artefatos fortes, baixa resolução extrema ou carimbo fora da ROI.
- Mesmo com Gemini em 2 estágios, ainda pode haver erro sem revisão humana em casos extremos.
- Não usa modelo de ML/deep learning.
