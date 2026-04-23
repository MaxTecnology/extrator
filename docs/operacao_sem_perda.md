# Operação Sem Perda De Dados

Este guia define como operar o `carimbo-service` sem perder rastreabilidade de extração.

## 1) Como O Serviço Funciona Hoje

Endpoints principais:

- `POST /extrair-carimbo`: recorte por OpenCV.
- `POST /debug/visualizar`: página com overlay de bbox para auditoria visual.
- `POST /extrair-medico-gemini`: detecção + extração do médico.

No endpoint Gemini, a seleção final pode vir de:

- `gemini_duplo_estagio`
- `gemini_duplo_estagio_com_fallback_opencv`
- `opencv_fallback_apos_gemini`

## 2) Regra De Ouro: Não Descartar Resposta

Sempre persista o JSON completo retornado pela API.

Campos críticos para indexação/auditoria:

- `carimbo_encontrado`
- `confianca`
- `bbox`
- `regiao_fallback`
- `motivo`
- `mensagem`
- `estrategia`
- `candidatos_avaliados`
- `medico` (todos os subcampos)
- `soc_validacao` (todos os subcampos)
- `revisao_humana_recomendada`
- `carimbo_base64` ou `carimbo_url` (com retenção do PNG no storage)
- quando usar URL: guardar também o arquivo físico ou copiar para storage definitivo

## 3) Metadados Obrigatórios Do Seu Lado

Para cada tentativa, salve junto:

- `documento_id` interno
- `sha256` do arquivo original
- `pagina`
- `mime_type`
- `endpoint`
- `version` do serviço (`GET /health`)
- `processado_em` (timestamp)
- `tentativa` (1, 2, 3...)

## 4) Política De Aceite

Resultado automaticamente aceito quando:

- `carimbo_encontrado == true`
- `medico.valido == true`

Resultado com atenção/revisão quando:

- `regiao_fallback == true`
- `medico.valido == false`
- `medico.crm == null`
- `carimbo_encontrado == false`
- `revisao_humana_recomendada == true`
- `soc_validacao.nome_parecido == false` (quando SOC habilitado)
- `soc_validacao.correcao_sugerida == true` (SOC encontrou CRM alternativo compatível)

## 5) Reprocessamento Sem Perder Histórico

- Nunca sobrescreva a tentativa anterior.
- Grave cada nova execução como uma nova linha/versionamento.
- Mantenha referência `tentativa_anterior_id` para trilha.
- Se houver divergência entre tentativas, priorize revisão humana com `/debug/visualizar`.

## 6) Checklist Operacional

1. Validar `/health` antes de processar lote.
2. Processar com `/extrair-medico-gemini`.
3. Persistir resposta completa + metadados obrigatórios.
4. Aplicar política de aceite/revisão.
5. Reprocessar apenas os casos com atenção.
6. Guardar histórico de tentativas e artefatos.

## 7) Exemplo De Registro Em Banco

```json
{
  "documento_id": "aso-2026-000123",
  "sha256_arquivo": "...",
  "endpoint": "/extrair-medico-gemini",
  "version": "1.0.0",
  "tentativa": 2,
  "processado_em": "2026-04-23T10:15:22-03:00",
  "resultado": {
    "carimbo_encontrado": true,
    "confianca": 0.94,
    "estrategia": "gemini_duplo_estagio_com_fallback_opencv",
    "regiao_fallback": false,
    "revisao_humana_recomendada": false,
    "motivo": "gemini_detect_bbox",
    "medico": {
      "nome": "...",
      "crm": "...",
      "valido": true,
      "origem": "gemini_crop_fallback_opencv"
    },
    "soc_validacao": {
      "habilitada": true,
      "consultada": true,
      "nome_parecido": true,
      "similaridade_nome": 0.91,
      "motivo": "soc_ok_nome_e_crm_compativeis",
      "correcao_sugerida": false,
      "crm_sugerido": null,
      "nome_sugerido_soc": null
    }
  }
}
```
