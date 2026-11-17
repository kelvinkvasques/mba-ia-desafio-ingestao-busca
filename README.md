# Ingestão e Busca Semântica com LangChain e PostgreSQL

Solução do desafio Full Cycle: ingere o `document.pdf` existente e responde perguntas no terminal usando somente os trechos recuperados. Usa Python, os pacotes separados do LangChain, OpenAI e PostgreSQL com pgvector no Docker Compose.

## Arquitetura

1. `src/ingest.py`: lê o PDF com `PyPDFLoader`, rejeita ausência/erro/nenhum texto extraível, divide com `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)` e grava documentos com `OpenAIEmbeddings` e `langchain_postgres.PGVector`. Informa páginas e chunks.
2. `src/storage.py`: conexão, extensão `vector`, coleção PGVector e tabela auxiliar `rag_active_document`. Esta tabela publica a versão ativa do PDF por coleção.
3. `src/search.py`: centraliza recuperação e geração. Chama `similarity_search_with_score(query, k=10)` filtrando a versão ativa, concatena os textos e compõe `ChatPromptTemplate | ChatOpenAI | StrOutputParser`. Os scores são distâncias, não probabilidades. Não há limiar de distância; são recuperados até dez chunks conforme a quantidade disponível.
4. `src/chat.py`: loop em português; ignora entradas vazias, encerra com `sair`, `exit`, `quit`, Ctrl+C ou EOF. Cada pergunta é independente, sem histórico de conversa adicionado ao contexto.
5. `src/config.py`: carrega `.env` da raiz com `python-dotenv`, sem sobrepor variáveis já exportadas. Resolve caminhos relativos a partir da raiz do projeto.
6. `src/errors.py`: mensagens operacionais sem imprimir credenciais, URLs ou detalhes brutos das exceções.
7. `docker-compose.yml`: PostgreSQL 17 + pgvector, healthcheck, bootstrap da extensão e volume persistente `postgres_data`. Preserva os serviços, imagem, container e volume fornecidos. A porta foi restrita a `127.0.0.1`.
8. `requirements.txt`: versões exatas de todas as dependências instaladas e verificadas. `tests/test_app.py`: testes sem rede/API/banco.

O `document.pdf` original foi preservado. Para outro documento, substitua esse arquivo ou configure `PDF_PATH`. Um PDF de imagens precisa passar por OCR; este projeto não executa OCR automaticamente.

## Modelos e privacidade

A documentação oficial foi consultada durante a implementação em 16/11/2026. Foram escolhidos modelos econômicos listados na API: [text-embedding-3-small](https://developers.openai.com/api/docs/models/text-embedding-3-small) para embeddings e [gpt-4.1-mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini) para respostas. Isso confirma a presença no catálogo consultado, não o acesso de uma conta específica. Ambos podem ser alterados no `.env`. O fluxo real foi posteriormente validado pelo responsável pelo projeto, conforme registrado na seção Verificação.

O modelo padrão de chat recebe `temperature=0`. Para outro modelo que não suporte temperatura, deixe `OPENAI_CHAT_TEMPERATURE=` vazio; assim o parâmetro é omitido. Verifique os parâmetros aceitos pelo modelo antes de trocá-lo.

**O texto extraído do PDF é enviado à OpenAI para gerar embeddings.** Na busca, a pergunta também é enviada para embedding; os trechos recuperados e a pergunta são enviados ao modelo de chat. Vetores, textos e metadados ficam no banco local. As chamadas podem gerar custos; confira preços, limites e permissões no seu projeto OpenAI antes de usar documentos sensíveis.

O prompt exigido está em `src/search.py`, acompanhado de mensagem de sistema que trata PDF e pergunta como dados, incapazes de substituir as regras. Sem contexto textual, a aplicação retorna diretamente, sem chamar o chat:

> Não tenho informações necessárias para responder sua pergunta.

Com contexto, cabe ao modelo seguir as regras. O prompt **não garante eliminar alucinações ou ataques de instruções**; valide respostas contra o PDF, especialmente números. Recuperar dez trechos também não garante que toda informação existente no documento esteja entre eles.

## Preparação no macOS

- Python 3.9 ou superior; prefira Python 3.11/3.12 instalado pelo python.org ou Homebrew, compilado com OpenSSL. O Python de sistema/Command Line Tools pode usar LibreSSL e emitir aviso do urllib3.
- Docker Desktop instalado e iniciado, com o comando `docker compose` disponível no terminal.
- A imagem `pgvector/pgvector:pg17` suporta seleção nativa de arquitetura; não force `platform: linux/amd64` no Apple Silicon. Referência: [projeto pgvector](https://github.com/pgvector/pgvector#docker).
- Chave de API OpenAI com acesso aos modelos e saldo/cota de API.
- Porta local 5432 disponível (ou altere a configuração).

Na raiz deste repositório:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

Se tiver Python 3.11 instalado separadamente, use `python3.11 -m venv .venv` ao criar um ambiente novo. O lock foi produzido e testado em macOS arm64 com Python 3.9.6; usa as integrações 0.3 do LangChain já propostas pelo repositório, com APIs verificadas contra o código instalado. Não pretende selecionar automaticamente a versão mais recente de cada pacote. Em outra versão do Python, execute novamente `pip check` e os testes.

Crie o arquivo de configuração **somente se ainda não existir**:

```bash
if [ ! -e .env ]; then cp .env.example .env; fi
```

Edite `.env` no seu editor. Obtenha uma chave na área de [API keys da OpenAI](https://platform.openai.com/api-keys) e preencha `OPENAI_API_KEY` somente no `.env` local. Não publique o arquivo ou cole a chave em logs, código ou screenshots. `.env`, `.venv` e `venv` estão no `.gitignore`.

| Variável | Uso |
| --- | --- |
| `OPENAI_API_KEY` | Obrigatória, sem valor padrão real |
| `OPENAI_EMBEDDING_MODEL` | Obrigatória; exemplo: `text-embedding-3-small` |
| `OPENAI_CHAT_MODEL` | Obrigatória; exemplo: `gpt-4.1-mini` |
| `OPENAI_CHAT_TEMPERATURE` | `0` por padrão; vazio omite parâmetro |
| `DATABASE_URL` | Obrigatória; URL SQLAlchemy com driver `postgresql+psycopg` |
| `PG_VECTOR_COLLECTION_NAME` | Obrigatória; coleção lógica e chave da versão ativa |
| `PDF_PATH` | Padrão `document.pdf`; relativo à raiz ou absoluto |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | Configuração do container; padrões locais no exemplo |
| `POSTGRES_PORT` | Porta do host; padrão `5432` |

Mantenha usuário/senha/banco/porta de `DATABASE_URL` em acordo com os valores do Compose. Caracteres especiais na senha da URL precisam de percent-encoding. As credenciais `postgres/postgres` são apenas para desenvolvimento local. Alterar essas variáveis não muda credenciais de um volume já inicializado: use as credenciais existentes ou faça a alteração administrativamente no PostgreSQL, preservando o volume.

## Execução

Com o venv ativo e `.env` preenchido:

```bash
docker compose up -d
```

Confirme a saúde e o bootstrap:

```bash
docker compose ps -a
docker compose logs bootstrap_vector_ext
docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT extname FROM pg_extension WHERE extname = '\''vector'\'';"'
```

O serviço `postgres` deve estar `healthy`. O bootstrap é uma tarefa única: `Exited (0)` é sucesso. Espere o healthcheck antes da ingestão. A aplicação também cria a extensão com `IF NOT EXISTS`; o usuário do banco precisa dessa permissão.

A execução principal continua sendo:

```bash
docker compose up -d
python src/ingest.py
python src/chat.py
```

Os scripts podem ser chamados por caminho absoluto a partir de outro diretório; o `.env` e o PDF continuam sendo resolvidos pela raiz do projeto. O comando Compose deve ser executado na raiz (ou com `-f` apontando para o arquivo).

O PDF fornecido produziu **34 páginas e 67 chunks** na extração local. Exemplo de interação, após ingestão real:

```text
PERGUNTA: Qual o faturamento e o ano de fundação da Alfa Agronegócio Indústria?
RESPOSTA: ...
PERGUNTA: sair
Chat encerrado.
```

Para validar essa pergunta dentro do contexto, confira no próprio PDF: a primeira página registra faturamento de R$ 85.675.568,77 e fundação em 1931 para essa empresa. Esse exemplo é diferente da pergunta sobre faturamento validada pelo responsável, registrada abaixo.

Teste também perguntas fora do contexto:

- Qual é a capital da França?
- Quantos clientes temos em 2024?
- Você acha isso bom ou ruim?

A resposta esperada é a frase de recusa acima. Esses exemplos não constituem uma garantia sobre a LLM.

## Reingestão, atualizações e modelos

Execute `python src/ingest.py` novamente após substituir o PDF. A versão é um SHA-256 dos bytes do PDF, modelo de embeddings e configuração de divisão. Cada chunk tem ID determinístico por coleção/versão/índice e mantém metadados de origem (`source`) e página (`page`, começando em zero), além da versão e modelo.

- Mesmo PDF, modelo e divisão já ativos: nenhum chunk é inserido e nenhuma chamada de embeddings é feita.
- PDF/modelo alterado: grava todos os chunks da nova versão via PGVector e só então atualiza o ponteiro ativo em uma transação. A busca sempre filtra essa versão. Textos antigos e novos não se misturam numa resposta.
- Falha antes da ativação: a versão anterior continua disponível. Uma repetição usa UPSERT com os mesmos IDs, evitando duplicatas. Embeddings podem ser cobrados novamente em uma tentativa após falha.
- Um advisory lock por coleção serializa ingestões concorrentes. Uma consulta iniciada antes da troca pode concluir com a versão anterior inteira; as próximas usam a nova versão.
- Versões antigas e registros de tentativas interrompidas são mantidos para segurança; não há exclusão automática de coleções, chunks ou volumes. Isso consome armazenamento com o tempo.

**Sempre reingira ao trocar o modelo de embeddings, mesmo se a dimensão dos vetores permanecer igual.** Modelos diferentes usam espaços semânticos diferentes. O chat detecta a divergência entre o modelo configurado e o ativo e solicita reingestão. Trocar apenas o modelo de chat não exige reingestão.

A tabela PGVector criada por esta aplicação usa vetores sem dimensão fixa e a recuperação filtra a versão antes de ordenar por distância. Se já existir um esquema criado por outro software com dimensão fixa ou incompatível, uma mudança pode falhar: use banco/esquema compatível separado e reingira, sem apagar dados automaticamente. Uma nova coleção isola o documento lógico, mas não altera uma restrição de dimensão da tabela compartilhada preexistente.

Cada coleção gerencia um único documento ativo, inclusive se `PDF_PATH` mudar. Para manter vários PDFs consultáveis separadamente, use nomes de coleção distintos.

## Verificação

```bash
python -m pip check
python -m unittest discover -s tests -v
PYTHONPYCACHEPREFIX=/tmp/rag-python-cache python -m compileall -q src tests
docker compose config --quiet
```

Os testes não fazem chamadas de API nem exigem PostgreSQL. Cobrem montagem e ordem do contexto, `k=10`, filtro ativo, ausência de limiar, recusa sem LLM, propagação de falhas, prompt, configurações, PDF ausente/sem texto, metadados, idempotência, troca de PDF/modelo, falha/repetição de ingestão, loop do chat e estrutura do Compose. A persistência é simulada nesses testes; eles não substituem o teste real do PostgreSQL.

Verificações executadas nesta implementação:

- Ambiente `.venv` criado e dependências instaladas; `pip check`: sem incompatibilidades.
- 19 testes sem API: passaram na revisão final.
- Imports dos módulos e compilação de `src`/`tests`: passaram. A compilação precisou de cache temporário devido à restrição de escrita no cache do Python do macOS.
- PDF extraído: 34 páginas, 67 chunks, maior chunk com 999 caracteres. SHA-256 preservado: `487adc82839c525365caa5feef4c7de01be10d3375ee43c9a2dfd24331274ca8`.
- CLI e ingestão sem configuração: exibiram as variáveis obrigatórias ausentes, sem traceback/segredos.
- YAML e invariantes do Compose: passaram nos testes estruturais.
- `docker compose config --quiet` e `git diff --check`: passaram na revisão final. Inicialização, healthcheck e extensão não foram verificados isoladamente nesta revisão.
- Correção final: `OPENAI_CHAT_TEMPERATURE=` vazio omite o parâmetro na chamada de chat para modelos que o rejeitam. Erros `unsupported_value`/`unsupported_parameter` de temperatura agora exibem orientação segura para ajustar o `.env` e conferir a variável exportada no shell. Os testes cobrem a configuração vazia e a mensagem sem detalhes sensíveis do provedor.
- Validação do fluxo real informada pelo responsável pelo projeto em 16/11/2026: a pergunta sobre faturamento retornou **R$ 10.000.000,00**; a pergunta sobre clientes em 2024 retornou exatamente **“Não tenho informações necessárias para responder sua pergunta.”** A formulação completa da pergunta de faturamento não foi registrada; esse resultado não substitui os valores de outros exemplos do PDF.
- Reingestão real após troca de PDF/modelo e falhas de persistência continuam sem validação de integração registrada; esses cenários têm cobertura com persistência simulada.

## Diagnóstico

- **`docker: command not found` ou daemon indisponível:** instale/inicie Docker Desktop e verifique `docker version` e `docker compose version` em um novo terminal.
- **Postgres não saudável/conexão recusada:** veja `docker compose logs postgres`; espere o healthcheck. Se a porta estiver ocupada, ajuste `POSTGRES_PORT` e a porta em `DATABASE_URL`.
- **Falha de autenticação no banco:** confira credenciais e o volume já inicializado; mudar `.env` sozinho não altera usuários existentes.
- **Extensão ausente/permissão negada:** confira logs do bootstrap e use usuário autorizado a executar `CREATE EXTENSION IF NOT EXISTS vector`.
- **Configuração obrigatória ausente:** preencha os nomes apontados no `.env` da raiz. Variáveis exportadas no shell têm precedência.
- **Ingestão ainda não executada:** confirme coleção e banco configurados, execute a ingestão com sucesso antes do chat.
- **PDF ausente/corrompido/sem texto:** confira `PDF_PATH`, permissões e integridade; faça OCR se necessário.
- **Autenticação/permissão OpenAI:** confira chave, projeto e acesso ao modelo. Não imprima sua chave para diagnosticar.
- **Cota/limite OpenAI:** confira saldo e limites; aguarde antes de tentar novamente. A aplicação faz no máximo duas retentativas automáticas por chamada.
- **Modelo/parâmetro rejeitado:** confira os nomes no catálogo e deixe temperatura vazia se o modelo não aceitar esse parâmetro. O modelo precisa aceitar a API de chat usada pelo `ChatOpenAI` destas versões.
- **Temperatura rejeitada (`unsupported_value`/`unsupported_parameter`):** configure `OPENAI_CHAT_TEMPERATURE=` (sem valor) no `.env` e reinicie `python src/chat.py`. Se a variável estiver exportada no terminal, execute `unset OPENAI_CHAT_TEMPERATURE` antes de reiniciar, pois o shell tem precedência sobre o `.env`. Não é necessário reingerir o PDF ao mudar parâmetros do chat.
- **Aviso `NotOpenSSLWarning`/LibreSSL:** recrie o venv com Python do python.org/Homebrew compilado com OpenSSL e reinstale o lock. Não desative validação TLS.
- **Falha operacional:** aparece com prefixo `ERRO:`, nunca como frase de recusa documental. Detalhes brutos são deliberadamente omitidos para não expor segredos.

Para interromper sem apagar dados:

```bash
docker compose stop
```

Para retomar: `docker compose up -d`. `docker compose down` também preserva o volume nomeado; **não use `docker compose down -v`**, pois esse comando apaga os dados. Nenhum push ou publicação é necessário para executar o projeto.
