# Agentic RAG Pipeline — Week 6

A Dockerized document question-answering system that extends a hand-built Retrieval-Augmented Generation (RAG) pipeline with LangChain integration, validated structured responses, a tool-calling agent, streaming, and agent security controls.

Week 6 focuses on how different orchestration approaches affect answer grounding, citation behavior, latency, model usage, and refusal handling. The project retains the underlying retrieval implementation so the framework-based and agentic paths can be compared against the original pipeline.

## Project Overview

The application ingests documents, divides their text into searchable chunks, generates local embeddings, stores vectors in PostgreSQL with pgvector, retrieves relevant passages for a question, and uses a configured language model to produce a source-grounded answer.

Week 6 builds on that foundation with:

- **Day 35 — LangChain port:** exposes the existing retrieval and generation pipeline through LangChain's retriever and LCEL abstractions.
- **Day 36 — Tool-calling agent:** adds an agent loop that can call a document-search tool and use its results when answering.
- **Day 37 — Structured output:** adds schema validation and retry for predictable response structures.
- **Days 38–39 — Streaming and security:** adds a streaming agent endpoint and documents prompt-injection and scope risks.

RAG and agents are not competing alternatives. The agentic implementation still uses retrieval. The project therefore compares **direct RAG, structured RAG, and agentic RAG**.

## Main Features

- Document upload and ingestion for supported text, Markdown, and PDF files.
- Text parsing, normalization, cleaning, and sentence-aware chunking with overlap.
- Local semantic embeddings using FastEmbed and `BAAI/bge-small-en-v1.5`.
- PostgreSQL storage with pgvector vector search.
- Top-k retrieval using cosine distance and optional metadata filters.
- Source-labeled generation with numbered citations and citation-identifier validation.
- A LangChain retriever adapter and LCEL generation chain.
- An agent loop with a document-search tool and execution trace.
- Structured responses with schema validation and retry.
- Streaming agent responses.
- Evaluation scripts for retrieval quality and direct/structured/agent comparisons.
- Tests for retrieval, generation, structured output, agent behavior, security, and related API functionality.

## System Architecture

### 1. Document ingestion and indexing

```text
Document upload
    -> Parse file into page-oriented text
    -> Clean and normalize text
    -> Split into overlapping chunks
    -> Generate local embedding for each chunk
    -> Store document metadata, chunks, and vectors
       in PostgreSQL with pgvector
```

### 2. Direct RAG request

```text
Question
    -> Generate question embedding
    -> Retrieve top-k relevant chunks
    -> Apply optional metadata filters
    -> Build prompt with numbered source passages
    -> Generate answer with the configured LLM
    -> Validate citation identifiers
    -> Return answer, citations, and validation status
```

### 3. Agentic RAG request

```text
Question
    -> Agent loop evaluates the next action
    -> Agent may call document_search
    -> Application executes the tool and returns its result
    -> Agent continues or produces a final response
    -> Return answer, citations, and execution trace
```

The direct path follows a predefined retrieve-then-generate workflow. The agent path adds a model-controlled tool-selection loop. This flexibility can support more complex workflows, but it also adds model calls, latency, token usage, and additional security considerations.

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| API | FastAPI | HTTP endpoints and interactive API documentation |
| Database | PostgreSQL | Stores document and chunk metadata |
| Vector extension | pgvector | Stores and searches embedding vectors |
| Vector index | IVFFlat with cosine distance | Approximate nearest-neighbor retrieval |
| Embeddings | FastEmbed, ONNX Runtime | Local 384-dimensional embeddings |
| Embedding model | `BAAI/bge-small-en-v1.5` | Embeds document chunks and user questions |
| LLM integration | Configured chat-model API | Generates grounded answers and supports agent decisions |
| Framework integration | LangChain, LCEL | Composes retriever and model components |
| Runtime | Docker Compose | Runs the API and database services |
| Tests | pytest | Checks pipeline behavior and regression cases |
| Evaluation | Python scripts in `eval/` | Measures retrieval and compares answer modes |

Generation requires the configured LLM provider. Embeddings are generated locally and do not require a hosted embeddings endpoint. Configure provider values through `.env` using `.env.example` as a template. Never commit the real `.env` file or credentials.

## Requirements

- Docker and Docker Compose for the containerized setup.
- A configured LLM API key for endpoints that generate model responses.
- Git and a terminal for development workflows.
- Python and the dependencies in `requirements.txt` if running tests outside Docker.

## Setup and Run

Run commands from the application directory, `rag-pipeline/`, when using the original nested project layout.

1. Create a local environment file from the example:

   ```bash
   cp .env.example .env
   ```

2. Add the required database and LLM settings to `.env`. Keep this file out of version control.

3. Build and start the services:

   ```bash
   docker compose up --build
   ```

4. Open the application:

   - Chunk lab: http://127.0.0.1:8000/
   - API documentation: http://127.0.0.1:8000/docs
   - Health endpoint: `GET /ping`

FastEmbed downloads its embedding model on the first run. This initial download may take longer than subsequent runs; the model is cached in a Docker volume.

If the project has been moved to the repository root, run the same Docker commands from the directory that contains `docker-compose.yml`.

## Core Pipeline Implementation

### Parsing and cleaning

The parsing stage converts uploaded file bytes into text pages. PDF parsing preserves page numbers; plain text and Markdown are handled as single-page inputs.

The cleaning stage runs before chunking. It repairs hard line wrapping, rejoins words split by hyphenation across line breaks, normalizes Unicode with NFKC, handles compatibility ligatures and non-breaking spaces, and removes control characters. Cleaning first keeps stored offsets aligned with the text used by the rest of the pipeline.

Relevant modules:

- `app/ingestion/parsers.py`
- `app/ingestion/cleaning.py`

### Chunking

The chunking stage groups sentences, headings, and list lines into chunks within a configured token budget. The configured overlap carries part of the previous chunk into the next chunk so that information near chunk boundaries is less likely to be lost. An individual sentence that exceeds the chunk target can be split by words as a fallback.

Relevant module:

- `app/ingestion/chunking.py`

Chunk size, overlap, and retrieval top-k influence the context available to the model. The evaluation grid is used to compare combinations rather than assuming one configuration is best for every corpus.

### Local embeddings and vector storage

`app/core/embeddings.py` provides the embedding entry point. FastEmbed runs `BAAI/bge-small-en-v1.5` locally and produces 384-dimensional vectors. CPU-bound embedding work is moved to a worker thread so it does not unnecessarily block the async event loop. Embedding failures are surfaced as errors instead of treating an unsuccessfully embedded document as indexed.

PostgreSQL stores vectors in a pgvector column. The database migration enables the vector extension and creates an IVFFlat cosine-distance index. The SQLite test configuration uses a JSON variant for vector values because SQLite does not implement pgvector's vector operators.

Local embeddings avoid a separate embedding API key, embedding charges, and request-time dependence on an embedding provider. The trade-offs include an initial model download and potentially lower semantic retrieval quality than a larger hosted embedding model.

### Retrieval

`POST /retrieve` embeds the question using the same embedding model used during ingestion and returns the nearest chunks by cosine distance. Optional metadata filters such as document ID or source type narrow the candidate set before ranking.

Keeping retrieval separate from generation makes it possible to evaluate whether the correct evidence was found without conflating that result with the quality of the generated answer.

Relevant modules:

- `app/services/retrieval_service.py`
- `app/routers/retrieval.py`

#### IVFFlat search configuration

During development, retrieval returned no results against a populated PostgreSQL database without raising an error. The IVFFlat index used `lists = 100`, while the default search examined only one cluster. On a small corpus, the selected cluster could omit the nearest vectors.

The implementation addresses this by setting `ivfflat.probes = 100` locally for the transaction performing the query. This examines more clusters and can improve recall at the cost of additional search work. The appropriate setting should be benchmarked against the actual corpus size and latency requirements.

### Grounded generation and citations

`POST /ask` performs retrieval, builds a prompt containing numbered passages and source metadata, and asks the configured LLM to answer from the retrieved context. Source labels can include the filename and page number where available.

The response includes citation information and a validation result. Citation identifiers are checked against the retrieved sources; unknown identifiers are not treated as valid citations. If no passages are retrieved, the standard RAG path can return the configured no-documents response without calling the LLM.

Citation-identifier validation does not independently prove that a claim is entailed by its cited passage. That limitation is why source correctness and refusal behavior are also evaluated.

Relevant modules:

- `app/services/generation_service.py`
- `app/routers/ask.py`

## Week 6 Implementation

### Day 35 — LangChain integration

The LangChain implementation is located in `app/langchain_port/` and is exposed through `POST /ask/langchain`.

**Retriever adapter — `app/langchain_port/retriever.py`**

`PgVectorRetriever` delegates to the existing retrieval service. It does not reimplement vector search, metadata filtering, or the IVFFlat search configuration. It adapts the project's retrieval function to the interface expected by LangChain retriever components.

**LCEL chain — `app/langchain_port/chain.py`**

The generation chain composes a `ChatPromptTemplate` and a chat model using LangChain Expression Language (LCEL). It reuses the project's grounding prompt, citation pattern, and no-document response rather than maintaining a separate set of rules.

**Design outcome**

LangChain changes how the retrieval and generation components are composed, but it does not replace the underlying retrieval implementation or automatically improve retrieval quality. The framework provides useful interfaces and integration patterns while introducing dependency-managed behavior such as retries and timeouts.

The LangChain test suite replaces the module-level chain with a fake chain object. This avoids trying to monkeypatch an instance method on a Pydantic-backed `RunnableSequence`, which may reject assignment to undeclared fields.

### Day 36 — Tool-calling agent

The agent implementation adds a loop that allows the model to request an application tool, including document search. The application executes the requested tool, returns its output to the model, and continues until the model produces a final response or a stopping condition is reached.

The `POST /agent` endpoint returns the answer and associated execution details. Depending on the response, these can include citations, steps, tool calls, model-call counts, usage, budget status, and trace information.

This architecture enables the model to decide when retrieval is needed, but it also creates additional failure modes. A model can make unsupported inferences, call tools unnecessarily, or react incorrectly to malicious instructions embedded in retrieved content. Tool outputs should be handled as untrusted evidence, not as instructions that override application policy.

Relevant modules:

- `app/agent/loop.py`
- `app/agent/tools.py`
- `app/routers/agent.py`

### Day 37 — Validated structured output

The `POST /ask/structured` endpoint adds schema-based response validation and retry. Instead of relying only on free-form text, callers can request a response that follows a predictable structure suitable for downstream application code.

Schema validation checks the shape and types of a response; it does not establish that the response is factually correct. Grounding, source checks, and appropriate refusal behavior remain necessary.

Relevant modules:

- `app/langchain_port/structured.py`
- `app/routers/ask_structured.py`

### Days 38–39 — Streaming and agent security

The `POST /agent/stream` endpoint supports streaming agent responses, allowing clients to receive output incrementally rather than waiting for the complete response.

The threat model in `docs/AGENT_THREAT_MODEL.md` covers prompt-injection and scope risks associated with agent tools and retrieved content. Retrieved documents can contain adversarial text, so the system must distinguish application instructions from document content, constrain available tools, and validate requests and outputs.

The existence of a threat model documents risks and intended defenses; it is not a guarantee that every possible attack has been prevented. Security controls should be validated with tests and adversarial cases.

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /ping` | Health check |
| `POST /documents` | Upload and ingest a document |
| `POST /retrieve` | Retrieve relevant chunks without generating an answer |
| `POST /ask` | Direct RAG answer with citations |
| `POST /ask/langchain` | LangChain-based RAG answer path |
| `POST /ask/structured` | Schema-validated answer with retry |
| `POST /agent` | Agentic RAG with tool use and execution trace |
| `POST /agent/stream` | Streaming agent response |
| `/auth/*` | Authentication routes available in the current application |
| `/docs` | Interactive OpenAPI documentation |

The live OpenAPI page at `/docs` is the source of truth for exact request fields, response schemas, and authentication requirements in the checked-out version.

### Authentication

In the current project notes, protected endpoints require an `Authorization: Bearer <token>` header, with exceptions for `/ping`, authentication routes, and the chunk lab. Authenticate through the configured login endpoint and attach the returned access token to protected API requests. Check `/docs` for the exact login payload and token response.

## Evaluation

The repository includes separate evaluation paths for retrieval quality, parameter experiments, and answer-mode comparisons.

### Retrieval evaluation

`eval/eval_set.json` contains a frozen set of 30 questions against a small synthetic corpus made up of a leave policy, product FAQ, and science facts. The questions include factoid, paraphrased, ambiguous, and no-answer types.

`eval/run_eval.py` evaluates retrieval using Precision@k, Recall@k, and Mean Reciprocal Rank (MRR). Answerable questions use a target document and exact gold phrase. No-answer questions are considered separately because they do not have one correct target chunk. This design is reproducible and lightweight, but exact-span matching is a simplified relevance measure and does not capture every semantically valid answer.

### Chunk-size and top-k grid

`eval/run_grid.py` runs nine combinations:

- Chunk size: `250`, `500`, `1000`
- Top-k: `3`, `5`, `10`
- Overlap: held at 10% of chunk size

Holding the other settings constant makes the effect of chunk size and top-k easier to interpret. Results should be read alongside corpus size: when the corpus contains fewer chunks than top-k, precision values are constrained by the number of available chunks.

### Direct, structured, and agent comparison

`eval/run_compare.py` compares direct, structured, and agent modes using a sample from the evaluation set. Reported metrics include:

- Answer and source correctness
- Correct refusal rate for no-answer questions
- Invalid or unsupported citation identifiers
- Median and p95 latency
- Model calls and tool calls
- Token usage

One observed 12-question run reported 100% source correctness, 100% correct refusal, and 0% bad citations for all three modes. In that run, median latency was 3.78 seconds for direct mode, 9.19 seconds for structured mode, and 13.71 seconds for agent mode. Agent mode averaged 2.2 model calls, 1.2 tool calls, and 2,752 tokens; direct mode averaged one model call and 1,122 tokens.

These are results from a small evaluation sample and one runtime environment, not general performance guarantees. The test set includes no-answer cases, including a question about the monthly price of an Enterprise plan that is not described in the synthetic corpus. After the updated evaluation file was copied into the running container, the sampled run reported correct refusal for that case. Keep the host and container versions of `eval/eval_set.json` synchronized to reproduce the result.

### Run evaluation scripts

Run from the application directory with the services available:

```bash
docker compose exec api python eval/run_eval.py
docker compose exec api python eval/run_grid.py
docker compose exec api python eval/run_compare.py --per-type 3 --email "eval@x.test" --password "EvalPass123!"
```

If the evaluation files are not available inside the running container, copy them first:

```bash
docker compose cp eval api:/code/eval
```

Use the script's `--help` output if arguments have changed in your checkout.

## Testing

The test suite uses provider mocks where appropriate to avoid unnecessary external calls during unit tests. The repository includes tests for chunking, cleaning, embeddings, retrieval, document lifecycle, generation, LangChain, structured output, agent loops, agent security, authentication, and tenant isolation.

Typical local test command:

```bash
pip install -r requirements.txt
pytest -v
```

Run the tests from the directory containing `requirements.txt` and the test configuration. Tests that require PostgreSQL, Docker, or external provider configuration may need those dependencies configured separately.

## Repository Structure

```text
rag-pipeline/
├── app/
│   ├── agent/              # Agent loop, tools, and scope/security controls
│   ├── auth/               # Authentication and access control
│   ├── core/               # Configuration, database, embeddings, LLM client
│   ├── ingestion/          # Parsing, cleaning, and chunking
│   ├── langchain_port/     # LangChain retriever, chain, and structured output
│   ├── models/             # Database models
│   ├── routers/             # FastAPI route handlers
│   └── services/            # Ingestion, retrieval, generation, and audit logic
├── alembic/                 # Database migrations
├── docs/
│   ├── AGENT_THREAT_MODEL.md
│   └── WEEK6_AGENT_VS_RAG.md
├── eval/
│   ├── corpus/              # Synthetic evaluation documents
│   ├── eval_set.json        # Frozen evaluation questions
│   ├── run_eval.py          # Retrieval evaluation
│   ├── run_grid.py          # Chunk-size/top-k experiment
│   └── run_compare.py       # Direct/structured/agent comparison
├── tests/                   # Automated tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

The exact file layout may differ if the project has been moved from the original `rag-pipeline/` subdirectory to the repository root.

## Limitations and Engineering Considerations

- The evaluation corpus is small and synthetic; measured quality may not generalize to real-world collections.
- Exact gold-span matching uses one relevant passage per answerable question and may miss semantically correct retrievals.
- Semantic similarity indicates topical relatedness, not proof that a passage answers the question.
- Citation-ID validation catches invalid references but does not prove factual entailment.
- Agent loops can increase latency, token usage, and model calls; budgets and stopping conditions are important.
- Retrieved content can contain prompt-injection attempts. Treat it as untrusted data and validate agent behavior.
- Conflicting-document resolution is not established by the described synthetic evaluation corpus.
- Local embedding inference can be CPU-bound; large-scale ingestion and concurrent traffic require additional performance testing.
- IVFFlat configuration should be tuned against corpus size, recall, and latency requirements.
- Keep secrets in local environment files and never commit API keys, database passwords, or production tokens.

## Configuration and Secret Management

- Use `.env.example` to identify required settings.
- Store real credentials only in a local `.env` file or a secret-management system.
- Ensure `.env` is excluded by `.gitignore`.
- Do not include tokens, passwords, or API keys in logs, screenshots, evaluation artifacts, or commits.
- Review staged files with `git status` before pushing changes.

## Project Scope

Week 6 extends the existing document retrieval and answer-generation pipeline with LangChain composition, tool-calling agent behavior, validated structured responses, streaming, and security analysis. The central engineering goal is to compare these approaches while retaining explicit retrieval logic, citation checks, tests, and evaluation.

## Related Documentation

- [Agent threat model](docs/AGENT_THREAT_MODEL.md)
- [Direct, structured, and agent comparison](docs/WEEK6_AGENT_VS_RAG.md)
