# Agentic RAG Pipeline — Week 6

A Dockerized document question-answering project that builds a retrieval-augmented generation (RAG) pipeline from first principles, ports the same workflow to LangChain, and extends it with validated structured output and a tool-calling agent. The project focuses on making each stage inspectable, grounding answers in source documents, validating citations, evaluating retrieval and answer behavior, and documenting agent-specific security risks.

> **Project status:** Week 5 (Days 28–34) is complete. Week 6 extends that foundation and is underway. Later authentication work is identified separately below so that work from different weeks is not misrepresented as a Week 6 deliverable.

## 1. Project overview

The project explores how to build and evaluate document QA systems without hiding the underlying mechanics behind a framework. It begins with a hand-built RAG pipeline, then adds a LangChain implementation with the same intended request/response contract, schema-validated structured output with retry, and an agent that can call a document-search tool.

**RAG and agents are not mutually exclusive alternatives.** The agentic path still relies on retrieval. The meaningful comparison in this repository is between **direct RAG, structured RAG, and agentic RAG**: the first two follow predefined answer flows, while the agent can decide when to call a search tool as part of its execution loop.

### Key capabilities

- Ingest `.txt`, `.md`, and `.pdf` documents through the API and the browser-based chunk lab.
- Parse page-oriented input, clean and normalize text, and create overlapping, sentence-aware chunks.
- Generate local 384-dimensional embeddings with FastEmbed and `BAAI/bge-small-en-v1.5`.
- Store document metadata, chunks, and vectors in PostgreSQL with pgvector.
- Retrieve nearest chunks using cosine distance and optional metadata filters.
- Generate source-grounded answers with numbered citations and citation-identifier validation.
- Expose a LangChain/LCEL version of the answer path.
- Provide schema-validated structured output with retry.
- Run a tool-calling agent with answer traces and a streaming endpoint.
- Evaluate retrieval and answer behavior with a frozen synthetic evaluation set, a parameter grid, and a comparison harness.
- Document prompt-injection and scope risks associated with tool-using agents.

## 2. Architecture

### 2.1 Indexing path

```text
Document bytes
    -> Parse into pages
    -> Clean and normalize text
    -> Split into overlapping chunks
    -> Embed each chunk locally
    -> Store document metadata, chunks, and vectors in PostgreSQL/pgvector
```

### 2.2 Standard retrieval-and-generation path

```text
Question
    -> Embed with the same local model
    -> Retrieve top-k chunks (optionally filter by metadata)
    -> Build a source-labelled prompt
    -> Call the configured language model
    -> Validate citation identifiers
    -> Return answer, citations, and validation status
```

### 2.3 Agentic path

```text
Question
    -> Agent loop
    -> Decide whether to call document_search
    -> Execute the search tool and return its result to the model
    -> Treat retrieved text as untrusted evidence
    -> Continue to a cited answer or an appropriate refusal
    -> Return answer and execution trace
```

The agent is an additional control loop around retrieval and generation; it does not replace RAG. Agentic execution introduces additional decisions and tool calls, which can help with flexible workflows but can also increase latency, token usage, and the number of failure modes.

## 3. Technology stack

| Area | Technology | Purpose |
|---|---|---|
| API | FastAPI | HTTP endpoints and interactive API documentation |
| Persistence | PostgreSQL | Document and chunk metadata |
| Vector search | pgvector, cosine distance, IVFFlat | Store and search embedding vectors |
| Local embeddings | FastEmbed, ONNX Runtime, `BAAI/bge-small-en-v1.5` | Generate 384-dimensional embeddings without a hosted embedding API |
| Answer generation | Configured chat LLM API | Produce answers from retrieved context |
| Framework port | LangChain retriever and LCEL | Compose retrieval and generation using framework abstractions |
| Structured generation | Schema validation and retry | Produce predictable output fields |
| Agent | Tool-calling loop and document-search tool | Let the model request retrieval during execution |
| Runtime | Docker Compose | Run the API and database services together |
| Evaluation | pytest and scripts under `rag-pipeline/eval/` | Regression testing, retrieval metrics, and mode comparison |

Model and provider settings are read from environment configuration. Refer to `rag-pipeline/.env.example` and the current source code for the exact variables supported by the checkout. Never commit a real `.env` file, API key, password, or access token.

## 4. Run locally

### Prerequisites

- Docker Desktop with Docker Compose
- A chat-model API key for endpoints that call a hosted language model
- Git, if cloning or contributing to the repository

### Start the application

From the repository root, enter the application folder:

```bash
cd rag-pipeline
```

Create a local `.env` from `.env.example`, fill in the required settings, and keep it out of version control. Then start the services:

```bash
docker compose up --build
```

Open these local addresses:

- **Chunk lab:** http://127.0.0.1:8000/
- **Interactive API documentation:** http://127.0.0.1:8000/docs
- **Health check:** `GET /ping`

The browser-based chunk lab is served by the API itself. It lets you paste text and experiment with chunk size and overlap, or upload a supported file to run the ingestion path.

**First-run note:** The first ingestion can take longer while FastEmbed downloads its model (approximately 130 MB). The model is cached in a Docker volume afterward. Embedding generation runs locally, but answer-generation endpoints still require the configured LLM provider.

### Run tests

The repository includes pytest tests and mocked provider calls so the test suite can avoid unnecessary real embedding inference. A typical local test setup is:

```bash
pip install -r requirements.txt
pytest -v
```

Run these commands from the application folder. Some tests or evaluation paths may require PostgreSQL/Docker or additional environment variables; check the test configuration and module-level requirements for the current checkout.

## 5. End-to-end pipeline explained

### 5.1 Parse — `ingestion/parsers.py`

The parser turns uploaded file bytes into text organised by page. PDFs retain page numbers so a retrieved passage can point back to its source page; plain text and Markdown are represented as a single page.

### 5.2 Clean — `ingestion/cleaning.py`

Text is cleaned before chunking so chunk boundaries and stored offsets refer to the text the application actually uses. Cleaning repairs hard-wrapped lines and words split by hyphenation at line breaks, applies Unicode NFKC normalization (including compatibility ligatures and non-breaking spaces), and removes control characters. Without this step, formatting noise can be embedded and passed into the model as if it were meaningful content.

### 5.3 Chunk — `ingestion/chunking.py`

The chunker packs complete sentences, headings, and list lines into chunks up to a configured token budget. It repeats the configured overlap from one chunk at the beginning of the next to reduce information loss at boundaries. If a single sentence is larger than the target size, a word-based fallback splits it.

Chunk size and overlap affect retrieval granularity and the amount of context passed to the model. Smaller chunks can be more precise but may omit broader context; larger chunks preserve more context but can introduce irrelevant text. The Week 5 grid experiment measures chunk size and top-k rather than selecting them only by intuition.

### 5.4 Embed — `core/embeddings.py`

Each chunk is embedded locally with FastEmbed using `BAAI/bge-small-en-v1.5`. The model produces 384-dimensional vectors. This module is the common embedding entry point for ingestion and query-time retrieval, which keeps model selection centralised and makes the dependency easier to mock in tests.

Embedding inference is synchronous and CPU-bound, so it is run in a worker thread using `asyncio.to_thread` rather than blocking the asynchronous event loop. Embedding failures are surfaced as an error; a document should not be reported as successfully indexed when its embeddings could not be generated.

**Why local embeddings?** Local inference avoids a separate hosted embedding API key, embedding charges, and a network dependency at request time. The trade-offs are a one-time model download, local CPU cost, and potentially lower retrieval quality than a larger hosted model.

### 5.5 Store — PostgreSQL and pgvector

Document records hold upload-level metadata; chunk records hold the extracted text, position, relevant source metadata, and embedding. The Day 29 migration enables pgvector and creates an IVFFlat cosine-distance index. In SQLite-based tests, the vector field uses a JSON representation because SQLite does not provide pgvector's vector operators.

### 5.6 Retrieve — `services/retrieval_service.py` and `routers/retrieval.py`

`POST /retrieve` embeds the question using the same local model used during ingestion, then returns the nearest chunks ranked by cosine distance. Optional filters such as `document_id` and `source_type` are applied before ranking, so results are selected from the filtered set rather than ranking everything and filtering afterward.

Retrieval deliberately remains a separate endpoint without an LLM call. This makes it possible to ask two different questions independently: **did retrieval find the right evidence?** and **did generation use that evidence correctly?** On PostgreSQL, pgvector computes vector distance in the database. In SQLite tests, the small candidate set is ranked in Python because SQLite does not support the pgvector operator.

#### A real retrieval bug and its fix

During development, retrieval returned no results against a populated PostgreSQL database, without an exception. The IVFFlat index used `lists = 100`, but a default search probed only one cluster. On a small corpus, the one cluster examined might not contain the nearest vectors. The fix sets `ivfflat.probes = 100` locally for the transaction performing the query. More probes mean more search work and potentially better recall; the right value should be tuned against corpus size and latency requirements. Scoping the setting to the transaction also prevents it leaking through a pooled connection into another request.

### 5.7 Generate answers and validate citations — `services/generation_service.py`

`POST /ask` runs the standard RAG answer flow: retrieve relevant chunks, label them with numbered references and source metadata (for example, filename and PDF page), build a prompt instructing the model to answer from the supplied evidence, and generate a cited answer.

The application validates citation identifiers against the passages retrieved for that response. It does not simply trust a model because the prompt told it to cite correctly. If retrieval returns no passages, the standard path can return the configured no-documents response without calling the LLM.

**Important limitation:** validating that `[1]` refers to a retrieved source proves that the citation identifier exists; by itself, it does not prove that every claim in the answer is entailed by that source. Claim-level grounding still requires suitable tests and evaluation.

## 6. Week 5 foundation — Days 28–34 complete

| Day | Deliverable |
|---|---|
| 28 | Document ingestion and persistence, document/chunk relationships, duplicate detection, and deletion behavior |
| 29 | Local embeddings and pgvector indexing |
| 30 | Vector retrieval, metadata filters, and retrieval-side debugging |
| 31 | Grounded generation, numbered citations, citation validation, and no-context refusal |
| 32 | Frozen 30-question evaluation set and retrieval metrics |
| 33 | Nine-run grid experiment across chunk size and top-k |
| 34 | Failure analysis and a 10-minute oral defense in [`rag-pipeline/docs/WEEK5_DEFENSE.md`](rag-pipeline/docs/WEEK5_DEFENSE.md) |

### Evaluation design

The frozen evaluation set contains 30 questions over three small synthetic documents: a leave policy, a product FAQ, and a page of science facts. It includes factoid, paraphrased, ambiguous/loosely worded, and no-answer questions. Each answerable question is associated with one target document and an exact `gold_span` phrase. This is cheap and repeatable, but it simplifies relevance to one target passage and does not capture every passage that might be a valid answer.

`rag-pipeline/eval/run_eval.py` reports Precision@k, Recall@k, and MRR overall and by question type. No-answer examples are reported separately because they do not have a single gold passage. The script also records retrieval latency and an estimated prompt token count. `rag-pipeline/eval/run_grid.py` runs nine combinations of chunk size `{250, 500, 1000}` and top-k `{3, 5, 10}`, with overlap held at 10% of chunk size.

The Week 5 defense reports one database run with Recall@5 = 1.000 across 22 answerable questions and MRR = 0.977. These values describe that small synthetic corpus and its exact-match scoring design; they are not a production quality guarantee.

## 7. Week 6 work — underway

### Day 35 — LangChain port

The code under `rag-pipeline/app/langchain_port/` exposes the pipeline through `POST /ask/langchain`, using a LangChain `BaseRetriever` adapter and LCEL composition.

- **`retriever.py`:** `PgVectorRetriever` delegates to the existing retrieval service. It does not reimplement cosine search, metadata filters, or the IVFFlat probe setting.
- **`chain.py`:** composes a `ChatPromptTemplate` and `ChatOpenAI` chain while reusing the existing grounding prompt, citation pattern, and no-document response.
- **Contract:** the endpoint is designed to use the same request/response structure as `/ask`, allowing comparisons of both paths with the same question.

LangChain changes the composition and client abstractions; it does not inherently improve retrieval quality. Retrieval logic and citation verification remain application code. Some retry and timeout behavior is handled by the client dependency rather than an application-written retry loop, a trade-off between convenience and visibility.

A testing issue was discovered because LangChain's `RunnableSequence` is Pydantic-backed: monkeypatching an undeclared instance method such as `ainvoke` can fail. The tests instead replace the module-level chain with a fake chain object.

### Day 36 — Tool-calling agent

The agent adds a decision loop around the document-search tool. The model can request a tool call; the application executes the search, returns the tool output to the model, and continues until a final answer or a stop condition. `POST /agent` returns an answer with citations and execution details such as steps, tool calls, model calls, usage, budget state, and trace information.

This makes retrieval an explicit action available to the model. It also adds failure modes such as unsupported inference, repeated tool calls, tool misuse, prompt injection in retrieved content, and increased latency or token use. Tool output must be treated as untrusted evidence, not as instructions that can override system policy.

### Day 37 — Validated structured output

`POST /ask/structured` adds schema-oriented output validation with retry. This helps callers that need predictable fields rather than free-form prose. Schema validation confirms structure, not truth, so evidence grounding and citation validation remain important.

### Days 38–39 — Streaming and threat model

`POST /agent/stream` provides a streaming agent path. The prompt-injection and scope threat model is documented in [`rag-pipeline/docs/AGENT_THREAT_MODEL.md`](rag-pipeline/docs/AGENT_THREAT_MODEL.md).

The threat model describes risks and intended controls; it should not be read as proof that all attacks have been eliminated. Retrieved documents may contain malicious instructions. The design should keep system policy separate from retrieved evidence, limit tool capabilities, validate inputs/outputs, and test refusal behavior against adversarial examples.

### Mode comparison and observed results

`rag-pipeline/eval/run_compare.py` compares direct, structured, and agent modes using a per-type sample of the evaluation set. It reports answer/source correctness, correct refusals, bad citations, latency percentiles, model calls, tool calls, and token usage.

One observed run on 12 questions reported 100% source correctness, 100% correct refusal, and 0% bad citations in all three modes. Median latency was 3.78 seconds for direct, 9.19 seconds for structured, and 13.71 seconds for agent. Agent mode averaged 2.2 model calls, 1.2 tool calls, and 2,752 tokens, compared with one model call and 1,122 tokens for direct mode. These are results from one small run; they are environment- and provider-dependent, not universal benchmarks.

The sample included three no-answer cases. The `n3` question was changed to “What is the monthly price of the Enterprise plan?” because the synthetic product FAQ does not specify an Enterprise plan price. After the updated evaluation file was copied into the container, that run showed 100% correct refusal for the sampled no-answer cases. Keep the host `rag-pipeline/eval/eval_set.json` and the container copy synchronised so the evaluation remains reproducible.

## 8. API endpoint guide

| Endpoint | Purpose |
|---|---|
| `GET /ping` | Health check |
| `POST /documents` | Upload and ingest a document |
| `POST /retrieve` | Retrieve relevant chunks without generation |
| `POST /ask` | Hand-built RAG answer with citations |
| `POST /ask/langchain` | LangChain/LCEL answer path |
| `POST /ask/structured` | Schema-validated structured answer with retry |
| `POST /agent` | Agentic document QA with tool-use trace |
| `POST /agent/stream` | Streaming agent response |
| `/auth/*` | Authentication routes added in later work |
| `/docs` | Interactive OpenAPI documentation |

The current checkout is the definitive source for endpoint availability and request schemas; use `/docs` to inspect the live API contract.

### Authentication note — later work (Week 7)

Project notes state that, beginning in Week 7, every endpoint except `/ping`, `/auth/*`, and the chunk lab requires an `Authorization: Bearer <token>` header. Older Week 5 curl examples predate this change. Log in through the configured auth endpoint, obtain the returned access token, and attach it to protected requests. Confirm the exact request fields and response shape in `/docs` for the current checkout.

## 9. Evaluation commands

Run the commands from the `rag-pipeline/` application directory with services and credentials configured:

```bash
docker compose up --build
docker compose exec api python eval/run_eval.py
docker compose exec api python eval/run_grid.py
docker compose exec api python eval/run_compare.py --per-type 3 --email "eval@x.test" --password "EvalPass123!"
```

If the evaluation folder is not mounted into the running container, copy it before running a script:

```bash
docker compose cp eval api:/code/eval
```

Evaluation arguments can evolve with the code; use `--help` if a command differs in your checkout. Use test-only credentials for the evaluation account and never reuse production passwords.

## 10. Known limitations and engineering trade-offs

- **Small synthetic corpus:** useful for repeatable development tests, but not representative of real-world document diversity or scale.
- **Simplified labels:** answerable questions use one target passage and exact substring matching, which may miss semantically correct evidence.
- **Similarity is not answerability:** related passages can rank highly even when they do not answer the specific question. No-answer behavior needs explicit evaluation and abstention controls.
- **Citation validation is bounded:** checking citation identifiers does not independently establish claim entailment.
- **Agent cost:** extra model and tool calls can increase latency and token usage; tool budgets and stop conditions are important.
- **Conflicting documents:** the Week 5 corpus did not test conflicting-source behavior. The Week 5 defense identifies this as a gap, not a completed capability.
- **CPU-bound local embeddings:** large ingestion volumes and concurrent workloads require performance measurement and may need background workers or a queue.
- **Vector index tuning:** IVFFlat settings should be benchmarked against representative data volume and latency targets.
- **Secrets management:** real credentials must remain local and must not be committed.

## 11. Repository map

The repository currently uses a top-level project folder and an inner application folder:

- `README.md` — this project overview
- `rag-pipeline/app/` — API, structured output, agent loop, and LangChain port
- `rag-pipeline/ingestion/` — parsing, cleaning, and chunking
- `rag-pipeline/core/` — settings, embedding client, and LLM client
- `rag-pipeline/services/` — ingestion, retrieval, and generation services
- `rag-pipeline/routers/` — API route definitions
- `rag-pipeline/models/` and `rag-pipeline/alembic/` — data models and database migrations
- `rag-pipeline/eval/` — frozen question set, synthetic corpus, and evaluation/grid/comparison scripts
- `rag-pipeline/tests/` — test suite
- [`rag-pipeline/docs/WEEK5_DEFENSE.md`](rag-pipeline/docs/WEEK5_DEFENSE.md) — Week 5 failure analysis and oral defense
- [`rag-pipeline/docs/AGENT_THREAT_MODEL.md`](rag-pipeline/docs/AGENT_THREAT_MODEL.md) — agent prompt-injection and scope threat model
- [`rag-pipeline/docs/WEEK6_AGENT_VS_RAG.md`](rag-pipeline/docs/WEEK6_AGENT_VS_RAG.md) — direct/structured/agent comparison report

## 12. Further reading

- [Week 5 failure analysis and oral defense](rag-pipeline/docs/WEEK5_DEFENSE.md)
- [Agent threat model](rag-pipeline/docs/AGENT_THREAT_MODEL.md)
- [Week 6 mode comparison](rag-pipeline/docs/WEEK6_AGENT_VS_RAG.md)

## License

No license file was specified in the project notes used to prepare this README. Add an explicit license if you intend to distribute the repository as open-source software.
