from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://rag_user:rag_pass@localhost:5432/ragdb"

    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB
    default_chunk_size: int = 500  # in "tokens" -- see ingestion/chunking.py
    default_chunk_overlap: int = 50

    # Day 29: embeddings, generated LOCALLY via fastembed (ONNX Runtime) --
    # no API key, no network call at request time, no provider outage to
    # retry against. BAAI/bge-small is a small (~130MB), fast,
    # well-regarded general-purpose embedding model.
    #
    # Trade-off vs a hosted embedding API: no cost and no external
    # dependency at request time, at the price of somewhat lower embedding
    # quality than a large hosted model, and a one-time model download on
    # first use (cached afterward -- see the fastembed_cache volume in
    # docker-compose.yml).
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Must match the Vector(...) dimension in models.py AND the migration
    # that creates the pgvector column -- all three have to agree.
    embedding_dimensions: int = 384
    embedding_batch_size: int = 64

    # Day 31: generation. Embeddings run locally (Groq has no embeddings
    # endpoint -- confirmed the hard way in this project's Day 29 build),
    # but a fluent grounded answer needs a real chat model, so this talks
    # to an OpenAI-compatible chat API. Groq's chat endpoint is real and
    # free-tier friendly, which is why it's the default here.
    llm_api_key: str = "sk-placeholder-set-in-env"
    llm_base_url: str | None = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.1-8b-instant"
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2
    llm_retry_backoff_seconds: float = 0.5

    # Day 37 (Week 6): structured output. These are a SEPARATE retry layer
    # from llm_max_retries above: that one retries transport failures
    # (timeouts, 429s, 5xx) inside ChatOpenAI; this one re-asks the model
    # when the HTTP call succeeded but the OUTPUT broke the contract.
    # Worst-case LLM calls per request = (1 + llm_validation_retries) *
    # (1 + llm_max_retries) = 9 with defaults.
    llm_validation_retries: int = 2
    # "json_schema" (constrained decoding where the provider supports it),
    # "json_mode", or "none" (prompt-only; validation + retry does all the
    # work). Not every Groq model supports json_schema -- if a model
    # rejects it, switch this, not the code.
    llm_structured_method: str = "json_schema"

    # Week 7 (capstone): auth + tenancy.
    environment: str = "development"
    # HS256 signing key. The default exists so a fresh clone runs; in any
    # non-development environment startup REFUSES to run with it (see the
    # validator below) -- a default signing key means anyone can mint tokens.
    jwt_secret: str = "dev-only-insecure-secret-change-me-before-any-real-use"
    jwt_algorithm: str = "HS256"
    jwt_ttl_minutes: int = 60
    # scrypt cost (N). 2**14 is the interactive-login baseline; the test
    # suite lowers it because hashing speed is not what tests measure.
    password_hash_n: int = 2**14

    # Week 6 (Days 36/38/39): tool-calling agent. Every limit here exists to
    # bound a failure: a looping model (steps / tool calls), a prompt-injected
    # model probing for data (calls per step, denied calls still count), an
    # oversized tool result stuffing the context window (output chars), a
    # runaway query (row limit / timeout).
    agent_max_steps: int = 4
    agent_max_tool_calls: int = 6
    agent_max_calls_per_step: int = 3
    agent_max_tool_output_chars: int = 4000
    agent_sql_row_limit: int = 50
    agent_sql_timeout_ms: int = 2000

    @model_validator(mode="after")
    def _refuse_default_jwt_secret_outside_development(self):
        if self.environment != "development" and self.jwt_secret.startswith("dev-only-insecure"):
            raise ValueError("JWT_SECRET must be set when ENVIRONMENT is not 'development'")
        return self

    @field_validator("llm_base_url", mode="before")
    @classmethod
    def empty_string_means_unset(cls, v):
        """
        An env var set to LLM_BASE_URL= (nothing after the =) arrives here
        as an empty string, not "unset" -- normalizing "" to None means a
        blank env var falls back to OpenAI's default endpoint instead of
        failing later with a confusing "missing protocol" error.
        """
        if v == "":
            return None
        return v

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
